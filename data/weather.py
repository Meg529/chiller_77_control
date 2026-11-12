# 氣象資料（只給開發機分析用）：逐時觀測的抓取與讀取，以及濕空氣、太陽高度角等氣象特徵
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd

import settings
from db import db_interaction, db_schema

# 觀測數值欄位
OBS_COLS = [
    "temperature",
    "humidity",
    "pressure",
    "precipitation",
    "wind_speed",
    "dew_point",
    "solar_radiation",
    "wind_direction",
    "cloud_amount",
    "cloud_height",
    "visibility",
]
WEATHER_COLUMNS = ["timestamp", "station_id", "station_name", *OBS_COLS]

ONE_DAY = 24 * 60 * 60


# ---------- 抓 API ----------


# 呼叫氣象 API 並回傳 JSON
async def _get(endpoint: str, params: dict) -> list:
    url = settings.settings.cwa_crawler_base_url + endpoint
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


# :59 分的時間（如 23:59）進到下一個整點（隔日 00:00），對齊逐時資料
def _fix_midnight(timestamp: int) -> int:
    if timestamp % 3_600 == 3_540:
        return timestamp + 60
    return timestamp


# 從氣象 API 抓指定測站一段時間的逐時觀測，轉成資料表欄位格式
async def fetch_history(
    start_time: int,
    end_time: int,
    station_id: str | None = None,
) -> list[dict]:
    if not station_id:
        station_id = settings.settings.cwa_realtime_station_id

    params = {
        "station_id": station_id,
        "start_time": start_time,
        "end_time": end_time,
    }
    records = await _get(settings.settings.cwa_history_endpoint, params)

    # 轉 DB 格式
    rows = []
    for record in records:
        row = {
            "timestamp": _fix_midnight(record["time"]),
            "station_id": station_id,
            "station_name": record.get("station_name"),
        }
        for column in OBS_COLS:
            row[column] = record.get(column)
        rows.append(row)
    return rows


# ---------- 存 DB ----------


# 抓逐時觀測並存入 weather_history（重複的略過）
async def save_history(start_time: int, end_time: int) -> str:
    rows = await fetch_history(start_time, end_time)
    async with db_schema.async_session() as session:
        await db_interaction.DBManager(session).add_weather_history(rows)
    return f"已存入 {len(rows)} 筆逐時氣象資料"


# 同步最近 30 天的逐時觀測（逐日抓，單日失敗不影響其他日，下次再補）
async def update() -> str:
    now_datetime = datetime.now(ZoneInfo(settings.settings.timezone))
    now = int(now_datetime.timestamp())

    # 回補起點：30 天前的 00:00
    backfill_start_datetime = (now_datetime - timedelta(days=30)).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    backfill_start = int(backfill_start_datetime.timestamp())

    # 逐日回補，單日失敗不影響其他日
    failed_days = 0
    for day_start in range(backfill_start, now, ONE_DAY):
        day_end = min(day_start + ONE_DAY, now)
        try:
            await save_history(day_start, day_end)
        except Exception as error:
            failed_days += 1
            print(f">>> 歷史氣象同步失敗（{day_start}–{day_end}）：{error}")
    if failed_days:
        print(f">>> 歷史氣象有 {failed_days} 個日期待下次重試")
    return "每日氣象資料同步完成"


# ---------- 從 DB 讀 ----------


# 補齊缺漏的整點：數值內插、風向沿用前一小時，分析時不會因缺資料斷掉
def _interpolate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # 補上缺漏的整點
    first_hour = int(df["timestamp"].min())
    last_hour = int(df["timestamp"].max())
    hours = range(first_hour, last_hour + 1, 3_600)
    df = df.set_index("timestamp").reindex(hours)

    numeric_columns = [column for column in OBS_COLS if column != "wind_direction"]
    df[numeric_columns] = (
        df[numeric_columns].astype(float).interpolate(limit_area="inside")
    )
    df["wind_direction"] = df["wind_direction"].astype(float).ffill(limit_area="inside")
    df[["station_id", "station_name"]] = df[["station_id", "station_name"]].ffill()
    return df.rename_axis("timestamp").reset_index()


# 觀測紀錄轉成逐時表：竹東站缺資料時各欄記 0，視為缺值後內插；露點由溫濕度換算
def _records_to_dataframe(records: list) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {column: getattr(record, column) for column in WEATHER_COLUMNS}
        rows.append(row)
    df = pd.DataFrame(rows, columns=WEATHER_COLUMNS)

    # 竹東站缺資料時各欄記 0，視為缺值
    no_pressure = (df["pressure"] == 0) | df["pressure"].isna()
    df = df[~(no_pressure & (df["humidity"] == 0))].reset_index(drop=True)
    df = _interpolate_hourly(df)

    # 露點由溫濕度換算
    calculated_dew_point = pd.Series(
        dew_point(df["temperature"], df["humidity"]),
        index=df.index,
    )
    df["dew_point"] = df["dew_point"].astype(float).fillna(calculated_dew_point)
    return df


# 從資料庫讀逐時觀測（缺值內插），分析 notebook 用
async def load_history(
    start_time: int,
    end_time: int,
    station_id: str | None = None,
) -> pd.DataFrame:
    if not station_id:
        station_id = settings.settings.cwa_realtime_station_id

    async with db_schema.async_session() as session:
        db_manager = db_interaction.DBManager(session)
        records = await db_manager.get_weather_history(
            station_id=station_id,
            start_time=start_time,
            end_time=end_time,
        )
    return _records_to_dataframe(records)


# ---------- 氣象特徵：濕空氣、太陽高度角 ----------

BASE_CDD_TEMP = 24.0  # 冷房度時基準溫度 (°C)
# 濕空氣計算用固定氣壓 (hPa)，竹東站 2026-09~10 平均
STATION_PRESSURE_HPA = 996.6


# ---------- 濕空氣 ----------
# 公式內沿用文獻符號：t = 溫度 (°C)、t_k = 絕對溫度 (K)、rh = 相對濕度、
# p = 大氣壓力 (kPa)、p_vap = 水氣壓、w = 含濕量、v = 風速 (m/s)


# 濕球溫度 (°C)，Stull (2011) 近似式：只有溫濕度也能估算
def wet_bulb_temp(temp_c, rh_percent) -> np.ndarray:
    t = np.asarray(temp_c, dtype=float)
    rh = np.asarray(rh_percent, dtype=float)  # 百分比 (0~100)
    return (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


# 飽和水氣壓 (kPa)：其他濕空氣計算的基礎
def saturation_vapor_pressure(temp_c) -> np.ndarray:
    t = np.asarray(temp_c, dtype=float)
    return 0.61094 * np.exp(17.625 * t / (t + 243.04))


# 實際水氣壓 (kPa) = 相對濕度 × 飽和水氣壓
def _vapor_kpa(temp_c, rh_percent) -> np.ndarray:
    return np.asarray(rh_percent, dtype=float) / 100 * saturation_vapor_pressure(temp_c)


# 露點溫度 (°C)，Magnus 公式：竹東站沒有露點，由溫濕度換算
def dew_point(temp_c, rh_percent) -> np.ndarray:
    t = np.asarray(temp_c, dtype=float)
    rh = np.asarray(rh_percent, dtype=float) / 100  # 比例 (0~1)
    gamma = np.log(rh) + 17.625 * t / (t + 243.04)
    return 243.04 * gamma / (17.625 - gamma)


# 含濕量 (kg 水氣 / kg 乾空氣)：算焓值用
def humidity_ratio(temp_c, rh_percent, pressure_hpa) -> np.ndarray:
    p = np.asarray(pressure_hpa, dtype=float) / 10  # hPa → kPa
    p_vap = _vapor_kpa(temp_c, rh_percent)
    return 0.62198 * p_vap / (p - p_vap)


# 濕空氣焓值 (kJ/kg 乾空氣)：同時反映溫度與濕度的空氣熱量
def enthalpy(temp_c, rh_percent, pressure_hpa) -> np.ndarray:
    t = np.asarray(temp_c, dtype=float)
    w = humidity_ratio(t, rh_percent, pressure_hpa)
    return 1.006 * t + w * (2501 + 1.86 * t)


# 水氣壓差 VPD (kPa)：空氣離飽和還差多少
def vapor_pressure_deficit(temp_c, rh_percent) -> np.ndarray:
    return saturation_vapor_pressure(temp_c) - _vapor_kpa(temp_c, rh_percent)


# 濕空氣密度 (kg/m³)：候選氣象特徵之一
def air_density(temp_c, rh_percent, pressure_hpa) -> np.ndarray:
    t_k = np.asarray(temp_c, dtype=float) + 273.15
    p = np.asarray(pressure_hpa, dtype=float) / 10  # hPa → kPa
    p_vap = _vapor_kpa(temp_c, rh_percent)

    # 氣體常數 kJ/(kg·K)
    r_dry = 0.28705  # 乾空氣
    r_vap = 0.461495  # 水氣
    return (p - p_vap) / (r_dry * t_k) + p_vap / (r_vap * t_k)


# 溫濕度指數 THI：反映悶熱程度的候選氣象特徵
def temperature_humidity_index(temp_c, rh_percent) -> np.ndarray:
    t = np.asarray(temp_c, dtype=float)
    rh = np.asarray(rh_percent, dtype=float) / 100  # 比例 (0~1)
    return t - 0.55 * (1.0 - rh) * (t - 14.5)


# 體感溫度 (°C)，澳洲氣象局公式：同時考慮溫度、濕度與風速
def apparent_temp(temp_c, rh_percent, wind_speed) -> np.ndarray:
    t = np.asarray(temp_c, dtype=float)
    rh = np.asarray(rh_percent, dtype=float) / 100  # 比例 (0~1)
    v = np.asarray(wind_speed, dtype=float)
    p_vap = rh * 6.105 * np.exp(17.27 * t / (237.7 + t))  # hPa
    return t + 0.33 * p_vap - 0.70 * v - 4.00


# 冷房度時 (°C·h)：氣溫超過基準溫度的度數，常用來代表冷氣需求
def cooling_degree_hour(temp_c, base_temp: float = BASE_CDD_TEMP) -> np.ndarray:
    return np.maximum(np.asarray(temp_c, dtype=float) - base_temp, 0.0)


# ---------- 太陽高度角 ----------


# 太陽高度角 (°)，dates 為台灣時間：竹東站沒有日射量，用它代表日照強弱
def solar_elevation(dates: pd.DatetimeIndex) -> np.ndarray:
    config = settings.settings
    day_of_year = np.asarray(dates.dayofyear, dtype=float)
    hour = np.asarray(dates.hour + dates.minute / 60, dtype=float)

    # 太陽赤緯 (°)
    declination = 23.45 * np.sin(np.deg2rad(360 * (284 + day_of_year) / 365))
    # 經度修正（基準 120°E）
    solar_time = hour + (config.station_lon - 120.0) / 15.0
    hour_angle = 15.0 * (solar_time - 12.0)

    latitude_rad = np.deg2rad(config.station_lat)
    declination_rad = np.deg2rad(declination)
    hour_angle_rad = np.deg2rad(hour_angle)
    sin_elevation = np.sin(latitude_rad) * np.sin(declination_rad) + np.cos(
        latitude_rad
    ) * np.cos(declination_rad) * np.cos(hour_angle_rad)
    return np.degrees(np.arcsin(np.clip(sin_elevation, -1, 1)))


# 加入太陽高度角欄 SOLAR_ELEV
def add_solar_elevation(df: pd.DataFrame) -> pd.DataFrame:
    df["SOLAR_ELEV"] = solar_elevation(df.index)
    return df


# ---------- 逐時特徵表 ----------


# 觀測資料轉成逐時平均（台灣時間 index），才能和主機逐時資料對齊
def to_hourly(df_weather: pd.DataFrame, timezone: str | None = None) -> pd.DataFrame:
    if timezone is None:
        timezone = settings.settings.timezone

    df = df_weather.copy()
    utc_time = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df.index = utc_time.dt.tz_convert(timezone)
    df = df.drop(columns=["timestamp", "station_id", "station_name"], errors="ignore")
    return df.sort_index().resample("1h").mean()


# 加入濕空氣特徵（濕球、焓值、含濕量、VPD、密度、THI、體感溫度、冷房度時），氣壓用固定值
def add_humid_air_features(df: pd.DataFrame) -> pd.DataFrame:
    temperature = df["TEMPERATURE"]
    humidity = df["HUMIDITY"]
    pressure_hpa = STATION_PRESSURE_HPA

    df["WET_BULB"] = wet_bulb_temp(temperature, humidity)
    df["ENTHALPY"] = enthalpy(temperature, humidity, pressure_hpa)
    df["HUM_RATIO"] = humidity_ratio(temperature, humidity, pressure_hpa)
    df["VPD"] = vapor_pressure_deficit(temperature, humidity)
    df["AIR_DENSITY"] = air_density(temperature, humidity, pressure_hpa)
    df["THI"] = temperature_humidity_index(temperature, humidity)
    df["APPARENT_TEMP"] = apparent_temp(temperature, humidity, df["WIND_SPEED"])
    df["CDD_H"] = cooling_degree_hour(temperature)
    return df
