# 專案設定與時間換算：從 .env 讀資料庫、監控資料、控制參數等設定；
# 並提供 Unix 秒與台灣時間互換（內部一律用 Unix 秒，顯示、分組時才換台灣時間）
from pathlib import Path

import pandas as pd
from pydantic_settings import BaseSettings, SettingsConfigDict

# 專案根目錄的 .env
ENV_FILE = Path(__file__).resolve().parent / ".env"


# 專案設定：環境變數優先於 .env；開發機與案場只要改 .env，不用改程式
class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # PostgreSQL
    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: str
    postgres_db: str

    # 氣象（開發機分析用）
    cwa_crawler_base_url: str
    cwa_history_endpoint: str
    cwa_realtime_station_id: str
    station_lat: float
    station_lon: float
    timezone: str

    # 監控資料 VM
    iot_api: str
    # 點位，逗號分隔
    iot_point_ids: str
    # 要抓的監測項目（每個點位都抓這些），逗號分隔
    iot_metric_keys: str
    # 取樣間隔 (s)
    iot_step_seconds: int
    # 首次回補天數
    iot_first_days: int

    # 控制建議參數
    # 出水下限 (°C)
    control_t_min: float
    # 出水上限 (°C)
    control_t_max: float
    # 回水上限 (°C)
    control_t_return: float
    # 並聯門檻 (RT)
    control_parallel_above: int
    # 加2度呈現
    control_t_offset: int
    # 尖峰比例：段內超過並聯門檻的部分加總 > 並聯門檻 × 這個比例時，該段改依最高那小時
    control_peak_excess: float
    # 台數切換後至少維持幾小時 (h)
    control_min_stage_hours: int
    # 控制依據百分位
    control_load_pct: float

# 讀取專案設定（載入時讀一次，其他檔案用 settings.settings）
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


# ---------- 時間轉換：內部一律 Unix 秒，顯示、分組時才轉台灣時間 ----------


# Unix 秒 → 台灣時間：顯示或依日期、小時分組時用
def to_local(timestamp: int | float) -> pd.Timestamp:
    utc_time = pd.Timestamp(timestamp, unit="s", tz="UTC")
    return utc_time.tz_convert(settings.timezone)


# 整串 Unix 秒 → 台灣時間 index；naive=True 時去掉時區（畫圖用）
def to_local_index(index, naive: bool = False) -> pd.DatetimeIndex:
    utc_index = pd.to_datetime(index, unit="s", utc=True)
    local_index = utc_index.tz_convert(settings.timezone)
    if naive:
        return local_index.tz_localize(None)
    return local_index


# 某個台灣日期的某個整點：列出上班時段各整點、或算一天的起訖時間用
def at_hour(day, hour: int) -> pd.Timestamp:
    day_start = pd.Timestamp(day).tz_localize(settings.timezone)
    return day_start + pd.Timedelta(hours=hour)
