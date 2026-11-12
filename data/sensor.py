# 監控資料：從 VM（監控資料庫）抓主機資料存入資料庫，並提供讀取主機逐時資料與運轉判定
import asyncio
import time

import httpx
import pandas as pd

import settings
from db import db_interaction, db_schema
from model import rules

BATCH_SIZE = 5000  # 每累積幾筆寫一次 DB
ONE_DAY = 24 * 60 * 60  # query_range 有點數上限，每次查一天
ONE_HOUR = 60 * 60


# 組出查詢某點位某監測項目的 PromQL 字串（精確比對，只抓需要的項目）
def point_query(point_id: str, metric_key: str) -> str:
    return f'metric{{id="{point_id}",key="{metric_key}"}}'


# 向 VM 查詢一段時間的資料（每 iot_step_seconds 秒一筆）
async def get_history(query: str, start_time: int, end_time: int) -> dict:

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.get(
            f"{settings.settings.iot_api}/query_range",
            params={
                "query": query,
                "start": start_time,
                "end": end_time,
                "step": f"{settings.settings.iot_step_seconds}s",
            },
        )

    response.raise_for_status()
    return response.json()


# 決定這次從哪裡開始抓（每個點位、項目各自接續）：接續最後一筆並往回重疊 1 小時補抓延遲資料；
# 第一次抓（含之後新增的項目）則回補 IOT_FIRST_DAYS 天
async def get_start_time(
    db_manager: db_interaction.DBManager,
    point_id: str,
    metric_key: str,
    now_timestamp: int,
) -> int:

    last_time = await db_manager.get_last_sensor_data_time(point_id, metric_key)

    if last_time is None:
        return now_timestamp - settings.settings.iot_first_days * ONE_DAY

    # 重疊 1 小時以補抓延遲資料
    return last_time - ONE_HOUR


# 抓取單一點位單一項目的新資料並分批寫入資料庫（每次查一天，避開 VM 點數上限）
async def update_point(
    db_manager: db_interaction.DBManager,
    point_id: str,
    metric_key: str,
) -> None:

    now_timestamp = int(time.time())
    start_time = await get_start_time(db_manager, point_id, metric_key, now_timestamp)
    query = point_query(point_id, metric_key)

    sensor_data_batch = []
    total = 0

    for day_start in range(start_time, now_timestamp, ONE_DAY):
        day_end = min(day_start + ONE_DAY, now_timestamp)

        response = await get_history(
            query=query,
            start_time=day_start,
            end_time=day_end,
        )

        # 回傳格式：{"metric": {"id", "key"}, "values": [[timestamp, 數值], ...]}
        for metric_data in response["data"]["result"]:
            metric_id = metric_data["metric"]["id"]
            metric_key = metric_data["metric"]["key"]

            for timestamp, value in metric_data["values"]:
                if value == "NaN":
                    continue

                sensor_data_batch.append(
                    db_schema.SensorData(
                        timestamp=int(timestamp),
                        metric_id=metric_id,
                        metric_key=metric_key,
                        value=float(value),
                    )
                )

                # 累積到一批就寫入 DB
                if len(sensor_data_batch) >= BATCH_SIZE:
                    await db_manager.add_sensor_data(sensor_data_batch)
                    total += len(sensor_data_batch)
                    sensor_data_batch.clear()

    # 寫入剩下不足一批的資料
    if sensor_data_batch:
        await db_manager.add_sensor_data(sensor_data_batch)
        total += len(sensor_data_batch)

    print(f">>> {point_id} {metric_key} 處理完成，寫入 {total} 筆")


# 抓取 .env 設定的所有點位 × 監測項目（每小時排程執行），模型訓練與預測都靠這些資料
async def update() -> str:

    if not settings.settings.iot_point_ids:
        return "未設定 IOT_POINT_IDS，略過"
    if not settings.settings.iot_metric_keys:
        return "未設定 IOT_METRIC_KEYS，略過"

    point_ids = settings.settings.iot_point_ids.split(",")
    metric_keys = settings.settings.iot_metric_keys.split(",")

    async with db_schema.async_session() as session:
        db_manager = db_interaction.DBManager(session)

        for point_id in point_ids:
            for metric_key in metric_keys:
                await update_point(db_manager, point_id, metric_key)

    return (
        f"VM 資料抓取完成，共 {len(point_ids)} 個點位 × {len(metric_keys)} 個項目"
    )


# ---------- 讀取主機監控資料 ----------

# 監測項目代號 → 欄位
METRIC_ITEMS = {
    "電錶即時功率(通訊)": "kW",
    "冰水流量": "FLOW",
    "冰水流量進水溫度": "T_CHWS",
    "冰水流量回水溫度": "T_CHWR",
    "冰水流量冷凍噸": "Q_COOL",
}
POWER_KEY = "電錶即時功率(通訊)"
FLOW_KEY = "冰水流量"
RUNNING_POWER_KW = 20  # 運轉判定：功率 (kW)
RUNNING_FLOW_LPM = 100  # 運轉判定：冰水流量 (LPM)


# 運轉判定：功率與冰水流量都超過門檻才算在運轉，排除停機時的小讀值
def is_running(df: pd.DataFrame) -> pd.Series:
    return (df["kW"] > RUNNING_POWER_KW) & (df["FLOW"] > RUNNING_FLOW_LPM)


# 某監測項目的逐時平均（在資料庫端計算，比讀每分鐘資料快）；index 為整點 Unix 秒
async def get_hourly_mean(
    db_manager: db_interaction.DBManager,
    metric_id: str,
    metric_key: str,
    start_time: int | None,
    end_time: int | None,
) -> pd.Series:
    rows = await db_manager.get_sensor_hourly_mean(
        metric_id,
        metric_key,
        start_time,
        end_time,
    )
    return pd.Series(dict(rows), dtype=float)


# 兩台主機功率與冰水流量逐時平均：{主機名稱: 有 kW、FLOW 欄的表（index 為整點 Unix 秒）}；
# 判斷開機小時與算流量典型值用
async def get_chiller_hourly_power_flow(
    start_time: int | None,
    end_time: int | None,
) -> dict[str, pd.DataFrame]:
    hourly_by_chiller = {}
    async with db_schema.async_session() as session:
        db_manager = db_interaction.DBManager(session)
        for chiller_name, chiller in rules.CHILLERS.items():
            power = await get_hourly_mean(
                db_manager,
                chiller["id"],
                POWER_KEY,
                start_time,
                end_time,
            )
            flow = await get_hourly_mean(
                db_manager,
                chiller["id"],
                FLOW_KEY,
                start_time,
                end_time,
            )
            hourly_by_chiller[chiller_name] = pd.DataFrame(
                {"kW": power, "FLOW": flow},
                dtype=float,
            )
    return hourly_by_chiller


# 讀取單一監測項目的每分鐘資料（index 為台灣時間），分析用
async def get_metric_data(
    metric_id: str,
    metric_key: str,
    start_time: int | None = None,
    end_time: int | None = None,
) -> pd.Series:
    async with db_schema.async_session() as session:
        db_manager = db_interaction.DBManager(session)
        records = await db_manager.get_sensor_data(
            metric_id=metric_id,
            metric_key=metric_key,
            start_time=start_time,
            end_time=end_time,
        )

    # 以 timestamp 為索引，再換成台灣時間
    values = {record.timestamp: record.value for record in records}
    metric_series = pd.Series(values, dtype=float)
    metric_series.index = settings.to_local_index(metric_series.index)
    return metric_series


# 讀取單台主機全部監測項目（功率、流量、出回水溫度、冷凍噸）成一張表，分析用
async def load_chiller(
    metric_id: str,
    start_time: int | None = None,
    end_time: int | None = None,
) -> pd.DataFrame:
    columns = {}
    for metric_key, column_name in METRIC_ITEMS.items():
        columns[column_name] = await get_metric_data(
            metric_id=metric_id,
            metric_key=metric_key,
            start_time=start_time,
            end_time=end_time,
        )
    return pd.DataFrame(columns).sort_index()


if __name__ == "__main__":
    print(asyncio.run(update()))
