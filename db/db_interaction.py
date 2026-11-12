# 資料庫讀寫：所有查詢與寫入集中在 DBManager，其他程式不直接寫 SQL
import time

from sqlalchemy import delete, func
from sqlalchemy.dialects.postgresql import insert
from sqlmodel import select

from db import db_schema


# 資料表欄位名稱（不含 id）：組寫入資料用
def _columns(model) -> list[str]:
    return [column for column in model.model_fields if column != "id"]


# 組 INSERT 語法，遇到重複資料略過：重抓重疊時段也不會重複寫入
def _insert_ignore(model, values: list[dict], constraint: str):
    statement = insert(model).values(values)
    return statement.on_conflict_do_nothing(constraint=constraint)


# 依點位、監測項目加上篩選條件（沒給就不篩）
def _where_metric(
    statement,
    model,
    metric_id: str | None = None,
    metric_key: str | None = None,
):
    if metric_id:
        statement = statement.where(model.metric_id == metric_id)
    if metric_key:
        statement = statement.where(model.metric_key == metric_key)
    return statement


# 資料庫讀寫（所有查詢都透過同一個 session）
class DBManager:

    # 記下要用的資料庫 session
    def __init__(self, session):
        self.session = session

    # 執行寫入並 commit
    async def _execute(self, statement) -> None:
        await self.session.execute(statement)
        await self.session.commit()

    # 執行查詢，回傳全部結果
    async def _all(self, statement) -> list:
        result = await self.session.execute(statement)
        return result.scalars().all()

    # 執行查詢，回傳一筆結果（沒有則 None）
    async def _one_or_none(self, statement):
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    # 新增或更新一筆，並取回資料庫產生的欄位（如 id）
    async def _save(self, record):
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    # ============================================================
    # SensorData
    # ============================================================

    # 新增監控資料（重複略過）
    async def add_sensor_data(self, records: list) -> None:
        if not records:
            return
        values = [record.model_dump(exclude={"id"}) for record in records]
        statement = _insert_ignore(
            model=db_schema.SensorData,
            values=values,
            constraint="uq_sensor_data_metric",
        )
        await self._execute(statement)

    # 查詢監控資料 [start, end)：分析讀每分鐘資料用
    async def get_sensor_data(
        self,
        metric_id: str | None = None,
        metric_key: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list:
        model = db_schema.SensorData
        statement = _where_metric(
            statement=select(model),
            model=model,
            metric_id=metric_id,
            metric_key=metric_key,
        )
        if start_time is not None:
            statement = statement.where(model.timestamp >= start_time)
        if end_time is not None:
            statement = statement.where(model.timestamp < end_time)
        return await self._all(statement.order_by(model.timestamp))

    # 在資料庫端算逐時平均 [(整點, 平均值)]，區間 [start, end)：資料量大，交給資料庫算較快
    async def get_sensor_hourly_mean(
        self,
        metric_id: str,
        metric_key: str,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list:
        model = db_schema.SensorData
        hour_timestamp = model.timestamp - model.timestamp % 3_600  # 捨去到整點
        statement = _where_metric(
            statement=select(hour_timestamp, func.avg(model.value)),
            model=model,
            metric_id=metric_id,
            metric_key=metric_key,
        )
        if start_time is not None:
            statement = statement.where(model.timestamp >= start_time)
        if end_time is not None:
            statement = statement.where(model.timestamp < end_time)

        statement = statement.group_by(hour_timestamp).order_by(hour_timestamp)
        result = await self.session.execute(statement)
        return result.all()

    # 某點位最後一筆資料時間：決定下次從哪裡接著抓
    async def get_last_sensor_data_time(
        self,
        metric_id: str,
        metric_key: str,
    ) -> int | None:
        model = db_schema.SensorData
        statement = select(func.max(model.timestamp)).where(
            model.metric_id == metric_id,
            model.metric_key == metric_key,
        )
        return await self._one_or_none(statement)

    # ============================================================
    # WeatherHistory
    # ============================================================

    # 新增逐時觀測（重複略過）
    async def add_weather_history(self, rows: list[dict]) -> None:
        if not rows:
            return
        model = db_schema.WeatherHistory
        columns = _columns(model)
        values = [{column: row.get(column) for column in columns} for row in rows]
        await self._execute(_insert_ignore(model, values, "uq_weather_history"))

    # 查詢某測站逐時觀測 [start, end)
    async def get_weather_history(
        self,
        station_id: str,
        start_time: int,
        end_time: int,
    ) -> list:
        model = db_schema.WeatherHistory
        statement = select(model).where(
            model.station_id == station_id,
            model.timestamp >= start_time,
            model.timestamp < end_time,
        )
        return await self._all(statement.order_by(model.timestamp))

    # ============================================================
    # ModelRegistry
    # ============================================================

    # 存入 model_registry（同 model_id 覆蓋），上線預測讀這一筆
    async def save_model(
        self,
        model_id: str,
        model_type: str,
        model_obj,
        metrics: dict,
        features: dict,
    ):
        entry = await self.get_model(model_id)
        if entry is None:
            entry = db_schema.ModelRegistry(model_id=model_id)

        entry.model_type = model_type
        entry.features = features
        entry.metrics = metrics
        entry.set_model(model_obj)
        entry.created_at = time.time()  # 覆蓋時也要更新成這次的訓練時間
        return await self._save(entry)

    # 讀取某 model_id 的上線模型（沒有則 None）
    async def get_model(self, model_id: str):
        model = db_schema.ModelRegistry
        statement = select(model).where(model.model_id == model_id)
        return await self._one_or_none(statement)

    # ============================================================
    # ModelHistory
    # ============================================================

    # 把上線模型備份成歷史版本，並記下訓練資料起訖時間
    async def add_model_history(self, entry, train_start: int, train_end: int):
        history = db_schema.ModelHistory(
            model_id=entry.model_id,
            model_type=entry.model_type,
            features=entry.features,
            model_binary=entry.model_binary,
            metrics=entry.metrics,
            train_start=train_start,
            train_end=train_end,
        )
        return await self._save(history)

    # 刪除過期歷史版本，避免資料表無限長大
    async def delete_model_history_before(self, timestamp: int) -> None:
        model = db_schema.ModelHistory
        await self._execute(delete(model).where(model.created_at < timestamp))

    # 列出某模型的歷史版本（新到舊）
    async def get_model_history(self, model_id: str) -> list:
        model = db_schema.ModelHistory
        statement = select(model).where(model.model_id == model_id)
        return await self._all(statement.order_by(model.created_at.desc()))

    # 退版：把指定歷史版本放回 model_registry，成為上線模型
    async def restore_model(self, history_id: int):
        history = await self.session.get(db_schema.ModelHistory, history_id)
        if history is None:
            raise ValueError(f"model_history id {history_id} 不存在")

        entry = await self.get_model(history.model_id)
        if entry is None:
            entry = db_schema.ModelRegistry(model_id=history.model_id)

        entry.model_type = history.model_type
        entry.features = history.features
        entry.metrics = history.metrics
        entry.model_binary = history.model_binary
        entry.created_at = float(history.created_at)  # 用該版本的訓練時間
        return await self._save(entry)

    # ============================================================
    # LoadPrediction
    # ============================================================

    # 寫入一個整點的負荷預測（同整點覆寫）
    async def save_load_prediction(self, row: dict) -> None:
        model = db_schema.LoadPrediction
        values = {
            column: row.get(column) for column in _columns(model) if column in row
        }
        values["created_at"] = int(time.time())  # 直接 INSERT 不會套用欄位預設值

        # 衝突時更新除了 target_ts 以外的欄位
        update_values = {
            column: value for column, value in values.items() if column != "target_ts"
        }
        statement = (
            insert(model)
            .values(values)
            .on_conflict_do_update(
                index_elements=["target_ts"],
                set_=update_values,
            )
        )
        await self._execute(statement)

    # 查詢負荷預測 [start, end)
    async def get_load_predictions(self, start_time: int, end_time: int) -> list:
        model = db_schema.LoadPrediction
        statement = select(model).where(
            model.target_ts >= start_time,
            model.target_ts < end_time,
        )
        return await self._all(statement.order_by(model.target_ts))
