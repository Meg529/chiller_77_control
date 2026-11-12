# 資料庫連線與資料表定義：監控資料、逐時氣象、負荷模型（含歷史版本）、負荷預測；
# 所有程式都透過這裡的 async_session 連資料庫
import asyncio
import io
import time
from typing import Any, Dict, Optional

import asyncpg
import joblib
from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    Index,
    LargeBinary,
    UniqueConstraint,
)
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import Field, SQLModel

import settings

# 服務開機自動啟動時 PostgreSQL 可能還沒起來：連不上就每隔幾秒重試，最多等多久 (s)
DB_WAIT_SECONDS = 300
DB_RETRY_SECONDS = 5

# ============================================================
# 資料庫連線（統一使用 async_session）
# ============================================================

DATABASE_URL = (
    f"postgresql+asyncpg://"
    f"{settings.settings.postgres_user}:{settings.settings.postgres_password}@"
    f"{settings.settings.postgres_host}:{settings.settings.postgres_port}/"
    f"{settings.settings.postgres_db}"
)

engine = create_async_engine(DATABASE_URL, echo=False)

async_session = sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


# ============================================================
# SensorData
# ============================================================


# 監控資料表：每分鐘一筆（Unix 秒），模型訓練與分析的原始資料
class SensorData(SQLModel, table=True):

    __tablename__ = "sensor_data"
    __table_args__ = (
        UniqueConstraint(
            "timestamp",
            "metric_id",
            "metric_key",
            name="uq_sensor_data_metric",
        ),
        Index(
            "ix_sensor_data_metric_time",
            "metric_id",
            "metric_key",
            "timestamp",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: int = Field(sa_type=BigInteger, nullable=False)
    metric_id: str = Field(nullable=False)
    metric_key: str = Field(nullable=False)
    value: float = Field(nullable=False)


# ============================================================
# WeatherHistory
# ============================================================


# 逐時氣象觀測表（每日中午更新前一日），只給開發機分析用
class WeatherHistory(SQLModel, table=True):

    __tablename__ = "weather_history"
    __table_args__ = (
        UniqueConstraint(
            "timestamp",
            "station_id",
            name="uq_weather_history",
        ),
        Index(
            "ix_weather_history_station_time",
            "station_id",
            "timestamp",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: int = Field(sa_type=BigInteger, nullable=False)
    station_id: str = Field(nullable=False)
    station_name: Optional[str] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    humidity: Optional[float] = Field(default=None)
    pressure: Optional[float] = Field(default=None)
    precipitation: Optional[float] = Field(default=None)
    wind_speed: Optional[float] = Field(default=None)
    dew_point: Optional[float] = Field(default=None)
    solar_radiation: Optional[float] = Field(default=None)
    wind_direction: Optional[float] = Field(default=None)
    cloud_amount: Optional[float] = Field(default=None)
    cloud_height: Optional[float] = Field(default=None)
    visibility: Optional[float] = Field(default=None)


# ============================================================
# ModelRegistry / ModelHistory
# ============================================================


# 模型序列化成 bytes：模型要存進資料庫欄位
def dump_model(model_obj: Any) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(model_obj, buffer)
    return buffer.getvalue()


# 把資料庫裡的 bytes 還原成模型
def load_model(model_binary: bytes) -> Any:
    return joblib.load(io.BytesIO(model_binary))


# 目前 Unix 秒（整數），給 created_at 當預設值
def _now_timestamp() -> int:
    return int(time.time())


# 上線模型表：每個 model_id 一筆，預測時讀這裡
class ModelRegistry(SQLModel, table=True):
    __tablename__ = "model_registry"

    id: Optional[int] = Field(default=None, primary_key=True)
    model_id: str = Field(nullable=False, unique=True, index=True)
    model_type: str = Field(nullable=False)
    features: Dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    model_binary: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    metrics: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    created_at: float = Field(default_factory=time.time)

    # 存入模型（序列化成 bytes）
    def set_model(self, model_obj: Any) -> None:
        self.model_binary = dump_model(model_obj)

    # 取出模型（還原 bytes）
    def get_model(self) -> Any:
        return load_model(self.model_binary)


# 模型歷史版本表：每次重訓備份一筆，新模型不好時可退版
class ModelHistory(SQLModel, table=True):
    __tablename__ = "model_history"

    id: Optional[int] = Field(default=None, primary_key=True)
    model_id: str = Field(nullable=False, index=True)
    model_type: str = Field(nullable=False)
    features: Dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    model_binary: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    metrics: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    # 訓練資料起訖時間
    train_start: Optional[int] = Field(default=None, sa_type=BigInteger)
    train_end: Optional[int] = Field(default=None, sa_type=BigInteger)
    # 訓練時間
    created_at: int = Field(default_factory=_now_timestamp, sa_type=BigInteger)


# ============================================================
# LoadPrediction
# ============================================================


# 負荷預測表：每個整點一筆（重算覆寫），控制建議與時序圖讀這裡
class LoadPrediction(SQLModel, table=True):
    __tablename__ = "load_prediction"

    id: Optional[int] = Field(default=None, primary_key=True)
    # 被預測的整點
    target_ts: int = Field(
        sa_type=BigInteger,
        nullable=False,
        unique=True,
        index=True,
    )
    # 預測全館冷凍噸（RT）
    q_cool_pred: float = Field(nullable=False)
    model_id: str = Field(nullable=False)
    # 預測時間
    created_at: int = Field(default_factory=_now_timestamp, sa_type=BigInteger)


# ============================================================
# 建立資料表
# ============================================================


# 建立還不存在的資料表（既有表不更動），服務啟動時執行；
# PostgreSQL 還沒起來（連線被拒、正在啟動）時每 DB_RETRY_SECONDS 秒重試，超過 DB_WAIT_SECONDS 才放棄
async def create_tables_if_not_exists() -> None:
    start_time = time.monotonic()
    while True:
        try:
            async with engine.begin() as connection:
                await connection.run_sync(SQLModel.metadata.create_all)
            return
        except (
            OSError,
            DBAPIError,
            asyncpg.CannotConnectNowError,
            asyncpg.PostgresConnectionError,
        ) as error:
            waited = time.monotonic() - start_time
            if waited >= DB_WAIT_SECONDS:
                raise
            print(
                f">>> PostgreSQL 尚未就緒，{DB_RETRY_SECONDS} 秒後重試"
                f"（已等待 {waited:.0f} / {DB_WAIT_SECONDS} 秒）：{error}"
            )
            await asyncio.sleep(DB_RETRY_SECONDS)