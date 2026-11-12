# 負荷模型訓練：用 24 小時前的負荷、時間特徵與目標日期，預測下一個整點的全館冷凍噸；
# 控制建議靠這個預測決定台數與出水；指標只計開機小時
import asyncio
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import settings
from data import dates, sensor
from db import db_interaction, db_schema
from model import rules

# 上線模型
MODEL_ID = "load_pred_v1"
TARGET_COLUMN = "Q_COOL"
LOAD_KEY = "冰水流量冷凍噸"
# 用幾小時前的負荷預測：參考點與目標的間隔 (h)
LAG_HOURS = 24

# 目標時間的日期特徵
TARGET_CALENDAR_COLUMNS = ["IS_HOLIDAY"]
# 參考點的時間特徵
TIME_COLUMNS = ["HOUR_SIN", "HOUR_COS", "IS_WORKING_HOUR", "IS_HOLIDAY"]
# 重訓所需最少天數
MIN_TRAIN_DAYS = 14
# 歷史版本保留天數
MODEL_HISTORY_DAYS = 365


# 計算 MAE、RMSE、R²：評估預測準不準
def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(mean_squared_error(y_true, y_pred) ** 0.5, 2),
        "R2": round(r2_score(y_true, y_pred), 2),
    }


# 訓練模型並計算訓練集指標（只計被預測小時有開機的，控制只在開機時用得到）
def model_training(y, x, model, is_running) -> tuple:
    model.fit(x, y)
    # 負值歸零
    prediction = model.predict(x).clip(min=0)
    metric = evaluate(y[is_running], prediction[is_running])
    return prediction, model, metric


# 全館冷凍噸逐時平均（兩台相加）；任一台缺值則為 NaN，避免只算到一台而低估負荷
async def get_chiller_hourly(start_time, end_time) -> pd.DataFrame:
    hourly = []
    async with db_schema.async_session() as session:
        db = db_interaction.DBManager(session)
        for chiller in rules.CHILLERS.values():
            chiller_load = await sensor.get_hourly_mean(
                db,
                chiller["id"],
                LOAD_KEY,
                start_time,
                end_time,
            )
            hourly.append(chiller_load)
    total = pd.concat(hourly, axis=1).sum(axis=1, min_count=len(hourly))
    return total.rename(TARGET_COLUMN).rename_axis("ts_hour").reset_index()


# 開機小時（任一台功率 > 20 kW 且流量 > 100 LPM）：指標只計開機小時要用
async def get_running_hours(start_time, end_time) -> pd.Series:
    hourly_by_chiller = await sensor.get_chiller_hourly_power_flow(
        start_time,
        end_time,
    )
    running_by_chiller = []
    for chiller_hourly in hourly_by_chiller.values():
        running_by_chiller.append(sensor.is_running(chiller_hourly))
    running_table = pd.concat(running_by_chiller, axis=1)
    running = running_table.fillna(False).astype(bool).any(axis=1)
    running.index = settings.to_local_index(running.index)
    return running


# 每列要預測的小時（下一小時）是否開機：對齊特徵列與被預測小時
def is_target_running(index: pd.DatetimeIndex, running: pd.Series) -> np.ndarray:
    target = index + pd.Timedelta(hours=1)
    return running.reindex(target, fill_value=False).to_numpy(dtype=bool)


# 負荷轉成連續逐時序列（台灣時間），缺的整點補 NaN，後面位移對齊才不會錯位
def to_hourly_index(df_load: pd.DataFrame) -> pd.DataFrame:
    load = df_load.set_index("ts_hour")[TARGET_COLUMN]
    load.index = settings.to_local_index(load.index)
    return load.resample("1h").mean().to_frame()


# 逐時負荷加上時間特徵（小時、星期、上班時段、假日）：模型輸入
def build_hourly_features(df_load: pd.DataFrame) -> pd.DataFrame:
    return dates.add_time_features(to_hourly_index(df_load))


# 加入目標時間（下一小時）的日期特徵（_NEXT）：預測時已知目標是星期幾、是否假日
def add_target_features(df: pd.DataFrame) -> pd.DataFrame:
    for column in TARGET_CALENDAR_COLUMNS:
        df[f"{column}_NEXT"] = df[column].shift(-1)
    return df


# 模型特徵欄位清單：訓練與預測用同一份順序
def x_columns() -> list[str]:
    columns = [TARGET_COLUMN]
    columns.extend(TIME_COLUMNS)
    for column in TARGET_CALENDAR_COLUMNS:
        columns.append(f"{column}_NEXT")
    return columns


# 切分特徵 X 與目標 y：每列時間的 y 為下一小時負荷，X 取參考點（目標往回 LAG_HOURS 小時）特徵
def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[TARGET_COLUMN].shift(-1).dropna().to_frame()

    feature_columns = x_columns()
    current_columns = [
        column for column in feature_columns if not column.endswith("_NEXT")
    ]
    x = df[feature_columns].copy()

    # 參考點：目標往回 LAG_HOURS 小時；目標是週一時改用上週五同時段（再往回 48 小時）
    target_index = y.index + pd.Timedelta(hours=1)
    reference_index = target_index - pd.Timedelta(hours=LAG_HOURS)
    is_monday = target_index.weekday == 0
    reference_index = reference_index.where(
        ~is_monday,
        reference_index - pd.Timedelta(hours=48),
    )
    reference_features = df[current_columns].reindex(reference_index)
    reference_features.index = y.index
    x[current_columns] = reference_features

    # 只用目標是平日、特徵與目標都完整的列
    is_weekday = pd.Series(target_index.weekday < 5, index=y.index)
    valid = is_weekday & x.notna().all(axis=1) & y.notna().all(axis=1)
    return x.loc[valid], y.loc[valid]


# 負荷模型訓練流程：讀資料 → 清理 → 特徵 → 訓練；每週重訓與 notebook 示範共用
class ModelTrainer:

    # 記下模型代號與訓練資料起訖時間（Unix 秒）
    def __init__(
        self,
        model_id,
        start_date: int,
        end_date: int,
    ):
        self.model_id = model_id
        self.start_date = start_date
        self.end_date = end_date

    # 讀取全館負荷與開機小時
    async def get_data(self) -> None:
        self.df_load = await get_chiller_hourly(self.start_date, self.end_date)
        self.running = await get_running_hours(self.start_date, self.end_date)

    # 負值視為儀表異常改成缺值，不拿來訓練
    def clean_data(self) -> None:
        is_negative = self.df_load[TARGET_COLUMN] < 0
        self.df_load.loc[is_negative, TARGET_COLUMN] = np.nan

    # 產生逐時負荷與時間特徵
    def feature_engineering(self) -> None:
        self.df = build_hourly_features(self.df_load)

    # 加入目標時間日期特徵後切分 X、y
    def split_xy(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.df = add_target_features(self.df)
        return split_xy(self.df)

    # 訓練 LightGBM 模型，指標只計被預測小時有開機的
    def train_model(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        is_running = is_target_running(x.index, self.running)
        model = LGBMRegressor(n_estimators=100, random_state=42, min_child_samples=5)
        self.y_pred, self.model, self.metric = model_training(
            y=y,
            x=x,
            model=model,
            is_running=is_running,
        )
        self.x = x

    # 完整訓練流程：依序執行上面各步驟
    async def main(self) -> None:
        await self.get_data()
        self.clean_data()
        self.feature_engineering()
        x, y = self.split_xy()
        self.train_model(x, y)


# 把訓練好的模型存入 model_registry 給上線預測用，並備份到 model_history（可退版），再清除過期版本
async def save_model(trainer: ModelTrainer) -> None:
    async with db_schema.async_session() as session:
        db = db_interaction.DBManager(session)
        entry = await db.save_model(
            model_id=trainer.model_id,
            model_type=type(trainer.model).__name__,
            model_obj=trainer.model,
            metrics=trainer.metric,
            # history_steps、forecast_steps 固定寫入，相容既有版本
            features={
                "history_steps": 24,
                "forecast_steps": 1,
                "feature_cols": list(trainer.x.columns),
            },
        )
        # 備份並清除過期版本
        await db.add_model_history(
            entry,
            train_start=int(trainer.x.index.min().timestamp()),
            train_end=int(trainer.x.index.max().timestamp()),
        )
        expire_before = int(time.time()) - MODEL_HISTORY_DAYS * 86_400
        await db.delete_model_history_before(expire_before)


# 用 train_end 以前的全部資料訓練（replay 模擬當時的模型也用它）；資料不足 MIN_TRAIN_DAYS 天時不採用
# 回傳 (訓練結果, 說明)；不採用時訓練結果為 None
async def train_until(train_end: int, model_id: str = MODEL_ID) -> tuple:
    trainer = ModelTrainer(
        model_id=model_id,
        start_date=0,
        end_date=train_end,
    )
    await trainer.main()

    days = (trainer.x.index.max() - trainer.x.index.min()).days
    if days < MIN_TRAIN_DAYS:
        return None, f"資料只有 {days} 天（少於 {MIN_TRAIN_DAYS} 天）"
    return trainer, f"{days} 天資料，開機小時指標 {trainer.metric}"


# 最近一次每週重訓時間（週一 01:00，不晚於 now_timestamp）：replay 據此找出當時用哪一版模型
def last_retrain_timestamp(now_timestamp: int) -> int:
    now = settings.to_local(now_timestamp)
    week_start = now.normalize() - pd.Timedelta(days=now.weekday())
    monday = week_start + pd.Timedelta(hours=1)
    if monday > now:
        monday -= pd.Timedelta(days=7)
    return int(monday.timestamp())


# 每週重訓：以全部歷史資料重訓並存檔；資料不足時保留舊模型，不換上不可靠的模型
async def weekly_retrain(model_id: str = MODEL_ID) -> str:
    trainer, message = await train_until(int(time.time()), model_id)
    if trainer is None:
        return f"{message}，不重訓、保留舊模型"
    await save_model(trainer)
    return f"{model_id} 已重訓（{message}）"


# 手動重訓（同每週一 01:00）：.venv\Scripts\python.exe -m model.train
if __name__ == "__main__":
    print(asyncio.run(weekly_retrain()))
