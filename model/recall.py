# 負荷預測：載入存好的負荷模型預測各整點負荷並存檔、讀取存下的預測，
# 以及模型版本管理（列出歷史、退版）
# python -m model.recall        列出歷史版本
# python -m model.recall 12     退回 id 12
import asyncio
import sys
import time

import pandas as pd

import settings
from data import dates
from db import db_interaction, db_schema
from model import train

# 回補過去幾天缺漏的預測（與 /load-history 最多查 31 天一致）
BACKFILL_DAYS = 31
# 參考點缺監控資料時，最多往回找幾天的同整點平日資料代替
FALLBACK_DAYS = 7


# 時間顯示成「月/日 時:分」：log 訊息用
def _format_short(timestamp: int) -> str:
    return settings.to_local(timestamp).strftime("%m/%d %H:%M")


# 推算案場存下某整點預測的時間：每小時 :05 第一次能預測時（目標前 LAG_HOURS-1 小時）存入，
# 上班時段當天 07:00 再預測覆寫，且不晚於現在；replay 模擬與回補都據此找出當時那一版模型
def stored_at(target_timestamp: int, now_timestamp: int) -> int:
    # control 會 import recall，這裡延後 import 避免循環
    from model import control

    made_timestamp = target_timestamp - (train.LAG_HOURS - 1) * 3_600 + 300
    target = settings.to_local(target_timestamp)
    morning = int(settings.at_hour(target.date(), 7).timestamp())

    # 上班時段且已過當天 07:00：以 07:00 覆寫的預測為準
    in_work_hours = control.WORK_START <= target.hour <= control.WORK_END
    if in_work_hours and morning <= now_timestamp:
        made_timestamp = max(made_timestamp, morning)
    return min(made_timestamp, now_timestamp)


# 回測用：載入存好的模型並預測一段歷史區間，檢查模型在過去資料上的表現
class ModelRecaller:
    # 記下模型代號與回測區間（end_time 沒給時到現在）
    def __init__(
        self,
        model_id: str,
        start_time: int = 0,
        end_time: int | None = None,
    ):
        self.model_id = model_id
        self.start_time = start_time
        if end_time:
            self.end_time = end_time
        else:
            self.end_time = int(time.time())
        self.model = None

    # 從資料庫載入模型與特徵欄位清單
    async def recall_model(self) -> None:
        async with db_schema.async_session() as session:
            entry = await db_interaction.DBManager(session).get_model(self.model_id)
        if entry is None:
            raise ValueError(f"model_registry 裡沒有 {self.model_id}")

        self.model = entry.get_model()
        self.feature_cols = entry.features["feature_cols"]

    # 預測回測區間各小時負荷，回傳 (預測, 實際)，並計算開機小時的指標
    async def predict(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.model is None:
            raise ValueError("請先呼叫 recall_model()")

        # 特徵與實際負荷
        df_load = await train.get_chiller_hourly(self.start_time, self.end_time)
        hourly_features = train.build_hourly_features(df_load)
        hourly_features = train.add_target_features(hourly_features)
        x, y = train.split_xy(hourly_features)

        # 預測，負值歸零
        prediction_values = self.model.predict(x[self.feature_cols]).clip(min=0)
        prediction = pd.DataFrame(
            prediction_values,
            index=y.index,
            columns=["Q_COOL_PRED"],
        )

        # 指標只計被預測小時有開機的
        running = await train.get_running_hours(self.start_time, self.end_time)
        is_running = train.is_target_running(y.index, running)
        self.metric = train.evaluate(
            y["Q_COOL"][is_running],
            prediction["Q_COOL_PRED"][is_running],
        )
        return prediction, y


# 上線預測：預測指定整點的負荷，排程與 replay 都用它
class RealtimeModelRecaller(ModelRecaller):

    # 參考點（目標往回 LAG_HOURS 小時）的負荷與時間特徵；回傳 (特徵, 缺資料原因)
    async def reference_features(self, reference_timestamp: int) -> tuple:
        df_load = await train.get_chiller_hourly(
            reference_timestamp,
            reference_timestamp + 3_600,
        )
        if df_load[train.TARGET_COLUMN].dropna().empty:
            reason = (
                f"缺少 {_format_short(reference_timestamp)} 的監控資料"
                "（CH-1、CH-2 冷凍噸）"
            )
            return None, reason
        return train.build_hourly_features(df_load).iloc[0], None

    # 目標時間的日期特徵（是否國定假日）
    def target_features(self, target_timestamp: int) -> dict:
        target = settings.to_local(target_timestamp)
        is_holiday = dates.is_holiday(pd.DatetimeIndex([target]))[0]
        return {
            "IS_HOLIDAY_NEXT": int(is_holiday),
        }

    # 參考點缺監控資料時的備援：往回找最近一個有資料的平日同一整點（最多 FALLBACK_DAYS 天）；
    # 同一整點時間特徵一致、平日負荷型態相近；回傳 (特徵, 改用的參考點)，都沒有則 (None, None)
    async def fallback_features(self, reference_timestamp: int) -> tuple:
        for days_back in range(1, FALLBACK_DAYS + 1):
            candidate = reference_timestamp - days_back * 86_400
            if settings.to_local(candidate).weekday() >= 5:
                continue
            row, _ = await self.reference_features(candidate)
            if row is not None:
                return row, candidate
        return None, None

    # 預測 target_timestamp 所需的模型輸入（不需載入模型）：週末回傳 None（不預測）；
    # 參考點缺監控資料時改用備援參考點，連備援都沒有才印出原因並回傳 None
    async def target_input(self, target_timestamp: int) -> dict | None:
        # 參考點：往回 LAG_HOURS 小時（週一用上週五）
        target = settings.to_local(target_timestamp)
        if target.weekday() >= 5:
            return None
        lag_hours = 72 if target.weekday() == 0 else 24
        reference_timestamp = target_timestamp - lag_hours * 3_600
        row, reason = await self.reference_features(reference_timestamp)
        if row is None:
            row, fallback_timestamp = await self.fallback_features(reference_timestamp)
            if row is None:
                print(
                    f">>> {_format_short(target_timestamp)} 負荷預測略過：{reason}，"
                    f"往前 {FALLBACK_DAYS} 天的平日同時段也沒有資料"
                )
                return None
            print(
                f">>> {_format_short(target_timestamp)} 參考點 "
                f"{_format_short(reference_timestamp)} 缺資料，"
                f"改用 {_format_short(fallback_timestamp)}"
            )

        # 參考點特徵＋目標時間日期特徵
        features = row.to_dict()
        features.update(self.target_features(target_timestamp))
        return features

    # 預測 target_timestamp 那個整點的負荷（負值歸零）；週末或缺資料無法預測時回傳 None
    async def predict(self, target_timestamp: int) -> dict | None:
        features = await self.target_input(target_timestamp)
        if features is None:
            return None
        return self.predict_input(target_timestamp, features)

    # 用 target_input 算好的模型輸入預測負荷（負值歸零）：回補時先確認能預測才訓練模型，輸入不必重算
    def predict_input(self, target_timestamp: int, features: dict) -> dict:
        if self.model is None:
            raise ValueError("請先呼叫 recall_model()")

        x = pd.DataFrame([features])[self.feature_cols]
        return {
            "target_ts": target_timestamp,
            "q_cool_pred": max(float(self.model.predict(x)[0]), 0.0),
            "model_id": self.model_id,
        }


# 直接用剛訓練好的結果建立即時預測器（不讀寫資料庫），讓 replay 模擬當時的模型
def recaller_from_trainer(trainer: train.ModelTrainer) -> RealtimeModelRecaller:
    recaller = RealtimeModelRecaller(trainer.model_id)
    recaller.model = trainer.model
    recaller.feature_cols = list(trainer.x.columns)
    return recaller


# 預測指定日期各整點：{台灣時間: 預測結果（缺資料為 None）}
async def predict_day(recaller: RealtimeModelRecaller, day, hours) -> dict:
    target_times = [settings.at_hour(day, hour) for hour in hours]
    results = {}
    for target_time in target_times:
        results[target_time] = await recaller.predict(int(target_time.timestamp()))
    return results


# 預測指定日期上班時段各整點負荷並存入 load_prediction（每天 07:00 排程），控制建議讀這裡
async def predict_day_and_save(hours, day=None, model_id: str = train.MODEL_ID) -> str:
    recaller = RealtimeModelRecaller(model_id)
    await recaller.recall_model()
    if not day:
        day = settings.to_local(int(time.time())).date()

    # 只存有預測結果的整點
    results_by_time = await predict_day(recaller, day, hours)
    results = [result for result in results_by_time.values() if result is not None]
    await _save_predictions(results)
    return f"{day:%m/%d} 上班時段負荷預測 {len(results)}/{len(hours)} 小時"


# 補存未來各整點負荷預測（已存過的不覆寫，保留當時的預測）：目前整點之後，到現有資料能預測的
# 最遠整點；時序圖的後 24 小時預測與事後比對都靠這些存檔
async def predict_ahead_and_save(model_id: str = train.MODEL_ID) -> str:
    recaller = RealtimeModelRecaller(model_id)
    await recaller.recall_model()
    hour_now = int(time.time()) // 3_600 * 3_600
    targets = range(hour_now + 3_600, hour_now + train.LAG_HOURS * 3_600, 3_600)

    missing = await _missing_targets(targets)
    results = []
    for target_timestamp in missing:
        result = await recaller.predict(target_timestamp)
        if result is not None:
            results.append(result)
    await _save_predictions(results)
    last_target = _format_short(targets[-1])
    return f"未來負荷預測（到 {last_target}）：新增 {len(results)}/{len(missing)} 小時"


# 回補過去 BACKFILL_DAYS 天缺漏的負荷預測（服務停機、監控斷線後，/load-history 才不會少一段）：
# 和 replay 一樣，每個整點用「當時那一版」每週重訓模型（只用那時以前的資料訓練），
# 結果等同服務當時有在跑會存下的預測；已存的不覆寫；週末、當時資料不足還沒有模型的不補
async def backfill_past_predictions(model_id: str = train.MODEL_ID) -> str:
    now_timestamp = int(time.time())
    hour_now = now_timestamp // 3_600 * 3_600
    targets = range(hour_now - BACKFILL_DAYS * 86_400, hour_now + 3_600, 3_600)

    # 先算模型輸入（不需模型）：週末、連備援參考點都沒有的整點不必為它訓練模型
    input_reader = RealtimeModelRecaller(model_id)
    inputs = {}
    for target_timestamp in await _missing_targets(targets):
        features = await input_reader.target_input(target_timestamp)
        if features is not None:
            inputs[target_timestamp] = features

    # 各次每週重訓的模型（同一次只訓練一次）；當時資料不足為 None
    recallers = {}
    results = []
    for target_timestamp, features in inputs.items():
        train_end = train.last_retrain_timestamp(stored_at(target_timestamp, now_timestamp))
        if train_end not in recallers:
            trainer, _ = await train.train_until(train_end, model_id)
            if trainer is None:
                recallers[train_end] = None
            else:
                recallers[train_end] = recaller_from_trainer(trainer)
        if recallers[train_end] is not None:
            results.append(recallers[train_end].predict_input(target_timestamp, features))

    await _save_predictions(results)
    return f"過去 {BACKFILL_DAYS} 天負荷預測回補：新增 {len(results)}/{len(inputs)} 小時"


# targets 裡還沒存過預測的整點
async def _missing_targets(targets: range) -> list[int]:
    stored = await get_predictions(targets.start, targets.stop)
    return [target for target in targets if target not in stored.index]


# 存入多筆負荷預測（同整點覆寫）
async def _save_predictions(results: list[dict]) -> None:
    async with db_schema.async_session() as session:
        db = db_interaction.DBManager(session)
        for result in results:
            await db.save_load_prediction(result)


# ---------- 讀取存下的預測 ----------


# 指定日期 hours 各整點的逐時預測負荷 (RT)，從 load_prediction 讀；無預測為 NaN
async def load_forecast(day, hours) -> pd.Series:
    target_times = pd.DatetimeIndex([settings.at_hour(day, hour) for hour in hours])
    start_timestamp = int(target_times[0].timestamp())
    end_timestamp = int(target_times[-1].timestamp()) + 3_600
    predictions = await get_predictions(start_timestamp, end_timestamp)
    prediction = pd.Series(
        predictions.to_numpy(),
        index=settings.to_local_index(predictions.index),
        dtype=float,
    )
    return prediction.reindex(target_times)


# 讀 load_prediction 裡 start_timestamp ~ end_timestamp 的預測負荷 (RT)：index 為整點 Unix 秒
async def get_predictions(start_timestamp: int, end_timestamp: int) -> pd.Series:
    async with db_schema.async_session() as session:
        db = db_interaction.DBManager(session)
        rows = await db.get_load_predictions(start_timestamp, end_timestamp)
    return pd.Series({row.target_ts: row.q_cool_pred for row in rows}, dtype=float)


# ---------- 版本管理 ----------


# 時間顯示成「年-月-日 時:分」（台灣時間）：版本清單用
def _format_full(timestamp: int) -> str:
    return f"{settings.to_local(timestamp):%Y-%m-%d %H:%M}"


# 指令列版本管理：列出歷史版本；給 id 時退回該版本（新模型表現不好時可退版）
async def versions(args: list[str]) -> None:
    async with db_schema.async_session() as session:
        db = db_interaction.DBManager(session)

        if args:
            entry = await db.restore_model(int(args[0]))
            print(f"已退回 {entry.model_id}，訓練時間 {_format_full(entry.created_at)}")
            return

        for history in await db.get_model_history(train.MODEL_ID):
            print(
                f"id {history.id}｜訓練時間 {_format_full(history.created_at)}"
                f"｜資料 {_format_full(history.train_start)}"
                f" ~ {_format_full(history.train_end)}"
                f"｜指標 {history.metrics}"
            )


if __name__ == "__main__":
    asyncio.run(versions(sys.argv[1:]))
