# 排程：讓服務自動定時工作——啟動時與每天 07:00 預測上班時段負荷、回補過去缺漏的預測；
#       每小時 :05 抓監控資料、補存未來負荷預測；每週一 01:00 重訓模型
import datetime
import functools
from collections.abc import Awaitable, Callable

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler

import settings
from data import sensor
from db import db_interaction, db_schema
from model import control, recall, train

# 排程延遲容許 10 分鐘
scheduler = AsyncIOScheduler(job_defaults={"misfire_grace_time": 600})


# 執行一項工作並印出結果（含時間，從 log 才看得出工作何時跑、有沒有跑）；
# 失敗只印錯誤，不讓一項工作出錯影響其他排程；回傳失敗原因（成功為 None）
async def run_job(name: str, job: Callable[[], Awaitable[str]]) -> str | None:
    try:
        result = await job()
        print(f">>> {_now_text()} {result}")
        return None
    except Exception as error:
        print(f">>> {_now_text()} {name}失敗：{error}")
        return str(error)


# 現在時間「月/日 時:分」（台灣時間）：log 用
def _now_text() -> str:
    return f"{pd.Timestamp.now(tz=settings.settings.timezone):%m/%d %H:%M}"


# 確認上線模型可用：還沒有模型，或模型的特徵和程式現在的特徵不同（改過特徵、沒重訓）時先重訓；
# 不一致時每次預測都會失敗，所以預測前都先檢查；回傳重訓失敗原因（不需重訓或成功為 None）
async def ensure_model() -> str | None:
    async with db_schema.async_session() as session:
        manager = db_interaction.DBManager(session)
        entry = await manager.get_model(train.MODEL_ID)

    code_columns = train.x_columns()
    if entry is None:
        reason = f"還沒有負荷模型 {train.MODEL_ID}"
    else:
        model_columns = entry.features.get("feature_cols")
        if model_columns == code_columns:
            return None
        reason = f"模型特徵與程式不同（模型：{model_columns}；程式：{code_columns}）"

    print(f">>> {_now_text()} {reason}，重新訓練")
    return await run_job("負荷模型重訓", train.weekly_retrain)


# 每天 07:00（啟動時也執行）：先確認模型可用，再預測上班時段各整點負荷，
# 控制建議要用這些預測；day 為 None 時預測今天
async def daily_load_prediction(day: datetime.date | None = None) -> None:
    await ensure_model()

    predict_day = functools.partial(
        recall.predict_day_and_save,
        control.WORK_HOURS,
        day,
    )
    await run_job("上班時段負荷預測", predict_day)


# 回補過去缺漏的負荷預測（啟動時、每天 07:00）：停機或監控斷線後 /load-history 才不會少一段；
# 不放每小時，避免補不了的整點每小時觸發重訓
async def past_prediction_backfill() -> None:
    await run_job("過去負荷預測回補", recall.backfill_past_predictions)


# 每天 07:00：預測今天上班時段，再回補過去缺漏的預測
async def morning_update() -> None:
    await daily_load_prediction()
    await past_prediction_backfill()


# 每小時 :05：先抓監控資料，再補存未來負荷預測（時序圖後 24 小時的預測線）
async def hourly_update() -> None:
    await run_job("VM 資料抓取", sensor.update)
    await ensure_model()
    await run_job("未來負荷預測", recall.predict_ahead_and_save)


# 每週一 01:00：用累積到現在的資料重訓負荷模型，讓模型跟上負荷變化
async def weekly_load_retrain() -> None:
    await run_job("每週負荷模型重訓", train.weekly_retrain)


# 服務啟動：建資料表、抓資料、預測（下班後啟動則含隔日），再開始排程
async def start() -> None:
    timezone = settings.settings.timezone
    await run_job("啟動 VM 資料抓取", sensor.update)
    now = pd.Timestamp.now(tz=timezone)
    for day in control.plan_days(now):
        await daily_load_prediction(day)
    await run_job("未來負荷預測", recall.predict_ahead_and_save)

    # 每小時 :05、每天 07:00、每週一 01:00
    scheduler.add_job(hourly_update, "cron", minute=5, timezone=timezone)
    scheduler.add_job(
        morning_update,
        "cron",
        hour=7,
        minute=0,
        timezone=timezone,
    )
    scheduler.add_job(
        weekly_load_retrain,
        "cron",
        day_of_week="mon",
        hour=1,
        minute=0,
        timezone=timezone,
    )
    scheduler.start()
    # 回補要重訓多版模型、較久：排成馬上執行一次的背景工作，不拖慢服務啟動
    scheduler.add_job(past_prediction_backfill)


# 服務關閉：停止排程
def stop() -> None:
    scheduler.shutdown()
