# 業主 API 服務入口：提供各段控制建議與負荷時序圖，讓現場知道何時開哪幾台、出水設多少
# API 由 create_app(資料來源) 建立：正式服務用 LiveDataSource（真的現在、資料庫）；
# replay.py 用 SimulatedSite（假設的現在、模擬資料）；排程見 service/schedule.py、繪圖見 service/chart.py
import asyncio
import datetime
import enum
import functools
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import settings
from data import sensor
from db import db_schema
from model import control, recall
from service import chart, schedule

# 一天分幾段控制：API 下拉選項（文字）→ 段數
SegmentOption = enum.Enum(
    "SegmentOption",
    {name: name for name in control.SEGMENT_NAMES.values()},
    type=str,
)
SEGMENT_COUNT_BY_NAME = {
    name: segment_count for segment_count, name in control.SEGMENT_NAMES.items()
}

# API 測試頁與圖表用的檔案（案場無外網，不連外部網站）
STATIC = Path(__file__).parent / "static"
# /load-history 最多幾天
HISTORY_MAX_DAYS = 31

APP_DESCRIPTION = "冰水主機負荷預測與控制建議服務：提供各段控制建議與負荷時序圖"


# 正式服務的資料來源：真的現在時間與資料庫；介面和 replay 的模擬來源相同，API 程式才能共用
class LiveDataSource:
    # /recommendation 附加欄位
    extra = {}

    # 現在時間（台灣時間）：API 依它決定今天、現在在哪一段
    @property
    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz=settings.settings.timezone)

    # 同時多個查詢依序預測，不重複計算
    _predict_lock = asyncio.Lock()

    # 某天上班時段的逐時預測負荷、流量典型值與預測失敗原因（成功為 None）：產生控制建議用。
    # 平日每次查詢都用目前的模型重新預測（覆寫存檔）再讀回，不只讀之前存好的預測——
    # 程式或模型有問題時查詢當下就看得出來；預測失敗不拿舊存檔頂替，回傳失敗原因
    async def day_inputs(
        self,
        day: datetime.date,
    ) -> tuple[pd.Series, dict, str | None]:
        # 週末不預測
        if day.weekday() >= 5:
            prediction, flows = await control.day_inputs(day)
            return prediction, flows, None

        predict_day = functools.partial(
            recall.predict_day_and_save,
            control.WORK_HOURS,
            day,
        )
        async with self._predict_lock:
            error = await schedule.ensure_model()
            if error is None:
                error = await schedule.run_job("查詢時上班時段負荷預測", predict_day)
            if error is None:
                prediction, flows = await control.day_inputs(day)
                # 有整點缺預測（參考點缺監控資料）：先抓最新監控資料再預測一次，
                # 順便補未來 23 小時（同每小時 :05，時序圖預測線、明天的預測）
                if prediction.isna().any():
                    await schedule.run_job("查詢時 VM 資料抓取", sensor.update)
                    error = await schedule.run_job("查詢時上班時段負荷預測", predict_day)
                    await schedule.run_job("查詢時未來負荷預測", recall.predict_ahead_and_save)
                    prediction, flows = await control.day_inputs(day)

        if error is not None:
            prediction, flows = await control.day_inputs(day)
            prediction[:] = np.nan
            return prediction, flows, f"負荷預測失敗：{error}"
        return prediction, flows, None

    # 一段期間的逐時實際負荷與預測負荷：畫負荷時序圖用
    async def get_load_frame(
        self,
        start_timestamp: int,
        end_timestamp: int,
    ) -> pd.DataFrame:
        return await chart.get_load_frame(start_timestamp, end_timestamp)


# 數值取到小數一位，缺值回 None：API 回傳的 JSON 不能有 NaN，缺值要輸出 null
def _round_or_none(value: float) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 1)


# 一段控制建議轉成 API 回傳欄位（Start、End、Chillers、ChillerCount、SupplyWaterSetpoint）
def _segment_fields(segment: dict) -> dict:
    if pd.isna(segment["chiller_count"]):
        chiller_count = None
    else:
        chiller_count = int(segment["chiller_count"])
    return {
        "Start": segment["start"],
        "End": segment["end"],
        "Chillers": segment["chillers"],
        "ChillerCount": chiller_count,
        "SupplyWaterSetpoint": _round_or_none(segment["supply_temp"]),
    }


# /load-history 的開始、結束日期換成 Unix 秒期間（開始日 00:00 ~ 結束日隔天 00:00），
# 並擋掉結束早於開始、或超過 HISTORY_MAX_DAYS 天的查詢
def history_range(
    start_date: datetime.date,
    end_date: datetime.date,
) -> tuple[int, int]:
    if end_date < start_date:
        raise HTTPException(400, "結束日期不能早於開始日期")
    if (end_date - start_date).days + 1 > HISTORY_MAX_DAYS:
        raise HTTPException(400, f"最多查 {HISTORY_MAX_DAYS} 天")
    next_day = end_date + datetime.timedelta(days=1)
    start_timestamp = int(settings.at_hour(start_date, 0).timestamp())
    end_timestamp = int(settings.at_hour(next_day, 0).timestamp())
    return start_timestamp, end_timestamp


# 建立 API：source 提供 now、extra、day_inputs(day)、get_load_frame(start_timestamp, end_timestamp)；
# 正式服務與 replay 共用同一份端點，只換資料來源（LiveDataSource 或 replay.SimulatedSite）
def create_app(
    source: Any,
    lifespan: Callable | None = None,
    title: str = "chiller_77",
) -> FastAPI:
    app = FastAPI(
        title=title,
        description=APP_DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # 端點不寫回傳型別：FastAPI 會把回傳型別當成回應格式，改變 API 文件與輸出

    # API 測試頁：讀本機 /static 的檔案，案場無外網也能開
    @app.get("/docs", include_in_schema=False)
    def docs():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title,
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
            swagger_favicon_url="/static/favicon.png",
        )

    # 業主 API：今天從現在開始的各段控制建議（依預測負荷），現場照著設定主機
    @app.get("/recommendation")
    async def recommendation(
        segments: Annotated[SegmentOption, Query(description="一天分幾段控制")],
    ):
        segment_count = SEGMENT_COUNT_BY_NAME[segments.value]
        now = source.now
        prediction, flows, error = await source.day_inputs(now.date())
        schedule_rows = control.schedule_from(prediction, flows, segment_count, now)

        response = dict(source.extra)
        # 預測失敗時說明原因（正常時沒有這個欄位）
        if error is not None:
            response["Message"] = error
        response["Segments"] = segment_count
        response["Schedule"] = [_segment_fields(row) for row in schedule_rows]

        for sch in response["Schedule"]:
            if sch["SupplyWaterSetpoint"] is not None:
                sch["SupplyWaterSetpoint"] += settings.settings.control_t_offset

        return response

    # 業主時序圖：開啟當下前 24 小時實際與預測負荷、後 24 小時預測負荷，讓人看預測準不準
    @app.get("/load-chart", response_class=HTMLResponse)
    async def load_chart():
        now = source.now
        hour_now = int(now.timestamp()) // 3_600 * 3_600
        load_frame = await source.get_load_frame(
            hour_now - 24 * 3_600,
            hour_now + 24 * 3_600,
        )
        figure = chart.load_figure(load_frame, now.tz_localize(None))
        return chart.to_html(figure)

    # 指定期間的實際負荷與預測負荷時序圖：回頭檢查某段期間的預測表現
    @app.get("/load-history", response_class=HTMLResponse)
    async def load_history(
        start_date: Annotated[datetime.date, Query(description="開始日期")],
        end_date: Annotated[
            datetime.date,
            Query(description=f"結束日期（含當天，最多 {HISTORY_MAX_DAYS} 天）"),
        ],
    ):
        start_timestamp, end_timestamp = history_range(start_date, end_date)
        load_frame = await source.get_load_frame(start_timestamp, end_timestamp)
        figure = chart.history_figure(load_frame, start_date, end_date)
        return chart.to_html(figure)

    return app


# 服務啟動時開始排程（抓資料、預測、重訓），關閉時停止排程
@asynccontextmanager
async def lifespan(app: FastAPI):
    await db_schema.create_tables_if_not_exists()

    await schedule.start()
    yield
    schedule.stop()


app = create_app(LiveDataSource(), lifespan)
