# 模擬案場：指定過去時間當作現在，只用那時以前的資料重跑負荷預測與控制建議（不寫入資料庫），
# 用來確認程式在案場會給出什麼結果
# python -m replay --now "2026-10-01 07:00"
# python -m replay --now "2026-10-01 18:00" --segments 1
# python -m replay --now "2026-10-01 07:00" --serve
#   開 API（http://127.0.0.1:8001/docs；時序圖 /load-chart、期間圖 /load-history）
import argparse
import asyncio
import datetime
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException

import app
import settings
from model import control, recall, rules, train


# 時間顯示成「月/日 時:分」（台灣時間）：印訊息用
def _short_time(timestamp: int) -> str:
    return f"{settings.to_local(timestamp):%m/%d %H:%M}"


# 假設現在是 now_timestamp 的案場：各週模型只用當時以前的資料訓練，預測也只用現在以前的資料；
# 也是 app.create_app 的資料來源（now、extra、day_inputs、get_load_frame），API 與正式服務共用
class SimulatedSite:
    # 記下假設的現在時間，並準備模型、每日預測的暫存，避免重複計算
    def __init__(self, now_timestamp: int):
        self.now_ts = now_timestamp
        self.now = settings.to_local(now_timestamp)
        self.hour_now = now_timestamp // 3_600 * 3_600
        self.now_text = f"{self.now:%Y-%m-%d %H:%M}"
        # /recommendation 附加欄位
        self.extra = {"AssumedNow": self.now_text}
        self.models = {}
        self.messages = {}
        self.days = {}

    # 某次每週重訓的模型（同一次只訓練一次）；當時資料不足、沒有模型則為 None
    async def model(self, train_end: int):
        if train_end not in self.models:
            trainer, message = await train.train_until(train_end)
            self.messages[train_end] = message
            if trainer is None:
                self.models[train_end] = None
            else:
                self.models[train_end] = recall.recaller_from_trainer(trainer)
        return self.models[train_end]

    # 某天上班時段的原始預測：照案場當天 07:00 用當時那一版模型預測；當時沒有模型為 None
    async def day(self, day: datetime.date) -> pd.Series | None:
        if day not in self.days:
            day_morning = int(settings.at_hour(day, 7).timestamp())
            made_timestamp = min(day_morning, self.now_ts)
            model = await self.model(train.last_retrain_timestamp(made_timestamp))
            if model is None:
                self.days[day] = None
            else:
                results = await recall.predict_day(model, day, control.WORK_HOURS)

                # 各整點的預測值；沒有預測的整點為 NaN
                predictions = {}
                for target_time, result in results.items():
                    if result:
                        predictions[target_time] = result["q_cool_pred"]
                    else:
                        predictions[target_time] = np.nan
                self.days[day] = pd.Series(predictions, dtype=float)
        return self.days[day]

    # 模擬資料庫 start_timestamp ~ end_timestamp 的逐時資料（欄位同 app.get_load_frame），給時序圖用：
    #   pred：案場當時存下的預測（各整點用存檔當時那一版模型）
    async def history(self, start_timestamp: int, end_timestamp: int) -> pd.DataFrame:
        # 先訓練現在的模型
        await self.model(train.last_retrain_timestamp(self.now_ts))
        hours = range(start_timestamp, end_timestamp, 3_600)
        prediction = pd.Series(np.nan, index=hours)
        # 各次重訓的模型被用了幾個整點
        used = {}

        # 現在能預測到的最遠整點為止
        reachable_end = self.hour_now + train.LAG_HOURS * 3_600
        for timestamp in hours:
            if timestamp >= reachable_end:
                continue
            train_end = train.last_retrain_timestamp(
                recall.stored_at(timestamp, self.now_ts)
            )
            model = await self.model(train_end)

            # 當時沒有模型：之後重訓出模型、目標仍在未來時，每小時補存會補上
            while (
                model is None
                and train_end + 7 * 86_400 <= self.now_ts
                and train_end + 7 * 86_400 + 3_600 <= timestamp
            ):
                train_end += 7 * 86_400
                model = await self.model(train_end)

            result = None
            if model:
                result = await model.predict(timestamp)
            if result:
                prediction[timestamp] = result["q_cool_pred"]
                used[train_end] = used.get(train_end, 0) + 1

        for train_end, model in self.models.items():
            if model is None:
                print(
                    f">>> 模擬預測：{_short_time(train_end)} 重訓時"
                    f"{self.messages[train_end]}，案場當時沒有模型"
                )
        for train_end, hour_count in sorted(used.items()):
            print(
                f">>> 模擬預測：{hour_count} 個整點用 {_short_time(train_end)} 重訓的模型"
            )

        actual = await train.get_chiller_hourly(
            start_timestamp,
            min(end_timestamp, self.hour_now),
        )
        load_frame = pd.DataFrame(
            {
                "actual": actual.set_index("ts_hour")[train.TARGET_COLUMN],
                "pred": prediction,
            }
        ).reindex(hours)
        load_frame.index = settings.to_local_index(load_frame.index, naive=True)
        return load_frame

    # 確認現在已有模型；當時資料不足沒有模型時 API 回 404，不回傳空的建議
    async def _require_model(self) -> None:
        current = await self.model(train.last_retrain_timestamp(self.now_ts))
        if current is None:
            raise HTTPException(
                404,
                f"假設現在是 {self.now_text}，模型未建立，沒有預測",
            )

    # 資料來源：某天上班時段的預測、流量典型值與預測失敗原因（同正式服務介面；模擬不會失敗，為 None），
    # 產生控制建議用
    async def day_inputs(
        self,
        day: datetime.date,
    ) -> tuple[pd.Series, dict, str | None]:
        await self._require_model()
        prediction = await self.day(day)
        first_hour = int(prediction.index[0].timestamp())
        flows = await control.typical_flows(min(first_hour, self.now_ts))
        return prediction, flows, None

    # 資料來源：期間內逐時實際與預測（同正式 chart.get_load_frame），畫時序圖用
    async def get_load_frame(
        self,
        start_timestamp: int,
        end_timestamp: int,
    ) -> pd.DataFrame:
        await self._require_model()
        return await self.history(start_timestamp, end_timestamp)


# 某天各整點的實際負荷：只在指令列輸出時當事後對照，不參與預測
async def actual_load(index: pd.DatetimeIndex) -> pd.Series:
    start_timestamp = int(index[0].timestamp())
    end_timestamp = int(index[-1].timestamp()) + 3_600
    chiller_hourly = await train.get_chiller_hourly(start_timestamp, end_timestamp)
    return train.to_hourly_index(chiller_hourly)[train.TARGET_COLUMN].reindex(index)


# 負荷取整數顯示，缺值顯示「—」：指令列表格對齊用
def _format_load(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.0f}"


# 指令列：印出各日逐時預測與控制建議，快速確認某個時間點案場會給什麼建議
async def main(now_timestamp: int, segments: int) -> None:
    site = SimulatedSite(now_timestamp)
    print(f">>> 假設現在是 {site.now:%Y-%m-%d %H:%M}")
    print(
        f">>> 設定：出水 {rules.T_MIN}~{rules.T_MAX}°C、回水上限 {rules.T_RETURN}°C、"
        f"並聯門檻 {rules.PARALLEL_ABOVE} RT、尖峰門檻（超出加總）{control.PEAK_EXCESS:.0f} RT、台數至少維持 {rules.MIN_STAGE_HOURS} 小時、"
        f"{control.SEGMENT_NAMES[segments]}"
    )
    train_end = train.last_retrain_timestamp(now_timestamp)
    model = await site.model(train_end)
    print(
        f">>> 負荷模型（{settings.to_local(train_end):%m/%d %H:%M} 每週重訓）："
        f"{site.messages[train_end]}"
    )
    if model is None:
        return

    for day in control.plan_days(site.now):
        prediction = await site.day(day)
        first_hour = int(prediction.index[0].timestamp())
        flows = await control.typical_flows(min(first_hour, now_timestamp))
        actual = await actual_load(prediction.index)

        # 逐時預測與事後實際
        print(f"\n=== {day} ===")
        print(
            f"流量典型值 (LPM)：單機 {flows['single']:.0f}、"
            f"並聯 {flows['parallel']:.0f}"
        )
        print("時間   預測RT  實際RT（事後對照）")
        for hour in prediction.index:
            print(
                f"{hour:%H:%M}  {_format_load(prediction[hour]):>6}  "
                f"{_format_load(actual[hour]):>6}"
            )

        # 控制建議：今天從現在起重算
        today = day == site.now.date()
        if today:
            schedule_now = site.now
        else:
            schedule_now = None
        rows = control.schedule_from(
            prediction,
            flows,
            segments,
            schedule_now,
        )
        if today:
            print(f"控制建議（從現在 {site.now:%H:%M} 起，當段依剩餘時間重算）：")
        else:
            print("控制建議：")
        for row in rows:
            if pd.notna(row["supply_temp"]):
                temperature_text = f"{row['supply_temp']:.1f}°C"
            else:
                temperature_text = ""
            chillers_text = row["chillers"]
            if not chillers_text:
                chillers_text = "無建議"
            print(f"  {row['start']}~{row['end']}  {chillers_text}  {temperature_text}")


# 假設現在是 now_timestamp 的 API：與正式服務同一份（app.create_app），資料來源換成 SimulatedSite，
# 可在開發機用瀏覽器看當時的 API 結果
def create_app(now_timestamp: int) -> FastAPI:
    site = SimulatedSite(now_timestamp)

    # 啟動時先訓練現在的模型，確認模擬環境就緒
    @asynccontextmanager
    async def lifespan(_):
        train_end = train.last_retrain_timestamp(now_timestamp)
        await site.model(train_end)
        print(
            f">>> 假設現在是 {site.now_text}；"
            f"負荷模型（{settings.to_local(train_end):%m/%d %H:%M} 每週重訓）："
            f"{site.messages[train_end]}"
        )
        yield

    return app.create_app(
        site,
        lifespan,
        title=f"chiller_77 假設現在是 {site.now_text}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="指定過去時間當作現在，模擬負荷預測與控制建議（不寫入資料庫）"
    )
    parser.add_argument(
        "--now", required=True, help='台灣時間，例如 "2026-10-01 07:00"'
    )
    segment_names = [
        control.SEGMENT_NAMES[segment_count]
        for segment_count in sorted(control.SEGMENTS, reverse=True)
    ]
    parser.add_argument(
        "--segments",
        type=int,
        default=max(control.SEGMENTS),
        choices=sorted(control.SEGMENTS),
        help="一天分幾段控制：" + "、".join(segment_names),
    )
    parser.add_argument("--serve", action="store_true", help="開 API 服務，不印結果")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="API 位址（Docker 內用 0.0.0.0）",
    )
    parser.add_argument("--port", type=int, default=8001, help="API 埠號")
    args = parser.parse_args()

    # 指定時間當作台灣時間，換成 Unix 秒
    now_timestamp = int(
        pd.Timestamp(args.now).tz_localize(settings.settings.timezone).timestamp()
    )
    if args.serve:
        uvicorn.run(create_app(now_timestamp), host=args.host, port=args.port)
    else:
        asyncio.run(main(now_timestamp, args.segments))
