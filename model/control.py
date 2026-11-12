# 控制建議：依當日上班時段負荷預測，算出各時段開哪幾台、出水設定多少；
# 是 /recommendation 的核心
import numpy as np
import pandas as pd

import settings
from data import sensor
from model import recall, rules

# 上班時段 08:00~17:00（08~16 時各整點）
WORK_START = 8
WORK_END = 16
# 一天分幾段：各段起始整點（最後一段到 17:00）
SEGMENTS = {3: (8, 11, 14), 2: (8, 12), 1: (8,)}
SEGMENT_NAMES = {
    3: "3 段（每 3 小時）",
    2: "2 段（上午 4 小時、下午 5 小時）",
    1: "1 段（整天 9 小時）",
}
# 控制依據百分位
LOAD_PERCENTILE = settings.settings.control_load_pct
# 尖峰門檻 (RT)：並聯門檻 × 尖峰比例；段內各小時超過並聯門檻的部分加總超過它時，整段改依最高那小時
PEAK_EXCESS_RATIO = settings.settings.control_peak_excess
PEAK_EXCESS = rules.PARALLEL_ABOVE * PEAK_EXCESS_RATIO

# 流量典型值統計天數
FLOW_DAYS = 14
# 無資料時的預設流量 (LPM)
DEFAULT_FLOWS = {"single": 3852, "parallel": 6088}


# 一段的控制依據負荷：段內預測負荷第 percentile 百分位；段內各小時超過並聯門檻的部分加總
# 超過 PEAK_EXCESS 時，改用最高那小時（兩台並聯），避免尖峰時段供冷不足；段內有缺值就不給建議（NaN）。
# 回傳 (控制依據負荷, 是否因尖峰改用最高小時)
def segment_basis(
    segment_values: pd.Series,
    percentile: float = LOAD_PERCENTILE,
) -> tuple[float, bool]:
    if len(segment_values) == 0 or segment_values.isna().any():
        return np.nan, False
    basis = segment_values.quantile(percentile / 100)
    # 各小時超過並聯門檻的部分加總
    excess = (segment_values - rules.PARALLEL_ABOVE).clip(lower=0).sum()
    if excess > PEAK_EXCESS:
        return segment_values.max(), True
    return basis, False


# 逐時控制建議（同一段內相同）：各段依 segment_basis 決定台數與出水；
# segments=None 為逐時；上班時段外為 NaN
def recommend(
    prediction: pd.Series,
    flows: dict,
    segments: int | None = None,
    percentile: float | None = None,
) -> pd.DataFrame:
    hour = prediction.index.hour
    work_prediction = prediction[(hour >= WORK_START) & (hour <= WORK_END)]
    # 尖峰段的小時：台數不被「至少維持 3 小時」合併掉
    is_peak = pd.Series(False, index=prediction.index)
    if segments is None:
        load = work_prediction.reindex(prediction.index)
    else:
        # 每個小時屬於第幾段
        segment_number = (
            np.searchsorted(
                SEGMENTS[segments],
                work_prediction.index.hour,
                side="right",
            )
            - 1
        )

        if percentile is None:
            percentile = LOAD_PERCENTILE

        # 每天每段算一次控制依據，填到段內各小時
        load = pd.Series(np.nan, index=prediction.index)
        groups = work_prediction.groupby([work_prediction.index.date, segment_number])
        for _, segment_values in groups:
            basis, triggered = segment_basis(segment_values, percentile)
            load[segment_values.index] = basis
            is_peak[segment_values.index] = triggered

    recommendation = rules.control_rule(load, flows, hold=True, keep=is_peak)
    return recommendation.set_axis(prediction.index)


# start_time ~ end_time 的流量中位數 (LPM)：單機取 CH-2、並聯取兩台合計；
# 只用完整運轉小時（前後一小時都在運轉），排除開停機時不穩定的流量
async def _median_flows(start_time: int | None, end_time: int) -> dict:
    hourly_by_chiller = await sensor.get_chiller_hourly_power_flow(
        start_time,
        end_time,
    )
    flow_by_chiller = {}
    for chiller_name, chiller_hourly in hourly_by_chiller.items():
        if chiller_hourly.empty:
            flow_by_chiller[chiller_name] = pd.Series(dtype=float)
            continue

        # 補齊連續逐時
        first_hour = int(chiller_hourly.index.min())
        last_hour = int(chiller_hourly.index.max())
        chiller_hourly = chiller_hourly.reindex(range(first_hour, last_hour + 1, 3_600))

        # 完整運轉小時：前後一小時均運轉
        running = sensor.is_running(chiller_hourly)
        previous_running = running.shift(1, fill_value=False)
        next_running = running.shift(-1, fill_value=False)
        full_hours = running & previous_running & next_running
        flow_by_chiller[chiller_name] = chiller_hourly["FLOW"].where(full_hours, 0.0)

    flows = pd.DataFrame(flow_by_chiller, columns=list(hourly_by_chiller))
    flows = flows.fillna(0.0)
    ch1_flow = flows["CH-1"]
    ch2_flow = flows["CH-2"]

    ch1_running = ch1_flow > sensor.RUNNING_FLOW_LPM
    ch2_running = ch2_flow > sensor.RUNNING_FLOW_LPM
    single = ch2_flow[ch2_running & ~ch1_running].median()
    parallel = (ch1_flow + ch2_flow)[ch1_running & ch2_running].median()
    return {"single": single, "parallel": parallel}


# 流量典型值：出水設定要用流量換算溫差，取近 days 天完整運轉小時的流量中位數；
# 近期沒有資料時依序改用全部歷史、DEFAULT_FLOWS
async def typical_flows(end_time: int, days: int = FLOW_DAYS) -> dict:
    recent = await _median_flows(end_time - days * 86_400, end_time)
    if all(pd.notna(value) for value in recent.values()):
        return recent

    history = await _median_flows(None, end_time)

    # 各流量依序取第一個有值的：近期、全部歷史、預設值
    flows = {}
    for key in DEFAULT_FLOWS:
        for candidate in (recent[key], history[key], DEFAULT_FLOWS[key]):
            if pd.notna(candidate):
                flows[key] = candidate
                break
    return flows


# 上班時段各整點
WORK_HOURS = range(WORK_START, WORK_END + 1)


# 指定日期上班時段各整點（台灣時間）：預測與控制建議只看這些小時
def work_hours(day) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([settings.at_hour(day, hour) for hour in WORK_HOURS])


# 要排控制建議的日期：當天；下班後再加隔天，隔天一早就有預測
def plan_days(now: pd.Timestamp) -> list:
    days = [now.date()]
    if now.hour > WORK_END:
        tomorrow = (now + pd.Timedelta(days=1)).date()
        days.append(tomorrow)
    return days


# 整理一段控制建議的輸出欄位：起訖 "HH:MM"、運轉方式、台數、出水設定；
# 沒有建議時後三項都是 None，API 才能輸出 null
def _plan_row(start: str, end: int, plan, supply_temp: float) -> dict:
    # 沒有建議方案（NaN）時，運轉方式、台數、出水設定都記 None（API 輸出 null）
    if not isinstance(plan, str):
        plan = None
    if pd.isna(supply_temp):
        supply_temp = None
    else:
        supply_temp = float(supply_temp)
    chiller_counts = {rules.CH2_ONLY: 1, rules.BOTH: 2, rules.OFF: 0}
    return {
        "start": start,
        "end": f"{end:02d}:00",
        "chillers": plan,
        "chiller_count": chiller_counts.get(plan),
        "supply_temp": supply_temp,
    }


# 從現在開始的各段建議（/recommendation 用）：現在所在那一段從現在起、只依「現在到該段結束」
# 的剩餘小時重算，已過去的段不列；上班前或 now=None 列整天；下班後為空；各段 start、end 為 "HH:MM"
def schedule_from(
    prediction: pd.Series,
    flows: dict,
    segments: int,
    now: pd.Timestamp | None = None,
) -> list[dict]:
    recommendation = recommend(prediction, flows, segments)

    # 各段起訖整點
    starts = SEGMENTS[segments]
    ends = starts[1:] + (WORK_END + 1,)

    # 整天各段建議：同一段內建議相同，取段首整點
    plan = []
    for start, end in zip(starts, ends):
        segment_first_hour = recommendation[recommendation.index.hour == start].iloc[0]
        plan.append(
            _plan_row(
                f"{start:02d}:00",
                end,
                segment_first_hour["建議方案"],
                segment_first_hour["出水設定 °C"],
            )
        )

    # 上班前：列整天
    if now is None or now.hour < WORK_START:
        return plan

    # 下班後：沒有建議
    if now.hour > WORK_END:
        return []

    # 現在所在那一段的結束整點
    current_end = None
    for start, end in zip(starts, ends):
        if start <= now.hour < end:
            current_end = end
            break

    # 只依現在到該段結束的剩餘小時重算
    prediction_hour = prediction.index.hour
    is_remaining = (prediction_hour >= now.hour) & (prediction_hour < current_end)
    load, _ = segment_basis(prediction[is_remaining])
    current_rule = rules.control_rule([load], flows).iloc[0]
    current_row = _plan_row(
        f"{now:%H:%M}",
        current_end,
        current_rule["建議方案"],
        current_rule["出水設定 °C"],
    )

    # 現在這一段（從現在起）＋之後各段
    rows = [current_row]
    for start, row in zip(starts, plan):
        if start > now.hour:
            rows.append(row)
    return rows


# 指定日期上班時段的逐時預測負荷與流量典型值：產生建議的兩項輸入
async def day_inputs(day) -> tuple[pd.Series, dict]:
    prediction = await recall.load_forecast(day, WORK_HOURS)
    flows = await typical_flows(int(prediction.index[0].timestamp()))
    return prediction, flows
