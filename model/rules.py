# 控制規則：由預測冷負荷決定開哪幾台、出水設多少（參數見 .env）
#   台數：依預測冷負荷決定運轉模式；< CH-2 最低負載（額定 400 RT × 最低負載率 15% = 60 RT）停機、
#         ≤ PARALLEL_ABOVE 單機 CH-2、> PARALLEL_ABOVE 兩台並聯；同一日內 ≤ MIN_STAGE_HOURS 小時、
#         前後為相同另一模式的時段併入前後模式（避免頻繁加卸載）
#   出水：出水 = T_RETURN − 負荷 ÷ (K × 流量)，限制於 T_MIN～T_MAX（預估回水維持在回水上限）；
#         並聯兩台同一設定
import numpy as np
import pandas as pd

import settings

_config = settings.settings
# 並聯門檻 (RT)
PARALLEL_ABOVE = _config.control_parallel_above
# 台數切換後至少維持幾小時 (h)
MIN_STAGE_HOURS = _config.control_min_stage_hours
# 出水下限 (°C)
T_MIN = _config.control_t_min
# 出水上限 (°C)
T_MAX = _config.control_t_max
# 回水上限 (°C)
T_RETURN = _config.control_t_return
# RT = K × LPM × °C
K = 4.186 / 60 / 3.517
# 出水設定精度（小數位數）
SETPOINT_DECIMALS = 1

# 主機：rated 額定 (RT)；min_lr 最低負載率 (%)；x、degree 為性能曲線形式（分析用）
CHILLERS = {
    "CH-1": {
        "id": "77館材化所CH1",
        "rated": 375,
        "x": "1/LR",
        "degree": 1,
        "min_lr": 20,
    },
    "CH-2": {
        "id": "77館材化所CH2",
        "rated": 400,
        "x": "LR",
        "degree": 3,
        "min_lr": 15,
    },
}

# 運轉方式
CH2_ONLY = "CH-2"
CH1_ONLY = "CH-1"
BOTH = "CH-1 + CH-2"
OFF = "停機"
TOO_LOW = "低於最低負載"
TOO_HIGH = "超過兩台容量"


# ---------- 台數 ----------


# 主機最低負載 (RT) = 額定 × 最低負載率：負荷低於它主機無法穩定運轉
def minimum_load_rt(chiller: dict) -> float:
    return chiller["rated"] * chiller["min_lr"] / 100


# 依負荷決定台數：低於 CH-2 最低負載停機、≤ 門檻單開 CH-2、> 門檻並聯（供冷優先）
def staging(load_rt) -> pd.Series:
    load = pd.Series(load_rt, dtype=float)

    # 先依並聯門檻分成單開 CH-2 或並聯
    mode_by_threshold = np.where(load > PARALLEL_ABOVE, BOTH, CH2_ONLY)
    plan = pd.Series(mode_by_threshold, index=load.index, dtype=object)

    # 低於 CH-2 最低負載停機；沒有負荷的小時為 NaN
    ch2_minimum_rt = minimum_load_rt(CHILLERS[CH2_ONLY])
    plan[load < ch2_minimum_rt] = OFF
    plan[load.isna()] = np.nan
    return plan


# 把一天的台數切成連續相同台數的時段：[(台數, 該時段的時間 index), ...]，找短暫跳動用
def _split_runs(day_plan: pd.Series) -> list:
    run_number = (day_plan != day_plan.shift()).cumsum()
    runs = []
    for _, run in day_plan.groupby(run_number):
        runs.append((run.iloc[0], run.index))
    return runs


# 找出可以併入前後的短時段（runs 中的位置）：單開 CH-2 或並聯、不超過 MIN_STAGE_HOURS 小時、
# 前後時段是相同的單開 CH-2 或並聯
def _short_run_positions(runs: list) -> list[int]:
    states = (CH2_ONLY, BOTH)
    positions = []
    for i in range(1, len(runs) - 1):
        state, hours = runs[i]
        previous_state = runs[i - 1][0]
        next_state = runs[i + 1][0]
        if state not in states:
            continue
        if len(hours) > MIN_STAGE_HOURS:
            continue
        if previous_state not in states:
            continue
        if previous_state != next_state:
            continue
        positions.append(i)
    return positions


# 決定先處理哪個短時段：最短的優先；一樣長時單開 CH-2 優先（改成並聯，供冷優先）；
# 條件相同時取最前面的時段
def _first_run_to_merge(runs: list, positions: list[int]) -> int:
    chosen_position = None
    chosen_order = None
    for position in positions:
        state, hours = runs[position]
        order = (len(hours), state != CH2_ONLY)
        if chosen_order is None or order < chosen_order:
            chosen_position = position
            chosen_order = order
    return chosen_position


# 避免台數短暫跳動（頻繁加卸載）：同一天內夾在前後相同台數之間、不超過 MIN_STAGE_HOURS 小時的
# 單開 CH-2 或並聯時段，改成前後的台數；plan 需為逐時、依時間排列
def hold_staging(plan: pd.Series) -> pd.Series:
    plan = plan.copy()
    for _, day in plan.groupby(plan.index.date):
        while True:
            day_plan = plan.loc[day.index]
            runs = _split_runs(day_plan)
            positions = _short_run_positions(runs)
            if not positions:
                break
            position = _first_run_to_merge(runs, positions)
            # 改成前一個時段的台數
            hours_to_merge = runs[position][1]
            plan.loc[hours_to_merge] = runs[position - 1][0]
    return plan


# ---------- 出水 ----------


# 出水設定取到設定精度（小數一位），現場設定用
def round_setpoint(temperature):
    return np.round(temperature, SETPOINT_DECIMALS)


# 出水建議 (°C)：讓預估回水維持在回水上限（出水 = 回水上限 − 負荷 ÷ (K × 流量)），
# 再限制在出水上下限；flow 為運轉主機流量合計 (LPM)
def supply_target(load, flow) -> np.ndarray:
    load = np.asarray(load, dtype=float)
    flow = np.asarray(flow, dtype=float)
    # 預估回水維持在回水上限，限制於出水上下限
    supply_temperature = np.clip(T_RETURN - load / (K * flow), T_MIN, T_MAX)
    return round_setpoint(supply_temperature)


# 負荷 → 運轉方式與出水設定（並聯兩台同設定）：控制建議與回測共用的規則；
# hold=True 時（負荷為逐時時間序列）避免台數短暫跳動
def control_rule(
    load,
    flows: dict,
    hold: bool = False,
    keep: pd.Series | None = None,
) -> pd.DataFrame:
    load = pd.Series(load, dtype=float)

    # 台數
    plan = staging(load)
    if hold:
        plan = hold_staging(plan)
        # keep 的小時（分段的尖峰段）維持依負荷決定的台數，不被合併
        if keep is not None:
            plan = plan.where(~keep, staging(load))

    # 出水設定：依台數取流量；停機或沒有負荷的小時為 NaN
    flow = np.where(plan == BOTH, flows["parallel"], flows["single"])
    supply_values = supply_target(load, flow)
    supply = pd.Series(supply_values, index=load.index)
    is_running = plan.isin([CH2_ONLY, BOTH])
    supply = supply.where(is_running)

    return pd.DataFrame(
        {
            "全館負荷 RT": load,
            "建議方案": plan,
            "出水設定 °C": supply,
            "CH-1 出水設定 °C": supply.where(plan == BOTH),
            "CH-2 出水設定 °C": supply,
        }
    )
