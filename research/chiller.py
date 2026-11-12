# 主機性能曲線與耗電估算（只給分析 notebook 用）
# 性能曲線：CH-1 kW/RT = a + b / 負載率；CH-2 kW/RT 為負載率三次式（以負載率 ≤ 100% 穩定運轉資料擬合）
# decide、best_split 依效率選開法並做最佳負荷分配，僅作理論對照，不用於控制
import numpy as np
import pandas as pd

from data import sensor
from model.rules import (
    BOTH,
    CH1_ONLY,
    CH2_ONLY,
    CHILLERS,
    K,
    OFF,
    TOO_HIGH,
    TOO_LOW,
    minimum_load_rt,
)
from research import ch2_model

MAX_LR = 100  # 負載率上限 (%)
T_FULL_PCT = 10  # 資料推估滿載設定：CH-2 負荷 ≥ LOAD_LIMIT_RT 時出水溫度百分位
CH1_REF_T = 11.0  # CH-1 性能曲線對應出水溫度 (°C)


# ---------- 性能曲線 ----------


# 挑出穩定運轉資料並算負載率 LR (%) 與 kW/RT：擬合性能曲線只用穩定資料
def steady(df: pd.DataFrame, rated: float) -> pd.DataFrame:
    is_steady = (df["kW"] > 100) & (df["Q_COOL"] > 50)
    steady_rows = df[is_steady].copy()
    steady_rows["LR"] = steady_rows["Q_COOL"] / rated * 100
    steady_rows["KW_PER_RT"] = steady_rows["kW"] / steady_rows["Q_COOL"]
    return steady_rows


# 曲線自變數：form 為 "1/LR" 時用負載率倒數（CH-1），否則用負載率（CH-2）
def _curve_variable(load_ratio, form: str) -> np.ndarray:
    load_ratio = np.asarray(load_ratio, dtype=float)
    with np.errstate(divide="ignore"):
        if form == "1/LR":
            return 1 / load_ratio
        return load_ratio


# 依性能曲線算某負載率下的效率 kW/RT
def kw_per_rt(curve: dict, lr) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        curve_variable = _curve_variable(lr, curve["x"])
        return np.polyval(curve["coef"], curve_variable)


# 擬合單台主機性能曲線（負載率 ≤ 100% 的穩定資料），並算擬合優度 R²
def fit_curve(df: pd.DataFrame, name: str) -> dict:
    config = CHILLERS[name]
    steady_rows = steady(df, config["rated"])
    steady_rows = steady_rows[steady_rows["LR"] <= MAX_LR]

    coefficients = np.polyfit(
        _curve_variable(steady_rows["LR"], config["x"]),
        steady_rows["KW_PER_RT"],
        config["degree"],
    )
    curve = {
        **config,
        "name": name,
        "coef": coefficients,
        "lr_min": steady_rows["LR"].min(),
        "lr_max": steady_rows["LR"].max(),
        "n": len(steady_rows),
    }

    # 擬合優度 R²
    residual = steady_rows["KW_PER_RT"] - kw_per_rt(curve, steady_rows["LR"])
    residual_square_sum = (residual**2).sum()
    mean_kw_per_rt = steady_rows["KW_PER_RT"].mean()
    total_square_sum = ((steady_rows["KW_PER_RT"] - mean_kw_per_rt) ** 2).sum()
    curve["r2"] = 1 - residual_square_sum / total_square_sum
    return curve


# 讀兩台主機資料並各自擬合性能曲線
async def load_curves(
    start_time: int | None = None, end_time: int | None = None
) -> dict:
    curves = {}
    for name, config in CHILLERS.items():
        chiller_data = await sensor.load_chiller(config["id"], start_time, end_time)
        curves[name] = fit_curve(chiller_data, name)
    return curves


# 單台耗電 (kW) = 負荷 × kW/RT；超出運轉範圍（低於最低負載或超過滿載）為 NaN
def chiller_kw(curve: dict, rt) -> np.ndarray:
    rt = np.asarray(rt, dtype=float)
    load_ratio = rt / curve["rated"] * 100
    efficiency = kw_per_rt(curve, load_ratio)
    in_range = (
        (efficiency > 0) & (load_ratio >= curve["min_lr"]) & (load_ratio <= MAX_LR)
    )
    return np.where(in_range, rt * efficiency, np.nan)


# 是否超出曲線擬合資料的負載率範圍（超出時是外推，結果較不可靠）
def outside(curve: dict, rt) -> np.ndarray:
    load_ratio = np.asarray(rt, dtype=float) / curve["rated"] * 100
    return (load_ratio < curve["lr_min"]) | (load_ratio > curve["lr_max"])


# 並聯時兩台怎麼分負荷最省電（每 1 RT 搜尋）：回傳 (CH-1 RT, CH-2 RT, kW)
def best_split(curves: dict, load: float) -> tuple:
    ch1_lowest = np.ceil(minimum_load_rt(curves["CH-1"]))
    ch1_highest = min(
        load - minimum_load_rt(curves["CH-2"]),
        curves["CH-1"]["rated"],
    )
    if not ch1_highest >= ch1_lowest:
        return np.nan, np.nan, np.nan

    # CH-1 每 1 RT 一個候選，其餘由 CH-2 負擔
    ch1_rt = np.arange(ch1_lowest, ch1_highest + 0.5)
    ch1_kw_values = chiller_kw(curves["CH-1"], ch1_rt)
    ch2_kw_values = chiller_kw(curves["CH-2"], load - ch1_rt)
    total_kw = ch1_kw_values + ch2_kw_values
    if np.isnan(total_kw).all():
        return np.nan, np.nan, np.nan

    best_index = np.nanargmin(total_kw)
    return ch1_rt[best_index], load - ch1_rt[best_index], total_kw[best_index]


# 依性能曲線挑各負荷耗電最低的運轉方式（理論對照，不用於控制）；parallel_above 以上強制並聯
def decide(
    load_rt,
    curves: dict,
    parallel_above: float | None = None,
) -> pd.DataFrame:
    load = pd.Series(load_rt, dtype=float)

    # 並聯最佳分配與單機耗電
    split = pd.DataFrame(
        [best_split(curves, load_value) for load_value in load],
        index=load.index,
        columns=["ch1", "ch2", "kw"],
    )
    single_kw = {}
    for name in ["CH-2", "CH-1"]:
        single_kw[name] = chiller_kw(curves[name], load)
    if parallel_above is not None:
        for name in ["CH-2", "CH-1"]:
            single_kw[name] = np.where(load > parallel_above, np.nan, single_kw[name])

    result = pd.DataFrame(
        {
            "全館負荷 RT": load,
            f"{CH2_ONLY} kW": single_kw["CH-2"],
            f"{CH1_ONLY} kW": single_kw["CH-1"],
            f"{BOTH} kW": split["kw"],
        },
        index=load.index,
    )

    # 選耗電最低的方案：0 單開 CH-2、1 單開 CH-1、2 並聯
    options = result[[f"{CH2_ONLY} kW", f"{CH1_ONLY} kW", f"{BOTH} kW"]]
    feasible = options.notna().any(axis=1)
    choice = options.fillna(np.inf).to_numpy().argmin(axis=1)
    running = load > 0

    plan_names = np.array([CH2_ONLY, CH1_ONLY, BOTH])
    result["建議方案"] = np.where(running, plan_names[choice], OFF)
    result["CH-1 RT"] = np.select(
        [choice == 1, choice == 2],
        [load, split["ch1"]],
        0.0,
    )
    result["CH-2 RT"] = np.select(
        [choice == 0, choice == 2],
        [load, split["ch2"]],
        0.0,
    )
    for name in ["CH-1", "CH-2"]:
        result[f"{name} 負載率 %"] = result[f"{name} RT"] / curves[name]["rated"] * 100
    result["預估耗電 kW"] = options.to_numpy()[np.arange(len(result)), choice]

    # 任一台超出曲線資料範圍即為外推
    ch1_outside = (result["CH-1 RT"] > 0) & outside(curves["CH-1"], result["CH-1 RT"])
    ch2_outside = (result["CH-2 RT"] > 0) & outside(curves["CH-2"], result["CH-2 RT"])
    result["外推"] = running & feasible & (ch1_outside | ch2_outside)

    # 停機記 0；不可行不給建議
    stopped_columns = [
        "CH-1 RT",
        "CH-2 RT",
        "CH-1 負載率 %",
        "CH-2 負載率 %",
        "預估耗電 kW",
    ]
    result.loc[~running, stopped_columns] = 0.0
    infeasible = running & ~feasible
    no_advice_rows = load.isna() | infeasible
    advice_columns = result.columns.drop(["全館負荷 RT", "外推"])
    result.loc[no_advice_rows, advice_columns] = np.nan

    # 不可行時，低於兩台最小負荷為負荷太低，否則為負荷太高
    minimum_loads = [minimum_load_rt(curve) for curve in curves.values()]
    too_low = infeasible & (load < min(minimum_loads))
    result.loc[too_low, "建議方案"] = TOO_LOW
    result.loc[infeasible & ~too_low, "建議方案"] = TOO_HIGH
    return result


# ---------- 耗電估算 ----------


# 由資料推估滿載設定 (°C)：CH-2 負荷 ≥ LOAD_LIMIT_RT 時出水溫度的第 T_FULL_PCT 百分位
def full_load_supply(df: pd.DataFrame) -> float:
    is_full_load = sensor.is_running(df) & (df["Q_COOL"] >= ch2_model.LOAD_LIMIT_RT)
    full_load_rows = df[is_full_load]
    return float(full_load_rows["T_CHWS"].quantile(T_FULL_PCT / 100))


# 兩台共用回水時各自實際負擔的負荷 (RT)；同設定時依流量比例分配
# q 全館負荷、t1 / t2 兩台出水溫度、f1 / f2 兩台流量
def actual_split(q, t1, t2, f1, f2) -> tuple:
    return_temperature = (q / K + f1 * t1 + f2 * t2) / (f1 + f2)
    ch1_rt = K * f1 * (return_temperature - t1)
    ch2_rt = K * f2 * (return_temperature - t2)
    return ch1_rt, ch2_rt


# CH-2 耗電 (kW)：用灰箱模型估，回測各情境耗電用；q 冷凍噸、t 出水溫度、flow 流量
def ch2_kw(model, q, t, flow) -> np.ndarray:
    cooling_load, supply_temperature, flow_rate = np.broadcast_arrays(
        np.asarray(q, dtype=float),
        np.asarray(t, dtype=float),
        np.asarray(flow, dtype=float),
    )
    features = pd.DataFrame(
        {
            "FLOW": flow_rate.ravel(),
            "DT": (cooling_load / (K * flow_rate)).ravel(),
            "T_CHWS": supply_temperature.ravel(),
        }
    )
    prediction = model.predict(ch2_model.graybox_x(features))
    return prediction.reshape(cooling_load.shape)


# CH-1 耗電 (kW)：性能曲線（出水 CH1_REF_T）× 出水溫度修正；
# sens 沿用 CH-2 敏感度（出水每調高 1°C 耗電減少的比例），待型錄或試行驗證
def ch1_kw(curves: dict, q, t, sens: float) -> np.ndarray:
    reference_kw = chiller_kw(curves["CH-1"], q)
    supply_temperature = np.asarray(t, dtype=float)
    correction = 1 + sens * (CH1_REF_T - supply_temperature)
    return reference_kw * correction
