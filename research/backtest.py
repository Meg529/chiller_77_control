# 水側控制回測：照建議的台數與出水設定，用實際負荷推估耗電，估算能省多少電
# 耗電模型：CH-2 灰箱模型；CH-1 性能曲線 × 出水溫度修正（沿用 CH-2 敏感度）
# 各步驟逐步累加：S0 現況 → S1 台數照建議（出水維持現況）
# → S2 單機出水照建議 → S3 並聯出水照建議（兩台同設定）
import numpy as np
import pandas as pd

from model import rules
from research import chiller

STEPS = {"S0": "現況", "S1": "台數", "S2": "單機出水", "S3": "並聯出水"}
# 評估單位需要的實測欄位（CH-2 皆運轉）
UNIT_COLS = ["Q1", "Q2", "T1", "T2", "F1", "F2", "ON1", "METER"]
# 各步驟輸出欄位
OUT_COLS = ["CH-1 RT", "CH-2 RT", "CH-1 °C", "CH-2 °C", "CH-1 LPM", "CH-2 LPM", "kW"]


# 兩台耗電合計 (kW)；任一台超出運轉範圍為 NaN
# models 為 (性能曲線, CH-2 耗電模型, CH-1 出水敏感度)
def _total_kw(
    models: tuple,
    ch1_rt: float,
    ch2_rt: float,
    ch1_supply: float,
    ch2_supply: float,
    ch2_flow: float,
) -> float:
    curves, ch2_power_model, sensitivity = models
    if np.isnan(chiller.chiller_kw(curves["CH-2"], ch2_rt)):
        return np.nan

    if ch1_rt > 0:
        ch1_kw = float(chiller.ch1_kw(curves, ch1_rt, ch1_supply, sensitivity))
    else:
        ch1_kw = 0.0
    ch2_kw = float(chiller.ch2_kw(ch2_power_model, ch2_rt, ch2_supply, ch2_flow))
    return ch1_kw + ch2_kw


# 某運轉方式 mode 下的各機負荷、出水、流量與耗電；做不到（如超出運轉範圍）為 None
def _evaluate_mode(
    models: tuple,
    total_rt: float,
    mode: str,
    ch1_supply: float,
    ch2_supply: float,
    ch1_flow: float,
    ch2_flow: float,
) -> tuple | None:
    if np.isnan(ch2_supply):
        return None
    if mode == rules.BOTH and np.isnan(ch1_supply):
        return None

    if mode == rules.CH2_ONLY:
        ch1_rt = 0.0
        ch2_rt = total_rt
        ch1_supply = np.nan
        ch1_flow = 0.0
    else:
        ch1_split, ch2_split = chiller.actual_split(
            total_rt,
            ch1_supply,
            ch2_supply,
            ch1_flow,
            ch2_flow,
        )
        ch1_rt = float(ch1_split)
        ch2_rt = float(ch2_split)

    total_kw = _total_kw(models, ch1_rt, ch2_rt, ch1_supply, ch2_supply, ch2_flow)
    if np.isnan(total_kw):
        return None
    return (ch1_rt, ch2_rt, ch1_supply, ch2_supply, ch1_flow, ch2_flow, total_kw)


# 單一評估單位依序算 S0 現況 → S1 台數 → S2 單機出水 → S3 並聯出水的結果；現況無法評估時為空
def _evaluate_unit(
    models: tuple,
    unit,
    plan: str | None,
    setpoint: float,
    typical: dict | pd.Series,
) -> dict:
    if unit.ON1:
        actual_mode = rules.BOTH
        ch1_rt = unit.Q1
    else:
        actual_mode = rules.CH2_ONLY
        ch1_rt = 0.0

    # S0 現況
    actual_kw = _total_kw(models, ch1_rt, unit.Q2, unit.T1, unit.T2, unit.F2)
    if np.isnan(actual_kw):
        return {}
    if unit.ON1:
        ch1_supply = unit.T1
        ch1_flow = unit.F1
    else:
        ch1_supply = np.nan
        ch1_flow = 0.0
    step0 = (ch1_rt, unit.Q2, ch1_supply, unit.T2, ch1_flow, unit.F2, actual_kw)
    total_rt = ch1_rt + unit.Q2

    # S1 台數照建議，出水維持現況；台數與實際相同即為現況
    if plan in (rules.CH2_ONLY, rules.BOTH):
        mode = plan
    else:
        mode = actual_mode
    ch1_flow = unit.F1
    ch2_flow = unit.F2
    step1 = step0
    if mode != actual_mode:
        # 運轉方式改變：流量用該運轉方式的中位數；CH-1 出水用並聯時中位數
        if mode == rules.BOTH:
            ch1_flow = typical["F1"]
            ch2_flow = typical["F2"]
        else:
            ch1_flow = 0.0
            ch2_flow = typical["F2_single"]
        step1 = _evaluate_mode(
            models,
            total_rt,
            mode,
            typical["T1"],
            unit.T2,
            ch1_flow,
            ch2_flow,
        )
        # 建議的台數做不到（如超出 CH-2 單機範圍），沿用現況
        if step1 is None:
            mode = actual_mode
            ch1_flow = unit.F1
            ch2_flow = unit.F2
            step1 = step0

    # S2 單機出水照建議；做不到時沿用前一步
    step2 = None
    if mode == rules.CH2_ONLY:
        step2 = _evaluate_mode(
            models,
            total_rt,
            mode,
            np.nan,
            setpoint,
            ch1_flow,
            ch2_flow,
        )
    if step2 is None:
        step2 = step1

    # S3 並聯兩台同設定照建議；做不到時沿用前一步
    step3 = None
    if mode == rules.BOTH:
        step3 = _evaluate_mode(
            models,
            total_rt,
            mode,
            setpoint,
            setpoint,
            ch1_flow,
            ch2_flow,
        )
    if step3 is None:
        step3 = step2

    return {"S0": step0, "S1": step1, "S2": step2, "S3": step3}


# 回測主函式：各評估單位、各步驟的結果（欄位 "<步驟> <OUT_COLS>"）
# units：UNIT_COLS 與 HOUR（所屬小時）、W（權重：穩定運轉小時為 1，開停機小時分段為分鐘 ÷ 60）
# rec：control.recommend 的逐時建議；無建議的小時照實際
# typ：運轉方式改變時用的典型值 F1、F2（並聯時各機流量）、
#      F2_single（只開 CH-2 流量）、T1（並聯時 CH-1 出水）
def evaluate(
    units: pd.DataFrame,
    rec: pd.DataFrame,
    curves: dict,
    model,
    sens: float,
    typ: dict | pd.Series,
) -> pd.DataFrame:
    models = (curves, model, sens)
    plans = rec["建議方案"].reindex(units.HOUR).to_numpy()
    setpoints = rec["出水設定 °C"].reindex(units.HOUR).to_numpy(dtype=float)

    rows = []
    for unit, plan, setpoint in zip(units.itertuples(), plans, setpoints):
        # 無建議（NaN）的小時照實際
        if isinstance(plan, str):
            unit_plan = plan
        else:
            unit_plan = None
        cases = _evaluate_unit(models, unit, unit_plan, setpoint, typ)

        row = {}
        for step, case in cases.items():
            for column, value in zip(OUT_COLS, case):
                row[f"{step} {column}"] = value
        rows.append(row)

    columns = []
    for step in STEPS:
        for column in OUT_COLS:
            columns.append(f"{step} {column}")
    return pd.DataFrame(rows, index=units.index, columns=columns)


# 各評估單位的節電 (kWh) = (prev 耗電 − step 耗電) × 權重；無法評估為 0
def saving(
    res: pd.DataFrame,
    units: pd.DataFrame,
    step: str,
    prev: str = "S0",
) -> pd.Series:
    kw_saved = res[f"{prev} kW"] - res[f"{step} kW"]
    return kw_saved.fillna(0) * units.W


# 評估單位的值依所屬小時加總，畫逐時圖用
def by_hour(values: pd.Series, units: pd.DataFrame) -> pd.Series:
    return values.groupby(units.HOUR).sum()


# 各評估單位在 step 下回水比回水上限高幾度：檢查建議出水會不會讓回水超標
def return_excess(res: pd.DataFrame, step: str = "S3") -> pd.Series:
    ch1_flow = res[f"{step} CH-1 LPM"]
    ch2_flow = res[f"{step} CH-2 LPM"]
    ch1_supply = res[f"{step} CH-1 °C"].fillna(0)
    ch2_supply = res[f"{step} CH-2 °C"]

    # 混合供水
    mixed_supply = (ch1_flow * ch1_supply + ch2_flow * ch2_supply) / (
        ch1_flow + ch2_flow
    )
    total_rt = res[f"{step} CH-1 RT"] + res[f"{step} CH-2 RT"]
    return mixed_supply + total_rt / (rules.K * (ch1_flow + ch2_flow)) - rules.T_RETURN
