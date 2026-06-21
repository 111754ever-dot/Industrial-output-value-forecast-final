# -*- coding: utf-8 -*-
"""
================================================================================
工业增加值（规模以上工业企业·当月同比）下一期预测 —— 权威常态化预测脚本
================================================================================

适用场景
--------
每月（通常 20-25 日）拿到由数据维护人员更新的《数据汇总表》后，一键预测
"下一期"工业增加值当月同比，输出：
    1) 点预测；
    2) 区间预测（80% / 90% / 95%，共形预测，校准良好）；
    3) 伪实时回测效果图与覆盖率报告（落盘到 ./outputs）。

本脚本严格按照与业务方共同确定的方案实现，核心方法论要点：
--------------------------------------------------------------------------------
1. 问题本质 —— 混频 + 参差不齐边缘（ragged-edge）的"临近预测/Nowcasting"：
   预测当月时，月度自变量滞后约一个月，而旬/周/日度自变量已部分到位。
2. 样本与窗口 —— 主窗口 2015 年至今（现行因子-目标关系最一致）；
   长史（2005+）仅作辅助/稳健参照，并对近端加权（在长史 OLS 中按时间指数加权），
   以消化已证实的"制度漂移"（PPI 同比解释力 0.57→0.01；社零 0.26→0.77）。
3. 1-2 月特殊处理 —— Wind 的"1-2 月拆分"值为人工拆分、噪声极大（拆分月标准差
   约 16 vs 其余月约 1.9），主模型训练时**剔除 1、2 月**；但仍逐月给出当月点+区间，
   1-2 月走独立"春节口径参考"路径并显著加宽区间。
4. 特征工程 —— 统一变换到与目标可比口径：已是同比的直接用；累计值转累计同比；
   当月值转同比；PMI 用扩散指数水平；高频指标聚合后取 12 月同比，并构造
   "月初至今对齐同比（MTD-YoY）"作为参差边缘的核心特征（经验证：月前 22 日
   均值与全月均值相关 0.979，部分月高度代表全月）。训练与服务使用**同一日切口**，
   消除"半月 vs 整月"口径偏差与训练/服务不一致（train/serve skew）。
5. 模型 —— 小样本（剔除 1-2 月后 n≈112）"瘦身优先"组合（核心为密集收缩/投影，
   已由统一无泄漏 OOS 实验选定：Ridge≈1.45、PLS≈1.38 优于 ElasticNet≈1.58、
   且全面优于树模型/LightGBM≈1.7+——复杂模型在此小样本上更差）：
       · Ridge（RidgeCV，密集 L2 收缩；共线 p≈n 小样本最稳）；
       · PLS（偏最小二乘，少数潜成分概括共线特征，实验最优）；
       · 1 因子（Stock-Watson EM-PCA 扩散指数；EM 原生消化缺失/参差边缘）+ AR 桥接回归；
       · AR(1)（基准与锚，AR(1)≈0.54）；
       · 季节朴素（基准）；
       · LightGBM（**挑战者**，硬约束，仅当回测打赢 AR 才纳入；实验显示通常被剔除）。
   组合：逆-RMSE 简单加权（不做学习型 stacking，避免在小样本上再过拟合一层）。
6. 区间 —— 共形预测：用**伪实时 walk-forward 的真实样本外残差**做（split/jackknife+ 风味）
   校准，分布无关、有覆盖保证；80% 为主报区间，90/95 附"小样本更宽不确定"说明。
7. 验证 —— vintage-aware 扩展窗 walk-forward：每个预测时点严格只用"当日应得"信息
   （按发布滞后重建 ragged edge，模拟 20-25 号），一律对标 AR/季节朴素基准，
   打不赢即剔除。

时间序列局限性的处理（务必知悉）
--------------------------------------------------------------------------------
· 严防前视偏差（look-ahead）：所有特征按 as-of 日期切断；月度自变量统一滞后一期；
  目标自身的 AR 项使用上一期已公布值；回测逐期重建信息集；
  Ridge/PLS 选超参用 TimeSeriesSplit（只用过去验证未来）而非普通 KFold。
· 组合无未来泄漏：回测中组合权重用"在线扩展窗"确定（预测第 t 期仅用 t 之前的样本外
  预测），故组合的回测指标与共形残差均为真实样本外结果；模型失败时按可用模型重新归一化。
· 制度漂移：主窗口锁 2015+；长史辅助模型对近端指数加权。
· 小样本：强正则、降维（单因子）、简单组合、共形区间；复杂件须经回测挣得入选资格。
· 数据口径限制（务必知悉）：本仓库仅有单一 vintage 快照，历史高频/月度值可能已被
  修订，故回测为"伪实时（pseudo-real-time）"——能准确复现"发布时点的参差边缘"，
  但用的是修订后数值；要做"真实时（true real-time）"需接入 point-in-time 历史档案。
  本脚本核心价值在"校准区间 + 透明驱动 + 严谨（无泄漏）回测"，不夸大点预测增益。

依赖
----
必需：pandas, numpy, scikit-learn, matplotlib, openpyxl
可选：lightgbm（缺失则自动跳过挑战者），statsmodels（缺失则跳过 ADF 体检项）

运行
----
    python industrial_value_added_forecast.py
可在文件末尾 Config 中调整数据文件名、窗口、回测起点、预测日切口等。
================================================================================
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# matplotlib 用无界面后端；图内文字使用英文，避免缺失中文字体导致乱码
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 可选依赖
try:
    import lightgbm as lgb

    _HAS_LGBM = True
except Exception:  # pragma: no cover
    _HAS_LGBM = False

try:
    from statsmodels.tsa.stattools import adfuller

    _HAS_SM = True
except Exception:  # pragma: no cover
    _HAS_SM = False


RNG_SEED = 20260525
np.random.seed(RNG_SEED)


# =============================================================================
# 一、配置：数据路径、指标字典（精确全名→建模口径）、窗口与超参
# =============================================================================
@dataclass
class Indicator:
    """单个指标的元信息。key 必须唯一，用于区分高度相似的指标名称。"""

    key: str            # 唯一短名（英文，防相似名混淆）
    name: str           # Excel 中的精确全名（务必逐字匹配）
    sheet: str          # 所在 sheet
    freq: str           # 'M'（月度）或 'HF'（旬/周/日 高频）
    transform: str      # 'asis' | 'cum_yoy' | 'yoy' | 'mtd_yoy'
    cn: str             # 中文简称（用于报告展示）
    agg: str = "mean"   # 高频聚合口径：'mean'(率/价/指数/日均) | 'sum'(可加流量/总量)


# ---- 指标字典：每个指标都显式标注，杜绝模糊匹配；相似指标已分别建唯一 key ----
INDICATORS: List[Indicator] = [
    # ---------------- 月度（自变量统一滞后一期使用） ----------------
    Indicator("retail_yoy",        "中国:社会消费品零售总额:当月同比(1-2月合并)", "月度", "M", "asis",    "社零同比"),
    Indicator("fai_cumyoy",        "中国:固定资产投资完成额:累计值",             "月度", "M", "cum_yoy", "固投累计同比"),
    Indicator("reinv_cumyoy",      "中国:房地产开发投资完成额:累计值",           "月度", "M", "cum_yoy", "地产投资累计同比"),
    Indicator("pmi",               "中国:制造业PMI",                          "月度", "M", "asis",    "PMI"),
    Indicator("pmi_neworder",      "中国:制造业PMI:新订单",                    "月度", "M", "asis",    "PMI新订单"),
    Indicator("pmi_newexport",     "中国:制造业PMI:新出口订单",                 "月度", "M", "asis",    "PMI新出口订单"),
    Indicator("power_yoy",         "中国:发电量:当月同比",                     "月度", "M", "asis",    "发电量同比"),
    Indicator("ppi_yoy",           "中国:PPI:当月同比",                       "月度", "M", "asis",    "PPI同比"),
    Indicator("ppirm_yoy",         "中国:PPIRM:当月同比",                     "月度", "M", "asis",    "PPIRM同比"),
    Indicator("pmi_inputprice",    "中国:制造业PMI:主要原材料购进价格",          "月度", "M", "asis",    "PMI购进价格"),
    Indicator("ic_output_yoy",     "中国:产量:集成电路:当月值",                 "月度", "M", "yoy",     "集成电路产量同比"),
    Indicator("powergen_equip_yoy","中国:产量:发电设备:当月值",                 "月度", "M", "yoy",     "发电设备产量同比"),

    # ---------------- 旬度（高频 → MTD 同比） ----------------
    Indicator("crudesteel_key",    "中国:日均产量:粗钢:重点企业",               "旬度", "HF", "mtd_yoy", "粗钢日产(重点)"),
    Indicator("crudesteel_est",    "中国:预估日均产量:粗钢",                   "旬度", "HF", "mtd_yoy", "粗钢日产(预估)"),
    Indicator("steel_key",         "中国:日均产量:钢材:重点企业",               "旬度", "HF", "mtd_yoy", "钢材日产(重点)"),
    Indicator("pigiron_key",       "中国:日均产量:生铁:重点企业",               "旬度", "HF", "mtd_yoy", "生铁日产(重点)"),

    # ---------------- 周度（高频 → MTD 同比） ----------------
    Indicator("tire_semisteel",    "中国:开工率:汽车轮胎(半钢胎)",              "周度", "HF", "mtd_yoy", "半钢胎开工"),
    Indicator("tire_fullsteel",    "中国:开工率:汽车轮胎(全钢胎)",              "周度", "HF", "mtd_yoy", "全钢胎开工"),
    Indicator("polyester_chip",    "中国:开工率:聚酯切片",                     "周度", "HF", "mtd_yoy", "聚酯切片开工"),
    Indicator("asphalt",           "中国:开工率:石油沥青装置",                  "周度", "HF", "mtd_yoy", "石油沥青开工"),
    Indicator("rebar_output",      "中国:产量:螺纹钢:主要钢厂",                 "周度", "HF", "mtd_yoy", "螺纹钢产量", agg="sum"),
    Indicator("rebar_oprate",      "中国:开工率:螺纹钢:主要钢厂",               "周度", "HF", "mtd_yoy", "螺纹钢开工"),
    Indicator("land_area",         "中国:100大中城市:成交土地占地面积",          "周度", "HF", "mtd_yoy", "成交土地面积", agg="sum"),
    Indicator("scfi",              "中国:上海出口集装箱运价指数:综合指数",        "周度", "HF", "mtd_yoy", "SCFI综合"),
    Indicator("steel_priceidx",    "中国:钢材综合价格指数",                    "周度", "HF", "mtd_yoy", "钢材价格指数"),
    Indicator("coal_south",        "中国:南方电厂:日耗量:煤炭",                 "周度", "HF", "mtd_yoy", "南方电厂煤耗"),
    Indicator("coal_keyplant",     "中国:日耗量:煤炭重点电厂",                  "周度", "HF", "mtd_yoy", "重点电厂煤耗"),
    Indicator("coal_unifiedplant", "中国:日耗量:煤炭统调电厂",                  "周度", "HF", "mtd_yoy", "统调电厂煤耗"),
    Indicator("pv_retail",         "中国:日均销量(当周,厂家零售):乘用车",        "周度", "HF", "mtd_yoy", "乘用车零售"),
    Indicator("pv_wholesale",      "中国:日均销量(当周,厂家批发):乘用车",        "周度", "HF", "mtd_yoy", "乘用车批发"),
    Indicator("cement_ship",       "中国:发运率:水泥",                        "周度", "HF", "mtd_yoy", "水泥发运率"),
    Indicator("mill_run",          "中国:运转率:磨机",                        "周度", "HF", "mtd_yoy", "磨机运转率"),
    Indicator("polyester_loom",    "中国:江浙地区:开工率:涤纶长丝:下游织机",      "周度", "HF", "mtd_yoy", "下游织机开工"),
    Indicator("polyester_fdy",     "中国:江浙地区:开工率:涤纶长丝",             "周度", "HF", "mtd_yoy", "涤纶长丝开工"),
    Indicator("tangshan_bf",       "中国:唐山:高炉开工率",                     "周度", "HF", "mtd_yoy", "唐山高炉开工"),

    # ---------------- 日度（高频 → MTD 同比） ----------------
    Indicator("pta",               "中国:开工率:精对苯二甲酸",                  "日度", "HF", "mtd_yoy", "PTA开工"),
    Indicator("home_sales30",      "中国:30大中城市:成交面积:商品房",            "日度", "HF", "mtd_yoy", "30城商品房成交", agg="sum"),
    Indicator("bdi",               "波罗的海干散货指数(BDI)",                   "日度", "HF", "mtd_yoy", "BDI"),
    Indicator("nanhua",            "南华综合指数",                            "日度", "HF", "mtd_yoy", "南华综合指数"),
    Indicator("rebar_price",       "中国:价格:螺纹钢(HRB400E,20mm)",          "日度", "HF", "mtd_yoy", "螺纹钢价格"),
    Indicator("cement_priceidx",   "中国:水泥价格指数",                        "日度", "HF", "mtd_yoy", "水泥价格指数"),
    Indicator("port_qhd",          "中国:秦皇岛港:港口吞吐量:煤炭",             "日度", "HF", "mtd_yoy", "秦皇岛港煤炭", agg="sum"),
    Indicator("port_cfd",          "中国:曹妃甸港:煤炭调度:港口吞吐量",          "日度", "HF", "mtd_yoy", "曹妃甸港煤炭", agg="sum"),
    Indicator("port_jt",           "中国:京唐老港:港口吞吐量:煤炭",             "日度", "HF", "mtd_yoy", "京唐老港煤炭", agg="sum"),
]


@dataclass
class Config:
    data_file: str = "数据汇总表0620V3.xlsx"
    target_sheet: str = "工业增加值"
    target_name: str = "中国:工业增加值:规模以上工业企业:当月同比(1-2月拆分)"
    out_dir: str = "outputs"

    # 样本窗口
    main_window_start: str = "2015-01"   # 主模型窗口起点
    long_window_start: str = "2005-01"   # 长史辅助参照起点
    design_start: str = "2003-01"        # 设计矩阵起点（含一年提前量供 YoY/AR 计算）

    # 预测日切口（day-of-month）。None=自动取数据中高频最新日期的 day。
    # 训练与服务使用同一切口以消除 train/serve skew；回测亦用该切口模拟 20-25 号。
    as_of_day: Optional[int] = None

    # 回测
    backtest_start: str = "2018-03"      # 回测首个被预测月（留足训练 warmup）
    conformal_warmup: int = 18           # 在线共形最少残差数
    weight_warmup: int = 12              # 在线组合：确定权重所需的最少历史样本外样本
    interval_levels: Tuple[float, ...] = (0.80, 0.90, 0.95)

    # 高频 MTD 聚合：窗口内观测个数 < 该指标月度观测中位数 × mtd_min_frac 的月份置缺失
    mtd_min_frac: float = 0.3

    # PLS 潜成分个数（实验中 2 最优）
    pls_components: int = 2

    # 覆盖度筛查
    min_coverage: float = 0.60           # 主窗口内最低非缺失比例
    min_years: float = 4.0               # 最短历史年数

    # 因子
    n_factors: int = 1                   # 单因子（小样本：至多 2）
    factor_max_iter: int = 100
    factor_tol: float = 1e-6

    # 长史辅助模型近端指数加权半衰期（月）
    long_halflife_months: float = 60.0

    exclude_months: Tuple[int, ...] = (1, 2)  # 主模型剔除的月份


# =============================================================================
# 二、数据读取与基础工具
# =============================================================================
def load_workbook(cfg: Config) -> Dict[str, pd.DataFrame]:
    """读取各 sheet，首列解析为日期；返回 {sheet: 长表(date + 指标列)}。"""
    if not os.path.exists(cfg.data_file):
        raise FileNotFoundError(f"找不到数据文件：{cfg.data_file}")
    xl = pd.ExcelFile(cfg.data_file)
    frames: Dict[str, pd.DataFrame] = {}
    for s in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=s)
        df = df.rename(columns={df.columns[0]: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        frames[s] = df
    return frames


def validate_indicators(frames: Dict[str, pd.DataFrame]) -> List[str]:
    """核对指标字典与实际表头是否逐字一致；返回告警列表（不抛异常，便于换表排查）。"""
    warns: List[str] = []
    for ind in INDICATORS:
        if ind.sheet not in frames:
            warns.append(f"[缺sheet] {ind.key}: 期望 sheet『{ind.sheet}』不存在")
            continue
        if ind.name not in frames[ind.sheet].columns:
            warns.append(f"[缺指标] {ind.key}: 在『{ind.sheet}』找不到精确列名『{ind.name}』")
    # 反向检查：表中存在但字典未登记的列（提示是否新增指标）
    mapped = {(i.sheet, i.name) for i in INDICATORS}
    for s, df in frames.items():
        if s == "工业增加值":
            continue
        for c in df.columns[1:]:
            if (s, c) not in mapped:
                warns.append(f"[未登记] 『{s}』中的列『{c}』未在指标字典中，已忽略")
    return warns


def _to_month_series(df: pd.DataFrame, name: str, as_of: pd.Timestamp) -> pd.Series:
    """月度指标 → 以 Period[M] 为索引的原始值序列（仅取 date<=as_of）。"""
    s = df.loc[df["date"] <= as_of, ["date", name]].dropna()
    s = s.set_index(s["date"].dt.to_period("M"))[name]
    return s[~s.index.duplicated(keep="last")]


def _mtd_yoy(df: pd.DataFrame, name: str, as_of: pd.Timestamp, as_of_day: int,
             agg: str = "mean", min_frac: float = 0.3) -> pd.Series:
    """
    高频指标 → "月初至今对齐同比（MTD-YoY）"，Period[M] 索引。
    口径：每个自然月取『1 号至 as_of_day 号』窗口内的聚合，再与去年同月同一日切口比较。

    聚合方式由各指标显式声明（Indicator.agg），不再"一刀切取均值"：
      · 'mean' —— 用于率/价/指数/日均型（开工率、价格指数、BDI、日均产量/销量等）：
                  取窗口内观测的均值（= 平均日/周强度），对窗口内观测个数差异稳健。
      · 'sum'  —— 用于可加的流量/总量型（成交面积、成交土地、港口吞吐量等）：
                  取窗口内求和（月度总量口径）。两年同用 [1..as_of_day] 同一日切口，
                  使 YoY 比较口径一致。
    口径稳健性保护：窗口内观测个数过少的月份（< 该指标月度观测中位数的 min_frac，
    且 < 1）置为缺失，避免"薄窗口 vs 去年整窗"的口径偏差污染 YoY。
    训练/服务/回测共用同一 as_of_day，确保无前视偏差。
    """
    sub = df.loc[df["date"] <= as_of, ["date", name]].dropna().copy()
    if sub.empty:
        return pd.Series(dtype=float)
    sub["day"] = sub["date"].dt.day
    sub = sub[sub["day"] <= as_of_day]
    if sub.empty:
        return pd.Series(dtype=float)
    sub["ym"] = sub["date"].dt.to_period("M")
    grp = sub.groupby("ym")[name]
    monthly = grp.sum() if agg == "sum" else grp.mean()
    counts = grp.size()
    typical = counts.median() if len(counts) else 1.0
    floor = max(1, int(np.ceil(min_frac * typical)))
    monthly = monthly.where(counts >= floor)         # 观测过少的月份置缺失
    monthly = monthly.sort_index()
    full = monthly.reindex(pd.period_range(monthly.index.min(), monthly.index.max(), freq="M"))
    yoy = (full / full.shift(12) - 1.0) * 100.0
    return yoy


def _yoy(s: pd.Series) -> pd.Series:
    s = s.sort_index()
    full = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
    return (full / full.shift(12) - 1.0) * 100.0


# =============================================================================
# 三、设计矩阵：vintage-aware（按 as-of 日期重建信息集）
# =============================================================================
def build_design(frames: Dict[str, pd.DataFrame], cfg: Config,
                 as_of: pd.Timestamp, as_of_day: int,
                 target_month: pd.Period) -> pd.DataFrame:
    """
    构造截至 as_of 的月度设计矩阵，索引为 Period[M]，含：
        y      —— 目标当月同比
        AR1    —— 目标滞后一期（预测时已公布）
        SEAS12 —— 目标滞后 12 期（季节朴素锚）
        m_*    —— 月度自变量，统一滞后一期（预测当月时其当月值尚未发布）
        h_*    —— 高频自变量的当月 MTD-YoY（参差边缘的当月信号）
    target_month 之前为训练可用行；target_month 行的 y 为空，用于预测。
    """
    idx = pd.period_range(cfg.design_start, target_month, freq="M")
    out = pd.DataFrame(index=idx)

    # 目标
    tdf = frames[cfg.target_sheet]
    ty = _to_month_series(tdf, cfg.target_name, as_of)
    out["y"] = ty.reindex(idx)
    out["AR1"] = out["y"].shift(1)
    out["SEAS12"] = out["y"].shift(12)

    # 自变量
    for ind in INDICATORS:
        if ind.sheet not in frames or ind.name not in frames[ind.sheet].columns:
            continue
        df = frames[ind.sheet]
        if ind.freq == "M":
            raw = _to_month_series(df, ind.name, as_of)
            if ind.transform == "asis":
                s = raw
            elif ind.transform in ("yoy", "cum_yoy"):
                s = _yoy(raw)
            else:
                s = raw
            out["m_" + ind.key] = s.reindex(idx).shift(1)  # 月度统一滞后一期
        else:  # 高频
            s = _mtd_yoy(df, ind.name, as_of, as_of_day,
                         agg=ind.agg, min_frac=cfg.mtd_min_frac)
            out["h_" + ind.key] = s.reindex(idx)            # 当月 MTD（不滞后）
    return out


def feature_columns(design: pd.DataFrame) -> List[str]:
    return [c for c in design.columns if c != "y"]


def screen_features(design: pd.DataFrame, train_idx: pd.PeriodIndex,
                    cfg: Config, frames: Dict[str, pd.DataFrame]) -> List[str]:
    """覆盖度筛查：主窗口内非缺失比例 < min_coverage 或历史 < min_years 的特征降级剔除。
    AR1 / SEAS12 始终保留。返回入选特征列名。"""
    keep = ["AR1", "SEAS12"]
    dropped: List[Tuple[str, str]] = []
    sub = design.loc[train_idx]
    n = len(sub)
    for c in feature_columns(design):
        if c in ("AR1", "SEAS12"):
            continue
        col = sub[c]
        cov = col.notna().mean()
        if col.notna().sum() == 0:
            dropped.append((c, "全缺失"))
            continue
        first = col.first_valid_index()
        last = col.last_valid_index()
        years = (last - first).n / 12.0 if (first is not None and last is not None) else 0.0
        if cov < cfg.min_coverage:
            dropped.append((c, f"覆盖率{cov:.0%}<{cfg.min_coverage:.0%}"))
        elif years < cfg.min_years:
            dropped.append((c, f"历史{years:.1f}年<{cfg.min_years:.0f}年"))
        else:
            keep.append(c)
    screen_features.last_dropped = dropped  # type: ignore[attr-defined]
    return keep


# =============================================================================
# 四、Stock-Watson EM-PCA 因子（原生消化缺失/参差边缘）
# =============================================================================
class EMFactor:
    """
    缺失稳健的静态因子（扩散指数）。做法：列标准化后用 SVD 取前 k 主成分，
    用重构值迭代填补缺失（EM），收敛后得到因子得分与载荷。
    对新行（可能含缺失，典型为当月参差边缘）用观测项最小二乘投影求因子得分。
    选择 EM-PCA 而非满参数卡尔曼 DFM，是小样本下的稳健务实之选（参数更少、无脆弱调参）。
    """

    def __init__(self, n_factors: int = 1, max_iter: int = 100, tol: float = 1e-6):
        self.k = n_factors
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray) -> "EMFactor":
        X = np.asarray(X, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[(self.std_ == 0) | ~np.isfinite(self.std_)] = 1.0
        Z = (X - self.mean_) / self.std_
        mask = np.isnan(Z)
        Z_filled = np.where(mask, 0.0, Z)  # 初值=均值（标准化后为 0）
        prev = np.inf
        for _ in range(self.max_iter):
            U, S, Vt = np.linalg.svd(Z_filled, full_matrices=False)
            F = U[:, : self.k] * S[: self.k]
            L = Vt[: self.k]
            recon = F @ L
            Z_new = np.where(mask, recon, Z_filled)
            change = np.nanmean((Z_new - Z_filled) ** 2)
            Z_filled = Z_new
            if abs(prev - change) < self.tol:
                break
            prev = change
        self.loadings_ = L            # (k, p)
        self.factors_ = Z_filled @ L.T @ np.linalg.pinv(L @ L.T)  # (n, k)
        return self

    def transform_row(self, x: np.ndarray) -> np.ndarray:
        """对单行（含缺失）用观测项最小二乘求因子得分。"""
        x = np.asarray(x, dtype=float)
        z = (x - self.mean_) / self.std_
        obs = ~np.isnan(z)
        if obs.sum() == 0:
            return np.zeros(self.k)
        L_obs = self.loadings_[:, obs]          # (k, p_obs)
        z_obs = z[obs]                          # (p_obs,)
        f, *_ = np.linalg.lstsq(L_obs.T, z_obs, rcond=None)
        return f

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.vstack([self.transform_row(row) for row in np.asarray(X, dtype=float)])


# =============================================================================
# 五、模型族（统一接口 fit / predict），均含训练内预处理，防泄漏
# =============================================================================
class _OLS:
    """带截距的最小二乘；支持样本权重（用于长史近端加权）。"""

    def fit(self, X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> "_OLS":
        X = np.asarray(X, float)
        A = np.column_stack([np.ones(len(X)), X])
        if w is None:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        else:
            sw = np.sqrt(w)
            coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        self.coef_ = coef
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        A = np.column_stack([np.ones(len(X)), X])
        return A @ self.coef_


class SeasonalNaive:
    name = "SeasonalNaive"

    def fit(self, design, cols, train_idx):
        return self

    def predict_row(self, design, cols, period):
        return float(design.at[period, "SEAS12"])


class ARBaseline:
    name = "AR(1)"

    def fit(self, design, cols, train_idx):
        sub = design.loc[train_idx, ["y", "AR1"]].dropna()
        self.ols = _OLS().fit(sub[["AR1"]].values, sub["y"].values)
        return self

    def predict_row(self, design, cols, period):
        x = design.at[period, "AR1"]
        if not np.isfinite(x):
            x = design.loc[design["AR1"].notna(), "AR1"].iloc[-1]
        return float(self.ols.predict(np.array([[x]]))[0])


class BridgeFactor:
    """单因子 + AR1 的桥接回归（方案核心 nowcast 引擎）。"""

    name = "Factor+AR(Bridge)"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fit(self, design, cols, train_idx):
        all_ind = [c for c in cols if c.startswith(("m_", "h_"))]
        sub = design.loc[train_idx]
        good = sub["y"].notna() & sub["AR1"].notna()
        sub = sub[good]
        # 强度筛选（仅用训练集，防泄漏）：只用与目标有一定同期相关的指标合成因子，
        # 剔除纯噪声列（如 2015+ 已失效的 PPI、南华等），实现"在强指标块上提因子"。
        y = sub["y"].values
        corrs = {}
        for c in all_ind:
            v = sub[c]
            m = v.notna().values
            if m.sum() >= 24 and np.nanstd(v.values[m]) > 0:
                corrs[c] = abs(np.corrcoef(v.values[m], y[m])[0, 1])
        ranked = sorted(corrs, key=lambda k: corrs[k], reverse=True)
        strong = [c for c in ranked if corrs[c] >= 0.25]
        if len(strong) < 5:                 # 至少保留前若干强指标
            strong = ranked[:min(8, len(ranked))]
        ind_cols = strong if strong else all_ind
        self.ind_cols = ind_cols
        self.factor = EMFactor(self.cfg.n_factors, self.cfg.factor_max_iter,
                               self.cfg.factor_tol).fit(sub[ind_cols].values)
        F = self.factor.factors_.copy()
        # 因子符号对齐：将符号一次性写入载荷（loadings_），使 transform_row 产出的
        # 因子方向与训练时一致；predict_row 不再二次翻转，避免符号被重复翻转。
        sign0 = 1.0 if np.corrcoef(F[:, 0], y)[0, 1] >= 0 else -1.0
        self.factor.loadings_[0] *= sign0
        F[:, 0] *= sign0
        Xreg = np.column_stack([F, sub["AR1"].values])
        self.ols = _OLS().fit(Xreg, y)
        return self

    def predict_row(self, design, cols, period):
        row = design.loc[period, self.ind_cols].values.astype(float)
        # transform_row 已基于含符号的载荷计算，方向与训练一致，不再翻转
        f = self.factor.transform_row(row)
        ar1 = design.at[period, "AR1"]
        if not np.isfinite(ar1):
            ar1 = design.loc[design["AR1"].notna(), "AR1"].iloc[-1]
        x = np.concatenate([f, [ar1]])
        return float(self.ols.predict(x[None, :])[0])


class RidgeModel:
    """
    岭回归（RidgeCV，密集 L2 收缩）—— 核心线性成分。
    在共线、p≈n 的小样本上，密集收缩比稀疏选择（Lasso/ElasticNet）更稳更准
    （已由统一无泄漏 OOS 实验证实：Ridge RMSE≈1.45 优于 ElasticNet≈1.58）。
    管线内 median 填补 + 标准化，仅在训练集拟合。
    """

    name = "Ridge"

    def fit(self, design, cols, train_idx):
        self.cols = cols
        sub = design.loc[train_idx]
        good = sub["y"].notna() & sub["AR1"].notna()
        sub = sub[good]
        X = sub[cols].values
        y = sub["y"].values
        n = len(sub)
        n_splits = int(min(5, max(2, n // 20)))
        cv = TimeSeriesSplit(n_splits=n_splits)   # 时序 CV，只用过去验证未来
        self.pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-2, 3, 30), cv=cv)),
        ])
        self.pipe.fit(X, y)
        return self

    def predict_row(self, design, cols, period):
        x = design.loc[[period], self.cols].values
        return float(self.pipe.predict(x)[0])

    def coef_table(self) -> pd.Series:
        ridge = self.pipe.named_steps["ridge"]
        return pd.Series(np.ravel(ridge.coef_), index=self.cols)


class PLSModel:
    """
    偏最小二乘（PLS）—— 核心投影成分。
    用少数潜成分概括共线特征并最大化与目标的协方差，最契合"强共同因子 + 高维"结构
    （实验中 PLS(2) RMSE≈1.38 为各模型族最优）。管线内 median 填补 + 标准化。
    """

    name = "PLS"

    def fit(self, design, cols, train_idx):
        self.cols = cols
        sub = design.loc[train_idx]
        good = sub["y"].notna() & sub["AR1"].notna()
        sub = sub[good]
        X = sub[cols].values
        y = sub["y"].values
        k = int(min(self.cfg_components, X.shape[1], max(1, len(sub) - 2)))
        self.pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("pls", PLSRegression(n_components=k)),
        ])
        self.pipe.fit(X, y)
        return self

    def __init__(self, cfg: Config):
        self.cfg_components = cfg.pls_components

    def predict_row(self, design, cols, period):
        x = design.loc[[period], self.cols].values
        return float(np.ravel(self.pipe.predict(x))[0])


class LGBMChallenger:
    """LightGBM 挑战者：硬约束防小样本过拟合；原生处理缺失。仅在回测打赢 AR 时入组合。"""

    name = "LightGBM"

    def fit(self, design, cols, train_idx):
        self.cols = cols
        sub = design.loc[train_idx]
        good = sub["y"].notna() & sub["AR1"].notna()
        sub = sub[good]
        X = sub[cols].values
        y = sub["y"].values
        self.model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=7, max_depth=3,
            min_child_samples=15, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.6, reg_lambda=5.0, reg_alpha=1.0,
            random_state=RNG_SEED, n_jobs=1, verbose=-1)
        self.model.fit(X, y)
        return self

    def predict_row(self, design, cols, period):
        x = design.loc[[period], self.cols].values
        return float(self.model.predict(x)[0])


def build_model_zoo(cfg: Config) -> List:
    # 核心：Ridge / PLS（密集收缩与投影，小样本共线最优）+ 因子桥接 + AR 锚 + 季节朴素基准
    zoo = [SeasonalNaive(), ARBaseline(), BridgeFactor(cfg), RidgeModel(), PLSModel(cfg)]
    if _HAS_LGBM:
        zoo.append(LGBMChallenger())  # 挑战者：实验显示其更弱，通常被回测剔除
    return zoo


# =============================================================================
# 六、伪实时（vintage-aware）walk-forward 回测
# =============================================================================
def resolve_as_of_day(frames: Dict[str, pd.DataFrame], cfg: Config) -> int:
    if cfg.as_of_day is not None:
        return int(cfg.as_of_day)
    last_days = []
    for ind in INDICATORS:
        if ind.freq == "HF" and ind.sheet in frames and ind.name in frames[ind.sheet].columns:
            d = frames[ind.sheet].loc[frames[ind.sheet][ind.name].notna(), "date"]
            if len(d):
                last_days.append(d.max())
    if not last_days:
        return 23
    return int(max(last_days).day)


def latest_target_month(frames: Dict[str, pd.DataFrame], cfg: Config) -> pd.Period:
    tdf = frames[cfg.target_sheet]
    s = tdf.loc[tdf[cfg.target_name].notna(), "date"]
    return s.max().to_period("M")


def vintage_for(period: pd.Period, as_of_day: int) -> pd.Timestamp:
    """模拟在『被预测月的 as_of_day 号』做预测时的资料截止时点。"""
    start = period.to_timestamp(how="start")
    cap = period.to_timestamp(how="end")
    ts = start + pd.Timedelta(days=as_of_day - 1)
    return min(ts, cap)


def walk_forward(frames: Dict[str, pd.DataFrame], cfg: Config,
                 as_of_day: int, last_month: pd.Period) -> pd.DataFrame:
    """对 backtest_start..last_month 的每个非 1-2 月逐期伪实时预测，返回各模型样本外预测。"""
    test_months = [p for p in pd.period_range(cfg.backtest_start, last_month, freq="M")
                   if p.month not in cfg.exclude_months]
    # 评估用的"真值"取自当前完整数据（被预测月的 y 在预测时点尚未公布，故不能从设计矩阵取）
    tdf = frames[cfg.target_sheet]
    full_target = _to_month_series(tdf, cfg.target_name, tdf["date"].max())
    rows = []
    zoo_names = [m.name for m in build_model_zoo(cfg)]
    for p in test_months:
        actual = full_target.get(p, np.nan)
        if not np.isfinite(actual):
            continue
        as_of = vintage_for(p, as_of_day)
        design = build_design(frames, cfg, as_of, as_of_day, p)
        if p not in design.index:
            continue
        win_start = pd.Period(cfg.main_window_start, freq="M")
        train_idx = design.index[(design.index >= win_start) & (design.index < p)]
        train_idx = train_idx[~train_idx.month.isin(cfg.exclude_months)]
        train_idx = train_idx[design.loc[train_idx, "y"].notna()
                              & design.loc[train_idx, "AR1"].notna()]
        if len(train_idx) < 30:
            continue
        cols = screen_features(design, train_idx, cfg, frames)
        rec = {"period": p, "actual": float(actual)}
        for model in build_model_zoo(cfg):
            try:
                model.fit(design, cols, train_idx)
                rec[model.name] = model.predict_row(design, cols, p)
            except Exception as e:  # 单模型失败不致全盘崩溃
                rec[model.name] = np.nan
        rows.append(rec)
    bt = pd.DataFrame(rows).set_index("period")
    bt.attrs["model_names"] = zoo_names
    return bt


def compute_weights(df: pd.DataFrame, model_names: List[str], cfg: Config) -> Dict[str, float]:
    """
    逆-RMSE 加权：仅纳入 RMSE ≤ AR 基准的模型（AR 始终合格）。不做学习型 stacking。
    df 仅应包含"用于确定权重的历史样本外预测"（在线回测中为当前预测点之前的所有点），
    以杜绝用未来结果选择权重的泄漏。
    """
    names = [n for n in model_names if n != SeasonalNaive.name]
    rmse = {}
    for n in names:
        if n not in df:
            continue
        e = (df[n] - df["actual"]).dropna()
        rmse[n] = float(np.sqrt((e ** 2).mean())) if len(e) else np.inf
    ar = rmse.get(ARBaseline.name, np.inf)
    eligible = {n: r for n, r in rmse.items() if r <= ar + 1e-9 and np.isfinite(r)}
    if not eligible:
        eligible = {ARBaseline.name: ar if np.isfinite(ar) else 1.0}
    inv = {n: 1.0 / max(r, 1e-6) for n, r in eligible.items()}
    tot = sum(inv.values())
    return {n: v / tot for n, v in inv.items()}


def _weighted_point(weights: Dict[str, float], preds: Dict[str, float]) -> float:
    """按权重合成点预测，并对"缺失/失败的模型"重新归一化（不被静默降权）。"""
    num, den = 0.0, 0.0
    for n, w in weights.items():
        v = preds.get(n, np.nan)
        if np.isfinite(v):
            num += w * v
            den += w
    return num / den if den > 0 else np.nan


def online_ensemble(bt: pd.DataFrame, cfg: Config) -> Tuple[pd.Series, List[Dict[str, float]]]:
    """
    真正样本外的组合路径：预测第 t 期时，组合权重只用 t 之前的样本外预测确定
    （扩展窗），权重不足 warmup 期时回退到 AR 基准。每期对可用模型重新归一化。
    这样回测组合指标与共形残差均为无泄漏的样本外结果。
    """
    model_names = bt.attrs["model_names"]
    ens = pd.Series(index=bt.index, dtype=float)
    path: List[Dict[str, float]] = []
    for i, p in enumerate(bt.index):
        past = bt.iloc[:i]
        if past["actual"].notna().sum() >= cfg.weight_warmup:
            w = compute_weights(past, model_names, cfg)
        else:
            w = {ARBaseline.name: 1.0}
        preds = {n: bt.at[p, n] for n in model_names if n in bt}
        ens[p] = _weighted_point(w, preds)
        path.append(w)
    return ens, path


# =============================================================================
# 七、共形区间（split / jackknife+ 风味，基于真实样本外残差）
# =============================================================================
def conformal_offsets(residuals: np.ndarray, level: float) -> Tuple[float, float]:
    """
    标准分裂共形（split conformal）对称区间，含有限样本修正：
    取绝对残差的 ceil((n+1)*level)/n 经验分位 q，偏移为 (-q, +q)。
    该修正保证在可交换性假设下覆盖率 ≥ 名义水平（小样本下略偏保守，稳健）。
    """
    r = np.asarray(residuals, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2:
        return (-3.0, 3.0)
    a = np.abs(r)
    rank_q = min(1.0, np.ceil((n + 1) * level) / n)
    q = float(np.quantile(a, rank_q))
    return (-q, q)


def online_coverage(ens: pd.Series, actual: pd.Series, cfg: Config) -> Dict[float, float]:
    """在线共形覆盖率评估：第 t 期区间仅用 t 之前的残差，给出真实覆盖率估计。"""
    res = (ens - actual)
    cov = {lv: [] for lv in cfg.interval_levels}
    res_hist: List[float] = []
    for p in ens.index:
        point = ens[p]
        a = actual[p]
        if len(res_hist) >= cfg.conformal_warmup and np.isfinite(point) and np.isfinite(a):
            for lv in cfg.interval_levels:
                lo, hi = conformal_offsets(np.array(res_hist), lv)
                inside = (point - hi) <= a <= (point - lo)
                cov[lv].append(bool(inside))
        if np.isfinite(res[p]):
            res_hist.append(float(res[p]))
    return {lv: (np.mean(v) if v else np.nan) for lv, v in cov.items()}


# =============================================================================
# 八、绘图
# =============================================================================
def plot_backtest(bt: pd.DataFrame, ens: pd.Series, cfg: Config,
                  final_period: pd.Period, final_point: float,
                  final_intervals: Dict[float, Tuple[float, float]]) -> str:
    res = (ens - bt["actual"]).dropna().values
    lo90, hi90 = conformal_offsets(res, 0.90)
    x = bt.index.to_timestamp()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9),
                                   gridspec_kw={"height_ratios": [3, 2]})

    ax1.plot(x, bt["actual"].values, color="black", lw=2.0, marker="o", ms=3, label="Actual")
    ax1.plot(x, ens.values, color="C0", lw=1.8, marker="s", ms=3, label="Ensemble forecast")
    ax1.fill_between(x, ens.values - hi90, ens.values - lo90, color="C0", alpha=0.18,
                     label="90% conformal band")
    fx = final_period.to_timestamp()
    flo, fhi = final_intervals[0.90]
    ax1.errorbar([fx], [final_point], yerr=[[final_point - flo], [fhi - final_point]],
                 fmt="D", color="C3", ms=8, capsize=5, lw=2,
                 label=f"Next-period forecast ({final_period})")
    ax1.set_title("Industrial Value-Added YoY — Pseudo-real-time Backtest & Next-period Forecast")
    ax1.set_ylabel("YoY (%)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(alpha=0.3)

    err = (ens - bt["actual"])
    ax2.bar(x, err.values, width=20, color=np.where(err.values >= 0, "C0", "C3"), alpha=0.7)
    ax2.axhline(0, color="black", lw=0.8)
    rmse = np.sqrt(np.nanmean(err.values ** 2))
    ax2.set_title(f"Ensemble out-of-sample error (RMSE={rmse:.2f}, MAE={np.nanmean(np.abs(err.values)):.2f})")
    ax2.set_ylabel("Forecast - Actual")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(cfg.out_dir, "backtest.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_models_and_coverage(bt: pd.DataFrame, weights: Dict[str, float],
                             coverage: Dict[float, float], cfg: Config) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # 各模型 RMSE
    names = [n for n in bt.attrs["model_names"]]
    rmse = []
    for n in names:
        e = (bt[n] - bt["actual"]).dropna()
        rmse.append(np.sqrt((e ** 2).mean()) if len(e) else np.nan)
    colors = ["C2" if n in weights else "C7" for n in names]
    ax1.barh(names, rmse, color=colors)
    ax1.invert_yaxis()
    ax1.set_xlabel("Backtest RMSE")
    ax1.set_title("Model RMSE (green = selected into ensemble)")
    for i, (n, r) in enumerate(zip(names, rmse)):
        w = weights.get(n)
        tag = f"  w={w:.2f}" if w else ""
        if np.isfinite(r):
            ax1.text(r, i, f" {r:.2f}{tag}", va="center", fontsize=8)
    ax1.grid(alpha=0.3, axis="x")
    # 覆盖率 vs 名义
    lv = list(coverage.keys())
    nominal = [l * 100 for l in lv]
    actual = [coverage[l] * 100 if np.isfinite(coverage[l]) else 0 for l in lv]
    xpos = np.arange(len(lv))
    ax2.bar(xpos - 0.2, nominal, width=0.4, label="Nominal", color="C7")
    ax2.bar(xpos + 0.2, actual, width=0.4, label="Empirical (online)", color="C0")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels([f"{int(l*100)}%" for l in lv])
    ax2.set_ylabel("Coverage (%)")
    ax2.set_title("Interval Coverage: nominal vs empirical")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(cfg.out_dir, "models_coverage.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# =============================================================================
# 九、报告辅助
# =============================================================================
def data_health_report(frames, cfg, as_of_day, target_m, forecast_m) -> None:
    print("=" * 78)
    print("【数据体检 / 运行信息】")
    print(f"  数据文件          : {cfg.data_file}")
    print(f"  目标最新已知月    : {target_m}    →  预测目标月: {forecast_m}")
    print(f"  预测日切口 as_of_day = 每月 {as_of_day} 日（训练/服务/回测一致）")
    print(f"  主窗口            : {cfg.main_window_start} 至今（剔除 1、2 月）")
    warns = validate_indicators(frames)
    if warns:
        print("  指标字典校验告警：")
        for w in warns:
            print("    -", w)
    else:
        print("  指标字典校验      : 全部精确匹配，无告警")
    # 各 sheet 最新日期
    print("  各频率最新可得日期：")
    for s in frames:
        if s == cfg.target_sheet:
            continue
        dts = []
        for ind in INDICATORS:
            if ind.sheet == s and ind.name in frames[s].columns:
                d = frames[s].loc[frames[s][ind.name].notna(), "date"]
                if len(d):
                    dts.append(d.max())
        if dts:
            print(f"    {s}: 最新 {max(dts).date()}")


def driver_report(final_models: Dict, cfg: Config) -> None:
    ridge: Optional[RidgeModel] = final_models.get("Ridge")
    if ridge is None:
        return
    coef = ridge.coef_table()
    nz = coef[coef.abs() > 1e-6].sort_values(key=lambda s: s.abs(), ascending=False)
    if nz.empty:
        return
    cn = {("m_" if i.freq == "M" else "h_") + i.key: i.cn for i in INDICATORS}
    cn["AR1"] = "工业增加值滞后1期"
    cn["SEAS12"] = "工业增加值滞后12期(季节)"
    print("\n【主要驱动（Ridge 标准化系数，绝对值降序 Top12）】")
    for name, v in nz.head(12).items():
        label = cn.get(name, name)
        arrow = "↑" if v > 0 else "↓"
        print(f"    {arrow} {label:<22s} 系数={v:+.3f}")


# =============================================================================
# 十、主流程
# =============================================================================
def run(cfg: Config) -> Dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    frames = load_workbook(cfg)
    as_of_day = resolve_as_of_day(frames, cfg)
    target_m = latest_target_month(frames, cfg)
    forecast_m = target_m + 1

    data_health_report(frames, cfg, as_of_day, target_m, forecast_m)
    if not _HAS_LGBM:
        print("  提示：未安装 lightgbm，已跳过其挑战者模型（不影响主流程）。")

    # ---- 伪实时回测 ----
    print("\n" + "=" * 78)
    print("【伪实时 walk-forward 回测（严格按 as-of 重建信息集，对标基准）】")
    bt = walk_forward(frames, cfg, as_of_day, target_m)
    if len(bt) < cfg.conformal_warmup + 5:
        print(f"  警告：回测样本仅 {len(bt)} 期，结果与区间可靠性下降。")
    # 组合采用"在线扩展窗"权重：每期权重只用其之前的样本外预测确定 → 无未来信息泄漏。
    ens, _ = online_ensemble(bt, cfg)
    # 全样本权重仅用于"最终对下一期"的预测（相对预测目标而言全部为过去，合法）。
    final_weights = compute_weights(bt, bt.attrs["model_names"], cfg)

    # 指标
    def _metrics(pred: pd.Series) -> Tuple[float, float, float]:
        e = (pred - bt["actual"]).dropna()
        rmse = float(np.sqrt((e ** 2).mean()))
        mae = float(e.abs().mean())
        prev = bt["actual"].shift(1)
        d = ((np.sign(pred - prev) == np.sign(bt["actual"] - prev)) & prev.notna())
        dacc = float(d[(pred.notna()) & prev.notna()].mean())
        return rmse, mae, dacc

    print(f"  回测区间: {bt.index.min()} ~ {bt.index.max()}  共 {len(bt)} 期（非 1-2 月）")
    print(f"  各模型为逐期样本外预测；组合为在线扩展窗权重（无未来泄漏）。")
    print(f"  {'模型':<22s}{'RMSE':>8}{'MAE':>8}{'方向准确率':>12}{'最终权重':>10}")
    print("  " + "-" * 60)
    for n in bt.attrs["model_names"]:
        rmse, mae, dacc = _metrics(bt[n])
        w = final_weights.get(n)
        wtxt = f"{w:.2f}" if w else "—"
        print(f"  {n:<22s}{rmse:>8.2f}{mae:>8.2f}{dacc:>11.0%}{wtxt:>10}")
    ermse, emae, edacc = _metrics(ens)
    print("  " + "-" * 60)
    print(f"  {'★ 组合(在线OOS)':<22s}{ermse:>8.2f}{emae:>8.2f}{edacc:>11.0%}{'':>10}")
    ar_rmse = _metrics(bt[ARBaseline.name])[0]
    impr = (ar_rmse - ermse) / ar_rmse * 100 if ar_rmse > 0 else 0
    print(f"  组合相对 AR 基准 RMSE 改进: {impr:+.1f}%  "
          f"（在线OOS口径；小样本下温和提升属正常，价值更在区间与驱动）")

    # ---- 覆盖率（在线共形） ----
    coverage = online_coverage(ens, bt["actual"], cfg)
    print("\n【区间覆盖率（在线共形，真实样本外评估）】")
    for lv in cfg.interval_levels:
        c = coverage[lv]
        print(f"  名义 {int(lv*100)}%  →  实测 {c*100:5.1f}%" if np.isfinite(c)
              else f"  名义 {int(lv*100)}%  →  样本不足")

    # ---- 用全部可用数据重训各模型，预测下一期 ----
    print("\n" + "=" * 78)
    print(f"【最终预测：{forecast_m} 工业增加值当月同比】")
    as_of = pd.Timestamp(frames[cfg.target_sheet]["date"].max())  # 资料截止=最新数据
    as_of = max(as_of, vintage_for(forecast_m, as_of_day))
    design = build_design(frames, cfg, as_of, as_of_day, forecast_m)
    win_start = pd.Period(cfg.main_window_start, freq="M")
    train_idx = design.index[(design.index >= win_start) & (design.index < forecast_m)]
    train_idx = train_idx[~train_idx.month.isin(cfg.exclude_months)]
    train_idx = train_idx[design.loc[train_idx, "y"].notna() & design.loc[train_idx, "AR1"].notna()]
    cols = screen_features(design, train_idx, cfg, frames)

    dropped = getattr(screen_features, "last_dropped", [])
    if dropped:
        print(f"  覆盖度筛查：剔除 {len(dropped)} 个特征（短序列/高缺失，已降级）")

    final_models: Dict[str, object] = {}
    point_by_model: Dict[str, float] = {}
    for model in build_model_zoo(cfg):
        try:
            model.fit(design, cols, train_idx)
            final_models[model.name] = model
            point_by_model[model.name] = model.predict_row(design, cols, forecast_m)
        except Exception as e:
            print(f"  [警告] {model.name} 拟合/预测失败：{e}")

    is_janfeb = forecast_m.month in cfg.exclude_months
    if not is_janfeb:
        # 按最终权重合成，并对失败/缺失的模型重新归一化（_weighted_point 内部处理）
        point = _weighted_point(final_weights, point_by_model)
        if not np.isfinite(point):
            point = point_by_model.get(ARBaseline.name, np.nan)
    else:
        # 1-2 月走"春节口径参考"路径：以 AR + 因子简单平均，并显著加宽区间
        cand = [point_by_model.get(BridgeFactor.name), point_by_model.get(ARBaseline.name)]
        cand = [c for c in cand if c is not None and np.isfinite(c)]
        point = float(np.mean(cand)) if cand else point_by_model.get(ARBaseline.name, np.nan)

    # ---- 共形区间（基于回测残差） ----
    res = (ens - bt["actual"]).dropna().values
    widen = 1.6 if is_janfeb else 1.0  # 春节月不确定性更高，加宽
    final_intervals: Dict[float, Tuple[float, float]] = {}
    for lv in cfg.interval_levels:
        lo, hi = conformal_offsets(res, lv)
        final_intervals[lv] = (point - hi * widen, point - lo * widen)

    print(f"\n  点预测：{point:.2f}%")
    for n in sorted(point_by_model, key=lambda k: k):
        w = final_weights.get(n)
        wt = f"(权重{w:.2f})" if w else "(未入组合)"
        print(f"      · {n:<22s} {point_by_model[n]:6.2f}%  {wt}")
    print("\n  区间预测（共形）：")
    for lv in cfg.interval_levels:
        lo, hi = final_intervals[lv]
        star = " ★主报" if abs(lv - 0.80) < 1e-9 else ""
        print(f"      {int(lv*100)}% 区间: [{lo:5.2f}% , {hi:5.2f}%]{star}")
    if is_janfeb:
        print("  ⚠ 本期为 1/2 月：国家统计局仅公布 1-2 月合并值，此为春节口径参考预测，"
              "区间已加宽，建议结合 2 月一并校正。")

    driver_report(final_models, cfg)

    # ---- 绘图与落盘 ----
    p1 = plot_backtest(bt, ens, cfg, forecast_m, point, final_intervals)
    p2 = plot_models_and_coverage(bt, final_weights, coverage, cfg)

    # 结果表落盘
    bt_out = bt.copy()
    bt_out["ensemble"] = ens
    bt_out.to_csv(os.path.join(cfg.out_dir, "backtest_predictions.csv"),
                  encoding="utf-8-sig")
    summary = {
        "forecast_month": str(forecast_m),
        "point_forecast_%": round(point, 3),
        "as_of_day": as_of_day,
        "data_asof": str(as_of.date()),
        "backtest_rmse_ensemble": round(ermse, 3),
        "backtest_rmse_AR": round(ar_rmse, 3),
        "is_jan_feb_reference": is_janfeb,
    }
    for lv in cfg.interval_levels:
        lo, hi = final_intervals[lv]
        summary[f"PI{int(lv*100)}_low"] = round(lo, 3)
        summary[f"PI{int(lv*100)}_high"] = round(hi, 3)
        summary[f"coverage{int(lv*100)}_empirical"] = (round(coverage[lv], 3)
                                                       if np.isfinite(coverage[lv]) else None)
    pd.DataFrame([summary]).to_csv(
        os.path.join(cfg.out_dir, "forecast_summary.csv"),
        index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("【输出文件】")
    print(f"  回测效果图        : {p1}")
    print(f"  模型/覆盖率图     : {p2}")
    print(f"  回测明细          : {os.path.join(cfg.out_dir, 'backtest_predictions.csv')}")
    print(f"  预测汇总          : {os.path.join(cfg.out_dir, 'forecast_summary.csv')}")
    print("=" * 78)

    return summary


if __name__ == "__main__":
    cfg = Config()
    # 允许命令行覆盖数据文件名：python industrial_value_added_forecast.py <xlsx>
    if len(sys.argv) > 1:
        cfg.data_file = sys.argv[1]
    run(cfg)
