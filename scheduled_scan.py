#!/usr/bin/env python3
"""
每日美股扫描（做多）：COIN NVDA GOOGL AVGO MU APP + SPY 环境
数据：yfinance（日线2年 + 1小时60天），单一来源（Yahoo），无跨源核验。

用法
  python scheduled_scan.py --once          # 立即运行一次
  python scheduled_scan.py --daemon        # 常驻，每个工作日美东 SCAN_RUN_AT(默认17:30) 运行
  python scheduled_scan.py --selftest      # 离线合成数据自测（不联网）
  python scheduled_scan.py --backtest [--years 1]   # 历史回测（联网拉3年日线）
  crontab 方案（服务器时区=UTC，周一至周五 22:30 UTC，覆盖夏/冬令时收盘后）：
    30 22 * * 1-5  cd /path/to/dir && /usr/bin/python3 scheduled_scan.py --once >> scan.log 2>&1

环境变量（均可选）
  SCAN_OUT=reports  SCAN_RUN_AT=17:30  SCAN_DUMP_DIR=data（保存已完成K线CSV）
  TG_TOKEN / TG_CHAT_ID                         Telegram 推送简报
  SMTP_HOST SMTP_PORT SMTP_USER SMTP_PASS MAIL_TO  邮件推送完整报告

研究与技术分析工具，不构成投资建议。
"""
import argparse, io, json, os, smtplib, ssl, sys, time, traceback
import datetime as dt
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TICKERS = ["COIN", "NVDA", "GOOGL", "AVGO", "MU", "APP"]
BENCH = "SPY"
NY, JST = ZoneInfo("America/New_York"), ZoneInfo("Asia/Tokyo")
OUT = Path(os.environ.get("SCAN_OUT", "reports"))
RUN_AT = os.environ.get("SCAN_RUN_AT", "17:30")  # 美东时间
MIN_EXEC_SCORE = 70   # 可执行信号最低分（B级及以上）
WATCH_SCORE = 55      # 观察名单下限（C级）：仅观察，不是买入信号
MIN_RR = 2.0          # 以 TP2 计的预期风险收益比下限
MIN_DAILY, MIN_HOURLY = 210, 60


# ----------------------------------------------------------------- 数据获取
def fetch(tickers, interval, period):
    import yfinance as yf
    raw = None
    for k in range(3):  # 共享IP(如GitHub Actions)偶发限流，简单重试
        raw = yf.download(tickers, period=period, interval=interval, auto_adjust=False,
                          group_by="ticker", progress=False, threads=True)
        if raw is not None and not raw.empty:
            break
        time.sleep(5 * (k + 1))
    out = {}
    for t in tickers:
        try:
            df = raw[t].copy()
        except Exception:
            continue
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        out[t] = df.dropna(subset=["open", "high", "low", "close"])
    return out


def completed_daily(df, now):
    """只保留已完成日线：美东16:10前不使用当天K线。"""
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    after_close = (now.hour, now.minute) >= (16, 10)
    if now.weekday() < 5 and not after_close:
        df = df[df.index.date < now.date()]
    else:
        df = df[df.index.date <= now.date()]
    return df


def completed_hourly(df, now):
    """只保留常规盘且已收盘的1小时K线（9:30起每小时，最后一根15:30-16:00）。"""
    df = df.copy()
    idx = pd.to_datetime(df.index)
    idx = idx.tz_localize("UTC").tz_convert(NY) if idx.tz is None else idx.tz_convert(NY)
    df.index = idx
    tod = idx.hour * 60 + idx.minute
    df = df[(tod >= 570) & (tod < 960)]
    ends = pd.DatetimeIndex([min(t + pd.Timedelta(hours=1), t.normalize() + pd.Timedelta(hours=16))
                             for t in df.index])
    return df[np.asarray(ends <= pd.Timestamp(now))]


def last_weekday(d):
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def validate(df, kind, now, min_rows):
    """返回 (critical, warnings)。critical 非空则该标的不产生信号。"""
    if df is None or df.empty:
        return ["无数据"], []
    crit, warn = [], []
    if len(df) < min_rows:
        crit.append(f"{kind}样本不足 {len(df)}<{min_rows}")
    if not df.index.is_monotonic_increasing:
        crit.append(f"{kind}时间乱序")
    if df.index.has_duplicates:
        warn.append(f"{kind}存在重复时间戳")
    if df[["open", "high", "low", "close"]].isna().any().any():
        crit.append(f"{kind}OHLC含缺失值")
    bad = ((df.high < df.low) | (df.close > df.high * 1.001) | (df.close < df.low * 0.999)
           | (df.open > df.high * 1.001) | (df.open < df.low * 0.999))
    if bad.any():
        crit.append(f"{kind}有{int(bad.sum())}根K线OHLC逻辑异常")
    if (df.volume.fillna(0) <= 0).mean() > 0.02:
        warn.append(f"{kind}零成交量占比偏高({(df.volume.fillna(0) <= 0).mean():.1%})")
    jump = df.close.pct_change().abs()
    if kind == "日线" and (jump > 0.35).any():
        warn.append("日线存在单日>35%波动，请确认拆股/复权或真实事件")
    if kind == "日线":
        after_close = (now.hour, now.minute) >= (16, 10)
        exp = last_weekday(now.date() if after_close else now.date() - dt.timedelta(days=1))
        gap = (exp - df.index[-1].date()).days
        if gap > 5:
            crit.append(f"日线最新K线 {df.index[-1].date()} 严重滞后")
        elif gap >= 1:
            warn.append(f"日线最新K线 {df.index[-1].date()} 早于预期 {exp}（可能休市或数据延迟）")
    return crit, warn


def next_earnings(t):
    try:
        import yfinance as yf
        cal = yf.Ticker(t).calendar
        e = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if not e:
            return None
        d0 = e[0]
        d0 = d0.date() if hasattr(d0, "date") else d0
        return str(d0) if d0 >= dt.date.today() else None
    except Exception:
        return None


# ----------------------------------------------------------------- 指标
def add_ind(d):
    d = d.copy()
    c = d.close
    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    d["sma50"] = c.rolling(50).mean()
    d["sma200"] = c.rolling(200).mean()
    for m in ("ema20", "sma50", "sma200"):
        d[m + "_slope"] = d[m].pct_change(5)
    pc = c.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    d["atr14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    d["rvol20"] = d.volume / d.volume.shift().rolling(20).mean()
    d["up"] = c > pc
    return d


def pos_in_bar(b):
    rng = b.high - b.low
    return (b.close - b.low) / rng if rng > 0 else 0.5


# ----------------------------------------------------------------- Wyckoff 候选识别（规则化近似）
def detect(d, sup, res):
    """仅基于已完成日线；结果均为'候选'，不证明机构行为。"""
    ev = {}
    win = d.iloc[-60:]
    ret = win.close.pct_change()
    for ts, b in win.iterrows():
        if b.rvol20 >= 2 and ret[ts] <= -0.03 and pos_in_bar(b) >= 0.4:
            ev["SC?"] = (ts, f"RVOL20={b.rvol20:.1f}，单日{ret[ts]:.1%}，收于K线上部；单独出现不构成底部确认")
    last5 = d.iloc[-5:]
    for ts, b in last5.iterrows():
        atr = b.atr14
        if b.low < sup and b.close > sup:
            ev["Spring"] = (ts, f"最低{b.low:.2f}刺破支撑{sup:.2f}后收回，收{b.close:.2f}，RVOL20={b.rvol20:.2f}")
    for ts, b in last5.iterrows():
        atr = b.atr14
        if (("Spring" in ev and ts > ev["Spring"][0]) or ("SC?" in ev and "Spring" not in ev and ts > ev["SC?"][0])) \
                and sup - 0.25 * atr <= b.low <= sup + 0.5 * atr and b.rvol20 < 0.8 and b.close > sup:
            ev["Test"] = (ts, f"回测支撑附近(低{b.low:.2f})，缩量RVOL20={b.rvol20:.2f}，收回支撑之上")
    for ts, b in last5.iterrows():
        if b.close > res and b.rvol20 >= 1.5 and pos_in_bar(b) >= 0.6 and b.close > b.ema20:
            ev["SOS"] = (ts, f"收盘{b.close:.2f}突破区间阻力{res:.2f}，RVOL20={b.rvol20:.2f}，收于K线上部")
    if "SOS" in ev:
        for ts, b in last5.iterrows():
            if ts > ev["SOS"][0] and res - b.atr14 <= b.low <= res + 0.5 * b.atr14 \
                    and b.rvol20 < 0.9 and b.close >= res - 0.25 * b.atr14:
                ev["LPS"] = (ts, f"突破后回踩阻力转支撑({res:.2f})，缩量RVOL20={b.rvol20:.2f}，未失守")
    return ev


def phase_of(L, ev, sup=None):
    if sup is not None and L.close <= sup and ("Spring" in ev or "Test" in ev) and "SOS" not in ev:
        return "收盘仍在区间支撑下方（吸筹未确认）"
    if "SOS" in ev or "LPS" in ev:
        return "Phase D（候选）"
    if "Spring" in ev or "Test" in ev:
        return "Phase C（候选）"
    if "SC?" in ev:
        return "Phase A–B（候选，仅SC迹象）"
    if L.close > L.sma200 and L.sma50 > L.sma200 and L.sma50_slope > 0 and L.sma200_slope > 0:
        return "上升趋势（Phase E/Markup）"
    if L.close < L.sma200:
        return "下跌/整理，未见可识别的吸筹事件"
    return "区间/阶段不明"


# ----------------------------------------------------------------- 评分
def score_all(d, ev, rs, spy_pts, rr):
    L = d.iloc[-1]
    # Wyckoff 30（无事件时仅给趋势基础分，最多8）
    pts = {"Spring": 8, "Test": 6, "SOS": 10, "LPS": 6}
    wy = sum(v for k, v in pts.items() if k in ev)
    if wy == 0:
        wy = 8 if (L.close > L.sma200 and L.sma200_slope > 0) else (4 if L.close > L.sma50 else 0)
    wy = min(30, wy + (4 if ("SC?" in ev and wy >= 8) else 0))
    # 成交量 25
    l20 = d.iloc[-20:]
    upv, dnv = l20.volume[l20.up].mean(), l20.volume[~l20.up].mean()
    ratio = (upv / dnv) if (dnv and not np.isnan(dnv) and not np.isnan(upv)) else np.nan
    vol = 0
    if "SOS" in ev: vol += 8
    vol += 8 if ratio > 1.2 else (4 if ratio > 1.0 else 0)
    if "LPS" in ev or "Test" in ev: vol += 5
    if L.up and L.rvol20 >= 1.2: vol += 4
    vol = min(25, vol)
    # 均线 20
    ma = 4 * (L.close > L.ema20) + 4 * (L.ema20 > L.sma50) + 4 * (L.sma50 > L.sma200) + 4 * (L.close > L.sma200)
    ma += round(4 * ((L.ema20_slope > 0) + (L.sma50_slope > 0) + (L.sma200_slope > 0)) / 3)
    # SPY环境 15
    env = spy_pts + (3 if rs["rs20"] > 0 else 0) + (1 if rs["rs5"] > 0 else 0)
    # 风险收益比 10
    has = any(k in ev for k in ("Spring", "Test", "SOS", "LPS"))
    rrp = 0 if (rr is None or rr < 2 or not has) else (7 if rr < 3 else 10)
    if has and rr is not None and 1 <= rr < 2: rrp = 3
    total = int(wy + vol + ma + env + rrp)
    grade = "A" if total >= 85 else "B" if total >= 70 else "C" if total >= 55 else "D"
    return dict(wyckoff=int(wy), volume=int(vol), ma=int(ma), env=int(env), rr=int(rrp), total=total,
                grade=grade, updown_vol_ratio=None if np.isnan(ratio) else round(float(ratio), 2))


# ----------------------------------------------------------------- 交易计划
def make_plan(d, ev, sup, res):
    L, atr = d.iloc[-1], d.iloc[-1].atr14
    entry = float(d.high.iloc[-3:].max() + 0.02 * atr)
    if "Spring" in ev: low = d.low[ev["Spring"][0]]
    elif "Test" in ev: low = d.low[ev["Test"][0]]
    elif "LPS" in ev: low = d.low[ev["LPS"][0]]
    elif "SOS" in ev: low = d.low[ev["SOS"][0]]
    else: low = d.low.iloc[-5:].min()
    stop = float(low - 0.5 * atr)
    risk = entry - stop
    if risk <= 0:
        return None
    cands = [res, d.high.iloc[-60:].max(), d.high.iloc[-120:].max(), d.high.iloc[-252:].max(), L.sma50, L.sma200]
    # TP1 至少 1R；相邻 TP 至少相隔 0.5R；优先取上方结构位，不足再用 R 倍数补齐
    floor = entry + 1.0 * risk
    tp = []
    for x in sorted({round(float(x), 2) for x in cands if x == x}):
        if x >= floor and (not tp or x >= tp[-1] + 0.5 * risk):
            tp.append(x)
        if len(tp) == 3:
            break
    for m in (1.5, 2.5, 4.0):
        if len(tp) == 3:
            break
        lo = floor if not tp else tp[-1] + 0.5 * risk
        tp.append(max(entry + m * risk, lo))
    R = [(x - entry) / risk for x in tp]
    return dict(entry=entry, stop=stop, risk=risk, risk_atr=risk / atr, tp=tp, R=R, rr=R[1],
                wide_stop=risk / atr > 3.5)


def hourly_check(h, entry):
    if h is None or len(h) < 25:
        return None
    h = h.copy()
    h["rv"] = h.volume / h.volume.shift().rolling(20).mean()
    b = h.iloc[-1]
    return dict(time=h.index[-1].strftime("%m-%d %H:%M ET"), close=float(b.close), rvol=float(b.rv),
                triggered=bool(b.close > entry and b.rv >= 1.2))


# ----------------------------------------------------------------- 单票分析
def analyze(t, d, h, spy, spy_pts, xc, warns):
    d = add_ind(d)
    L, P = d.iloc[-1], d.iloc[-2]
    base = d.iloc[-45:-5]
    sup, res = float(base.low.min()), float(base.high.max())
    ev = detect(d, sup, res)
    s = spy.reindex(d.index).ffill()

    def rs(n):
        return float(d.close.iloc[-1] / d.close.iloc[-1 - n] - 1 - (s.close.iloc[-1] / s.close.iloc[-1 - n] - 1))
    rsd = dict(rs1=rs(1), rs5=rs(5), rs20=rs(20))
    confirmed = [k for k in ("Spring", "Test", "SOS", "LPS") if k in ev]
    plan = make_plan(d, ev, sup, res) if confirmed else None   # 无已确认事件不生成触发方案
    rr = plan["rr"] if plan else None
    sc = score_all(d, ev, rsd, spy_pts, rr)
    hc = hourly_check(h, plan["entry"]) if plan else None
    miss = []
    if not confirmed: miss.append("日线无 Spring/Test/SOS/LPS 候选事件，不生成触发方案")
    elif plan is None: miss.append("无有效止损（止损≥入场）")
    elif plan["rr"] < MIN_RR: miss.append(f"预期R:R(TP2)={plan['rr']:.1f}<{MIN_RR:.0f}")
    elif plan["wide_stop"]: miss.append(f"止损过宽({plan['risk_atr']:.1f}ATR)")
    if sc["total"] < MIN_EXEC_SCORE: miss.append(f"评分{sc['total']}<{MIN_EXEC_SCORE}")
    if plan is not None:
        if hc is None: miss.append("1小时数据不足")
        elif not hc["triggered"]: miss.append(f"1h收盘未突破触发价{plan['entry']:.2f}或量能<1.2x")
    execu = not miss
    dist = (plan["entry"] / L.close - 1) if plan else None
    return dict(ticker=t, asof=str(d.index[-1].date()), price=float(L.close), chg1=float(L.close / P.close - 1),
                support=sup, resistance=res, atr14=float(L.atr14), rvol20=float(L.rvol20),
                ema20=float(L.ema20), sma50=float(L.sma50), sma200=float(L.sma200),
                slopes=dict(ema20=float(L.ema20_slope), sma50=float(L.sma50_slope), sma200=float(L.sma200_slope)),
                phase=phase_of(L, ev, sup), events={k: [str(v[0].date()), v[1]] for k, v in ev.items()},
                rs=rsd, score=sc, plan=plan, hourly=hc, missing=miss, executable=execu,
                dist_to_entry=dist, xcheck=xc[1], data_warnings=warns)


def spy_env(spy):
    s = add_ind(spy)
    L, P = s.iloc[-1], s.iloc[-2]
    pts = 3 * (L.close > L.ema20) + 3 * (L.close > L.sma50) + 3 * (L.close > L.sma200) + 2 * (L.ema20_slope > 0)
    label = "偏多" if pts >= 9 else "中性" if pts >= 5 else "偏空"
    return dict(close=float(L.close), asof=str(s.index[-1].date()), chg1=float(L.close / P.close - 1),
                chg5=float(L.close / s.close.iloc[-6] - 1), ema20=float(L.ema20), sma50=float(L.sma50),
                sma200=float(L.sma200), pts=int(pts), label=label,
                above=[n for n, v in (("EMA20", L.ema20), ("SMA50", L.sma50), ("SMA200", L.sma200)) if L.close > v])


# ----------------------------------------------------------------- 报告
def pct(x): return f"{x:+.2%}"


def build_report(env, rows, failures, now, notes):
    jst = now.astimezone(JST).strftime("%Y-%m-%d %H:%M JST")
    L = [f"# 每日美股扫描（做多）— 运行 {now.strftime('%Y-%m-%d %H:%M ET')} / {jst}",
         "数据源：**仅 Yahoo 单一来源**（yfinance），日线+1小时，已剔除未收盘K线；未做跨源核验，关键价位请对照券商行情确认。", ""]
    if notes:
        L += ["**数据说明**"] + [f"- {n}" for n in notes] + [""]
    L += ["## 一、市场总体判断（SPY）"]
    if env:
        L.append(f"SPY 收 {env['close']:.2f}（{env['asof']}，{pct(env['chg1'])}，5日{pct(env['chg5'])}）；"
                 f"EMA20 {env['ema20']:.2f} / SMA50 {env['sma50']:.2f} / SMA200 {env['sma200']:.2f}；"
                 f"位于 {'、'.join(env['above']) or '全部关键均线下方'} 之上。环境评分 {env['pts']}/11 → **{env['label']}**。")
    else:
        L.append("SPY 数据缺失或未通过校验，环境无法判断，所有标的不给可执行信号。")
    L += ["", "## 二、六只股票对比", "",
          "| 标的 | 收盘/日期 | 日涨跌 | 趋势/阶段 | 事件 | RVOL20 | 相对SPY(5d/20d) | 评分/等级 | 状态 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ev = "、".join(k for k in r["events"]) or "无"
        st = "**可执行**" if r["executable"] else ("观察（非买入）" if r["score"]["total"] >= WATCH_SCORE else "无信号")
        L.append(f"| {r['ticker']} | {r['price']:.2f}（{r['asof']}） | {pct(r['chg1'])} | {r['phase']} | {ev} | "
                 f"{r['rvol20']:.2f} | {pct(r['rs']['rs5'])}/{pct(r['rs']['rs20'])} | "
                 f"{r['score']['total']}（{r['score']['grade']}） | {st} |")
    for t, why in failures.items():
        L.append(f"| {t} | — | — | 数据未通过校验 | — | — | — | — | 无信号：{why} |")
    L += ["", "## 三、个股明细"]
    for r in rows:
        sc, pl, hc = r["score"], r["plan"], r["hourly"]
        L += [f"### {r['ticker']}  {r['price']:.2f}（日线 {r['asof']} 收盘）",
              f"- 阶段：{r['phase']}；区间支撑/阻力 {r['support']:.2f}/{r['resistance']:.2f}；ATR14 {r['atr14']:.2f}",
              f"- 均线：EMA20 {r['ema20']:.2f}（斜率{pct(r['slopes']['ema20'])}/5d），SMA50 {r['sma50']:.2f}（{pct(r['slopes']['sma50'])}），"
              f"SMA200 {r['sma200']:.2f}（{pct(r['slopes']['sma200'])}）"]
        L.append("- 事件证据（已收盘日线，候选而非定论）：" + (
            "；".join(f"{k}@{v[0]} {v[1]}" for k, v in r["events"].items()) or "无"))
        L.append(f"- 量能：RVOL20={r['rvol20']:.2f}；近20日上涨/下跌日均量比={sc['updown_vol_ratio']}")
        L.append(f"- 评分：Wyckoff {sc['wyckoff']}/30 + 量能 {sc['volume']}/25 + 均线 {sc['ma']}/20 + SPY环境 {sc['env']}/15 + R:R {sc['rr']}/10 = **{sc['total']}（{sc['grade']}级）**")
        if pl:
            L.append(f"- 条件性方案（需先满足触发）：触发价 {pl['entry']:.2f}；技术止损 {pl['stop']:.2f}（风险{pl['risk_atr']:.1f}ATR）；"
                     f"TP1/TP2/TP3 = {pl['tp'][0]:.2f}/{pl['tp'][1]:.2f}/{pl['tp'][2]:.2f}（{pl['R'][0]:.1f}R/{pl['R'][1]:.1f}R/{pl['R'][2]:.1f}R，预期R:R按TP2计）")
        else:
            L.append("- 触发方案：未生成（无已确认的 Spring/Test/SOS/LPS 候选事件，或止损无效）")
        if hc:
            L.append(f"- 1h最新已收盘K线 {hc['time']}：收 {hc['close']:.2f}，量能 {hc['rvol']:.2f}x → {'已触发' if hc['triggered'] else '未触发'}")
        if r["missing"]:
            L.append("- 缺失条件：" + "；".join(r["missing"]))
        if r.get("earnings"):
            L.append(f"- 下次财报（Yahoo，需自行核对）：{r['earnings']}")
        if r["data_warnings"]:
            L.append("- 数据提示：" + "；".join(r["data_warnings"]))
    ex = [r for r in rows if r["executable"]]
    L += ["", "## 四、当日结论"]
    if ex:
        for r in ex:
            L.append(f"- **{r['ticker']}** 满足全部过滤条件：触发价 {r['plan']['entry']:.2f}，止损 {r['plan']['stop']:.2f}，TP2 R:R {r['plan']['rr']:.1f}")
    else:
        L.append("**今日无合格信号。**")
        near = sorted(rows, key=lambda r: (len(r["missing"]), -r["score"]["total"]))[:2]
        for r in near:
            L.append(f"- 最接近：{r['ticker']}（{r['score']['total']}分）缺：{'；'.join(r['missing'])}")
    watch = [r for r in rows if not r["executable"] and r["score"]["total"] >= WATCH_SCORE]
    if watch:
        L += ["", f"**观察名单（评分≥{WATCH_SCORE}但未达可执行条件，不是买入信号）：**"]
        for r in sorted(watch, key=lambda r: -r["score"]["total"]):
            trig = f"，触发价{r['plan']['entry']:.2f}/止损{r['plan']['stop']:.2f}" if r["plan"] else ""
            L.append(f"- {r['ticker']}：{r['score']['total']}分（{r['score']['grade']}级）{trig}；待满足：{'；'.join(r['missing']) or '无'}")
    risks = []
    if env and env["label"] != "偏多":
        risks.append(f"SPY 环境{env['label']}，做多胜率与R:R均受压")
    risks += [f"{r['ticker']} 财报 {r['earnings']}" for r in rows if r.get("earnings")]
    L += ["", "**主要风险：** " + ("；".join(risks) if risks else "见各股缺失条件"),
          "", "*研究与技术分析，不构成投资建议；Wyckoff事件为规则化候选，不能证明机构真实建仓。*"]
    return "\n".join(L)


def brief_of(full):
    return full[:3500] + ("\n…（完整报告见文件/邮件）" if len(full) > 3500 else "")


def deliver(full, now):
    OUT.mkdir(parents=True, exist_ok=True)
    stem = now.strftime("%Y-%m-%d")
    (OUT / f"{stem}.md").write_text(full, encoding="utf-8")
    (OUT / "latest.md").write_text(full, encoding="utf-8")
    print(full)
    tok, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT_ID")
    if tok and chat:
        try:
            import urllib.parse, urllib.request
            data = urllib.parse.urlencode({"chat_id": chat, "text": brief_of(full)}).encode()
            urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data, timeout=20)
        except Exception as e:
            print("Telegram 发送失败:", e, file=sys.stderr)
    if os.environ.get("SMTP_HOST") and os.environ.get("MAIL_TO"):
        try:
            m = EmailMessage()
            m["Subject"] = f"每日美股扫描 {stem}"
            m["From"] = os.environ.get("SMTP_USER", "scanner@localhost")
            m["To"] = os.environ["MAIL_TO"]
            m.set_content(full)
            with smtplib.SMTP_SSL(os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT") or 465),
                                  context=ssl.create_default_context()) as s:
                if os.environ.get("SMTP_USER"):
                    s.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASS", ""))
                s.send_message(m)
        except Exception as e:
            print("邮件发送失败:", e, file=sys.stderr)


# ----------------------------------------------------------------- 主流程
def run(now=None, fetcher=fetch, earnings=True):
    now = now or dt.datetime.now(NY)
    names = TICKERS + [BENCH]
    notes, failures, rows = [], {}, []
    try:
        raw_d = fetcher(names, "1d", "2y")
        raw_h = fetcher(names, "1h", "60d")
    except Exception as e:
        full = build_report(None, [], {t: f"数据下载失败 {type(e).__name__}" for t in TICKERS}, now,
                            [f"行情下载失败：{e}"])
        deliver(full, now)
        return full
    D, H = {}, {}
    for t in names:
        d = completed_daily(raw_d[t], now) if t in raw_d else None
        h = completed_hourly(raw_h[t], now) if t in raw_h else None
        dump = os.environ.get("SCAN_DUMP_DIR")
        if dump:  # 保存已完成K线，便于在对话里直接读取
            Path(dump).mkdir(parents=True, exist_ok=True)
            if d is not None:
                d.to_csv(Path(dump) / f"{t}_1d.csv", index_label="date")
            if h is not None:
                h.to_csv(Path(dump) / f"{t}_1h.csv", index_label="datetime_et")
        crit, warn = validate(d, "日线", now, MIN_DAILY)
        hcrit, hwarn = validate(h, "1小时", now, MIN_HOURLY) if h is not None else (["无1小时数据"], [])
        D[t] = (d, crit, warn)
        H[t] = (h if not hcrit else None, hwarn + [f"1小时数据未用：{'; '.join(hcrit)}"] * bool(hcrit))
    env = None
    sd, scrit, swarn = D[BENCH]
    if not scrit:
        env = spy_env(sd)
    notes += [f"SPY：{w}" for w in swarn] + [f"SPY：{c}" for c in scrit]
    for t in TICKERS:
        d, crit, warn = D[t]
        if crit or env is None:
            failures[t] = "；".join(crit) if crit else "SPY环境缺失"
            continue
        xc = ("na", "仅Yahoo单一来源")
        try:
            r = analyze(t, d, H[t][0], sd, env["pts"], xc, warn + H[t][1])
            r["earnings"] = next_earnings(t) if earnings else None
            rows.append(r)
        except Exception as e:
            failures[t] = f"分析异常 {type(e).__name__}: {e}"
    full = build_report(env, rows, failures, now, notes)
    deliver(full, now)
    (OUT / f"{now.strftime('%Y-%m-%d')}.json").write_text(
        json.dumps(dict(env=env, rows=rows, failures=failures), ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    return full


def next_run(now):
    hh, mm = map(int, RUN_AT.split(":"))
    t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if t <= now: t += dt.timedelta(days=1)
    while t.weekday() >= 5: t += dt.timedelta(days=1)
    return t


def daemon():
    while True:
        now = dt.datetime.now(NY)
        t = next_run(now)
        print(f"下次运行：{t:%Y-%m-%d %H:%M ET}（{t.astimezone(JST):%m-%d %H:%M JST}）", flush=True)
        time.sleep(max(1, (t - now).total_seconds()))
        try:
            run()
        except Exception:
            tb = traceback.format_exc()
            deliver(f"# 扫描运行失败\n```\n{tb}\n```\n今日无信号（运行失败）。", dt.datetime.now(NY))


# ----------------------------------------------------------------- 历史回测（日线，同一套规则）
def backtest(daily, spy, years=1.0, fill_window=5, max_hold=40, slip=0.0005):
    """逐日滚动：只用 i 日及之前的数据生成信号，信号次日起挂单(有效fill_window根)。
    1小时数据只有约60天，无法回测一年，因此用'后续日线最高价触及触发价'代替1h收盘触发（偏乐观）。
    同一根K线内止损与止盈无法判先后，一律按先止损（保守）；入场当根K线不判断止盈。
    出场：1/3仓位各在TP1/TP2/TP3，止损不移动；持仓超过max_hold根按收盘平仓；双边滑点slip。"""
    sp = add_ind(spy)
    sp_pts = (3 * (sp.close > sp.ema20) + 3 * (sp.close > sp.sma50) + 3 * (sp.close > sp.sma200)
              + 2 * (sp.ema20_slope > 0)).astype(float)
    trades, stat, sigs = [], {}, []
    for t, d0 in daily.items():
        d = add_ind(d0)
        n = len(d)
        pts = sp_pts.reindex(d.index).ffill().fillna(0).values
        sclose = spy.close.reindex(d.index).ffill().values
        cl = d.close.values
        start = max(MIN_DAILY, n - int(252 * years) - 1)
        busy, nsig, nfill = -1, 0, 0
        for i in range(start, n - 1):
            if i <= busy:
                continue
            ds = d.iloc[:i + 1]
            base = ds.iloc[-45:-5]
            sup, res = float(base.low.min()), float(base.high.max())
            ev = detect(ds, sup, res)
            confirmed = [k for k in ("Spring", "Test", "SOS", "LPS") if k in ev]
            if not confirmed:
                continue
            plan = make_plan(ds, ev, sup, res)
            if not plan or plan["rr"] < MIN_RR or plan["wide_stop"]:
                continue
            rsd = dict(rs5=cl[i] / cl[i - 5] - sclose[i] / sclose[i - 5],
                       rs20=cl[i] / cl[i - 20] - sclose[i] / sclose[i - 20])
            sc = score_all(ds, ev, rsd, int(pts[i]), plan["rr"])
            if sc["total"] < 55:
                continue
            nsig += 1
            rec = dict(ticker=t, date=str(d.index[i].date()), score=sc["total"],
                       events="+".join(confirmed), filled=False)
            sigs.append(rec)
            entry, stop, tps = plan["entry"], plan["stop"], plan["tp"]
            fill = None
            for j in range(i + 1, min(n, i + 1 + fill_window)):
                b = d.iloc[j]
                if b.high >= entry:
                    fill = (j, max(b.open, entry) * (1 + slip))
                    break
                if b.low <= stop:
                    break
            if not fill:
                continue
            nfill += 1
            rec["filled"] = True
            j0, fpx = fill
            risk = fpx - stop
            if risk <= 0:
                continue
            rem, pnl, hit, exit_j, tag = 1.0, 0.0, 0, n - 1, "数据末尾平仓"
            for k in range(j0, min(n, j0 + max_hold + 1)):
                b = d.iloc[k]
                if b.low <= stop:
                    px = min(b.open, stop) * (1 - slip) if k > j0 else stop * (1 - slip)
                    pnl += rem * (px - fpx) / risk
                    rem, exit_j, tag = 0.0, k, "止损"
                    break
                if k > j0:
                    while hit < 3 and b.high >= tps[hit]:
                        px = max(tps[hit], b.open) * (1 - slip)
                        part = rem if hit == 2 else min(rem, 1 / 3)
                        pnl += part * (px - fpx) / risk
                        rem -= part
                        hit += 1
                    if rem <= 1e-9:
                        exit_j, tag = k, "TP3全部止盈"
                        break
                if k == j0 + max_hold or k == n - 1:
                    px = b.close * (1 - slip)
                    pnl += rem * (px - fpx) / risk
                    rem, exit_j = 0.0, k
                    tag = "超时平仓" if k == j0 + max_hold else "数据末尾平仓"
                    break
            busy = exit_j
            trades.append(dict(ticker=t, signal=str(d.index[i].date()), fill=str(d.index[j0].date()),
                               exit=str(d.index[exit_j].date()), score=sc["total"], grade=sc["grade"],
                               events="+".join(confirmed), entry=round(fpx, 2), stop=round(stop, 2),
                               tp_hit=hit, bars=exit_j - j0, R=round(pnl, 3), how=tag))
        stat[t] = dict(signals=nsig, fills=nfill,
                       bh=float(cl[-1] / cl[max(0, n - int(252 * years) - 1)] - 1))
    return trades, stat, sigs


def bt_stats(tr):
    if not tr:
        return None
    r = np.array([x["R"] for x in sorted(tr, key=lambda x: x["exit"])])
    w, l = r[r > 0], r[r <= 0]
    eq = np.cumsum(r)
    return dict(n=len(r), win=float((r > 0).mean()), avgR=float(r.mean()), totR=float(r.sum()),
                pf=float(w.sum() / abs(l.sum())) if len(l) and l.sum() != 0 else float("inf"),
                dd=float((np.maximum.accumulate(eq) - eq).max()), hold=float(np.mean([x["bars"] for x in tr])))


def bt_report(trades, stat, years, now, sigs=None):
    f = lambda s, k, fmt: "—" if s is None else format(s[k], fmt)
    L = [f"# 历史回测（过去约{years:g}年，日线）— 生成于 {now.strftime('%Y-%m-%d %H:%M ET')}",
         "规则与每日扫描一致：日线出现 Spring/Test/SOS/LPS 候选 + 止损有效 + TP2预期R:R≥2 + 止损≤3.5ATR；评分用同一套权重。",
         "入场：信号次日起5根K线内最高价触及触发价则成交（跳空按开盘价）；1/3仓位分别在TP1/TP2/TP3止盈，止损不移动；持仓最长40根；单边滑点0.05%。", "",
         "## 汇总（R = 每笔初始风险的倍数）", "",
         "| 评分门槛 | 笔数 | 胜率 | 平均R | 总R | 盈亏比PF | 最大回撤(R) | 平均持仓(根) |", "|---|---|---|---|---|---|---|---|"]
    for th in (70, 55):
        sub = [x for x in trades if x["score"] >= th]
        s = bt_stats(sub)
        L.append(f"| ≥{th} | {len(sub)} | {f(s,'win','.0%')} | {f(s,'avgR','+.2f')} | {f(s,'totR','+.1f')} | "
                 f"{f(s,'pf','.2f')} | {f(s,'dd','.1f')} | {f(s,'hold','.1f')} |")
    L += ["", "## 分标的（评分≥55）", "", "| 标的 | 信号数 | 成交数 | 笔数 | 胜率 | 平均R | 总R | 同期买入持有 |", "|---|---|---|---|---|---|---|---|"]
    for t, st in stat.items():
        sub = [x for x in trades if x["ticker"] == t]
        s = bt_stats(sub)
        L.append(f"| {t} | {st['signals']} | {st['fills']} | {len(sub)} | {f(s,'win','.0%')} | {f(s,'avgR','+.2f')} | "
                 f"{f(s,'totR','+.1f')} | {st['bh']:+.0%} |")
    ev = {}
    for x in trades:
        ev.setdefault(x["events"], []).append(x["R"])
    if ev:
        L += ["", "## 按事件组合（评分≥55）", "", "| 事件 | 笔数 | 平均R |", "|---|---|---|"]
        L += [f"| {k} | {len(v)} | {np.mean(v):+.2f} |" for k, v in sorted(ev.items(), key=lambda kv: -len(kv[1]))]
    if sigs:
        L += ["", "## 信号分布（含未成交的信号；评分≥55，已满足事件/止损/R:R过滤）", "",
              "| 评分区间 | 信号数 | 成交数 | 成交率 |", "|---|---|---|---|"]
        for lo, hi, nm in ((55, 70, "55–69 (C)"), (70, 85, "70–84 (B)"), (85, 101, "≥85 (A)")):
            g = [x for x in sigs if lo <= x["score"] < hi]
            fl = sum(x["filled"] for x in g)
            L.append(f"| {nm} | {len(g)} | {fl} | {(fl / len(g)):.0%} |" if g else f"| {nm} | 0 | 0 | — |")
        L += ["", "| 事件组合 | 信号数 | 平均评分 |", "|---|---|---|"]
        evs = {}
        for x in sigs:
            evs.setdefault(x["events"], []).append(x["score"])
        L += [f"| {k} | {len(v)} | {np.mean(v):.0f} |" for k, v in sorted(evs.items(), key=lambda kv: -len(kv[1]))]
        L.append(f"\n评分最高的信号：{max(x['score'] for x in sigs)} 分。当前每日扫描的可执行门槛为 {MIN_EXEC_SCORE} 分。")
    elif sigs is not None:
        L += ["", "## 信号分布", "回测期内没有任何信号同时满足事件、止损/R:R过滤且评分≥55。"]
    L += ["", "## 解读与局限（请务必阅读）",
          "- **样本小**：六只股票、一年、规则很严，笔数通常很少，胜率和平均R的统计误差很大，不能据此认定规则有效或无效。",
          "- **幸存者/选股偏差**：这六只是今天选定的标的，回测期间的表现带有事后选择的影响。",
          "- **1h触发被日线替代**：实盘要求1小时K线收盘确认，回测只要求日内最高价触及，成交更容易，结果偏乐观。",
          "- **未计入**：真实滑点与冲击成本、财报跳空对止损的影响、复权数据差异；Yahoo日线成交量可能不是全市场合并量。",
          "- **参数未优化**：阈值（RVOL、ATR倍数、评分权重）均为初始设定，不要在这一年数据上反复调参再拿同一段验证。",
          "- 研究工具，不构成投资建议。"]
    return "\n".join(L)


def run_backtest(years=1.0, fetcher=fetch, now=None):
    now = now or dt.datetime.now(NY)
    raw = fetcher(TICKERS + [BENCH], "1d", "3y")
    D = {}
    for t, df in raw.items():
        d = completed_daily(df, now)
        crit, _ = validate(d, "日线", now, MIN_DAILY + int(252 * years))
        if crit:
            print(f"{t} 跳过：{'; '.join(crit)}", file=sys.stderr)
        else:
            D[t] = d
    if BENCH not in D:
        raise RuntimeError("SPY 数据缺失或未通过校验，无法回测")
    spy = D.pop(BENCH)
    trades, stat, sigs = backtest(D, spy, years)
    rep = bt_report(trades, stat, years, now, sigs)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = now.strftime("%Y-%m-%d")
    (OUT / f"backtest_{stem}.md").write_text(rep, encoding="utf-8")
    pd.DataFrame(sigs).to_csv(OUT / f"backtest_{stem}_signals.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(trades).to_csv(OUT / f"backtest_{stem}_trades.csv", index=False, encoding="utf-8-sig")
    print(rep)
    return rep


# ----------------------------------------------------------------- 离线自测
def _synthetic(n, seed, spring=False, start="2025-01-02"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    if spring:  # 构造区间 + Spring
        c[-60:] = 100 + np.sin(np.linspace(0, 8, 60)) * 3
        c[-3] = 94.0; c[-2] = 98.0; c[-1] = 99.5
    o = np.r_[c[0], c[:-1]]
    h, l = np.maximum(o, c) * 1.007, np.minimum(o, c) * 0.993
    v = rng.integers(8_000_000, 12_000_000, n).astype(float)
    df = pd.DataFrame(dict(open=o, high=h, low=l, close=c, volume=v), index=idx)
    if spring:
        df.loc[idx[-3], ["low", "volume"]] = [92.0, 6_000_000.0]
    return df


def selftest():
    now = dt.datetime(2026, 11, 3, 18, 0, tzinfo=NY)
    end = pd.Timestamp("2026-11-02")
    nd = 300
    start = (end - pd.tseries.offsets.BDay(nd - 1)).strftime("%Y-%m-%d")

    def fake(names, interval, period):
        out = {}
        for i, t in enumerate(names):
            d = _synthetic(nd, i, spring=(t == "NVDA"), start=start)
            if interval == "1h":
                days = d.index[-30:]
                rows = []
                for day in days:
                    for k in range(7):
                        ts = pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=k)
                        px = d.close[day] * (1 + 0.001 * k)
                        rows.append((ts, px, px * 1.002, px * 0.998, px, 1e6))
                x = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).set_index("ts")
                x.index = x.index.tz_localize(NY)
                out[t] = x
            else:
                out[t] = d
        return out
    global OUT
    OUT = Path("selftest_reports")
    rep = run(now=now, fetcher=fake, earnings=False)
    assert "每日美股扫描" in rep and "NVDA" in rep
    print("\nSELFTEST OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true")
    g.add_argument("--daemon", action="store_true")
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--backtest", action="store_true")
    ap.add_argument("--years", type=float, default=1.0)
    ap.add_argument("--min-score", type=int, default=None, help="覆盖可执行信号评分门槛（默认70）")
    a = ap.parse_args()
    if a.min_score is not None:
        MIN_EXEC_SCORE = a.min_score
    if a.selftest: selftest()
    elif a.backtest: run_backtest(a.years)
    elif a.daemon: daemon()
    else: run()
