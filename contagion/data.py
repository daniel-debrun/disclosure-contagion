"""Build the firm x session panel: S&P 500 membership, unscheduled 8-K counts, OHLC prices, industry and sector.

A firm is a CIK. Inputs (paths in PATHS) are the EDGAR 8-K item table, point-in-time S&P 500 membership by month,
a ticker -> CIK map, the SEC's current ticker list, and Sharadar SEP prices and TICKERS classifications.
"""
import os

import numpy as np
import pandas as pd
import polars as pl

ROOT = os.environ.get("DC_DATA", "/media/barracuda/External")
PATHS = {
    "events": f"{ROOT}/filings/events_8k.parquet",
    "members": f"{ROOT}/quant_text1/member_months.parquet",
    "cik_map": f"{ROOT}/quant_text1/cik_map.parquet",
    "sec_tickers": os.environ.get("DC_SEC_TICKERS", f"{ROOT}/disclosure_contagion/company_tickers.parquet"),
    "sep": f"{ROOT}/disclosure_contagion/SHARADAR_SEP.csv",
    "tickers": f"{ROOT}/latpy_data/sharadar_tickers.parquet",
}
CACHE = f"{ROOT}/disclosure_contagion/panel.npz"

U = {"1.01", "1.02", "1.03", "1.04", "1.05", "2.01", "2.03", "2.04", "2.05", "2.06",
     "3.01", "3.03", "4.01", "4.02", "5.01", "5.02"}
U_NARROW = U - {"1.01", "2.01", "2.03"}
START, END = "2015-01-01", "2026-07-15"   # look-back before 2016-07; END leaves room for 5-day targets


def _members():
    """(month, cik) rows, plus every ticker a CIK was listed under."""
    mm = pd.read_parquet(PATHS["members"]).drop(columns="cik")
    cmap = pd.read_parquet(PATHS["cik_map"])[["ticker", "cik"]]
    mm = mm.merge(cmap, on="ticker", how="left")
    unmapped = mm["cik"].isna().mean()
    mm = mm.dropna(subset=["cik"]).astype({"cik": "int64"})
    mm["month"] = pd.to_datetime(mm["month"]).dt.to_period("M")
    return mm, unmapped


def _prices(tickers):
    sep = (pl.scan_csv(PATHS["sep"], try_parse_dates=True)
           .select("ticker", "date", "open", "high", "low", "close")
           .filter(pl.col("ticker").is_in(list(tickers)) & (pl.col("date") >= pl.lit(START).str.to_date())
                   & (pl.col("date") <= pl.lit(END).str.to_date()))
           .collect())
    return sep.to_pandas()


def build():
    mm, unmapped = _members()
    sec = pd.read_parquet(PATHS["sec_tickers"])
    sec = sec.assign(cik=sec["cik"].astype("int64")).drop_duplicates("cik")   # first row = the common share
    cur = dict(zip(sec["cik"], sec["ticker"]))
    cands = set(mm["ticker"]) | {cur[c] for c in mm["cik"].unique() if c in cur}
    px = _prices(cands)
    px["month"] = px["date"].dt.to_period("M")
    have = set(zip(px["ticker"], px["month"]))

    # One price ticker per (cik, month): the SEC's current ticker if Sharadar has it that month, else the member ticker.
    rows = []
    for (cik, month), g in mm.groupby(["cik", "month"]):
        pick = None
        for t in [cur.get(cik)] + sorted(g["ticker"]):
            if t is not None and (t, month) in have:
                pick = t
                break
        rows.append((cik, month, pick))
    res = pd.DataFrame(rows, columns=["cik", "month", "ticker"])

    # Sessions: days on which most members have a price.
    member_tk = set(res["ticker"].dropna())
    n_by_day = px[px["ticker"].isin(member_tk)].groupby("date").size()
    sessions = np.array(sorted(n_by_day[n_by_day >= 300].index), dtype="datetime64[D]")
    ciks = np.array(sorted(res["cik"].unique()))
    T, N = len(sessions), len(ciks)
    col = {c: j for j, c in enumerate(ciks)}
    smonth = pd.PeriodIndex(pd.to_datetime(sessions), freq="M")

    # Membership by session; months before the first list use the first list.
    first = res["month"].min()
    member = np.zeros((T, N), bool)
    tk_of = {}
    for (cik, month), tk in zip(zip(res["cik"], res["month"]), res["ticker"]):
        tk_of[(cik, month)] = tk
    by_month = res.groupby("month")["cik"].apply(list).to_dict()
    for t, m in enumerate(smonth):
        for c in by_month.get(max(m, first), []):
            member[t, col[c]] = True

    # Prices into T x N arrays, through the ticker resolved for that CIK and month.
    ohlc = {k: np.full((T, N), np.nan) for k in ("open", "high", "low", "close")}
    last_tk = res.dropna().sort_values("month").groupby("cik")["ticker"].last().to_dict()
    pmap = {}
    for (cik, month), tk in tk_of.items():
        if tk is not None:
            pmap[(tk, month)] = cik
    pos = {d: i for i, d in enumerate(sessions)}
    di =np.array([pos.get(d, -1) for d in px["date"].values.astype("datetime64[D]")])
    m_eff = px["month"].where(px["month"] >= first, first)
    ci = np.array([col.get(pmap.get((t, m)), -1) if pmap.get((t, m)) is not None else -1
                   for t, m in zip(px["ticker"], m_eff)])
    ok = (di >= 0) & (ci >= 0)
    for k in ohlc:
        ohlc[k][di[ok], ci[ok]] = px[k].values[ok]

    # Classification: the Sharadar TICKERS row for the firm's last price ticker, preferring common stock.
    tkr = pd.read_parquet(PATHS["tickers"])
    tkr["common"] = tkr["category"].fillna("").str.contains("Common Stock")
    tkr = tkr.sort_values(["common", "lastpricedate"]).groupby("ticker").last()
    industry = np.array([tkr["industry"].get(last_tk.get(c), None) or "?" for c in ciks], dtype=object)
    sector = np.array([tkr["sector"].get(last_tk.get(c), None) or "?" for c in ciks], dtype=object)

    # Events: original 8-Ks, one row per (accession, cik), mapped to the first session on or after `session`.
    ev = pd.read_parquet(PATHS["events"])
    ev = ev[(ev["form"] == "8-K") & ev["cik"].isin(col)]
    f = ev.groupby(["accession", "cik"]).agg(items=("item", set), session=("session", "first")).reset_index()
    f["u"] = f["items"].apply(lambda s: bool(s & U) and "2.02" not in s)
    f["un"] = f["items"].apply(lambda s: bool(s & U_NARROW) and "2.02" not in s)
    sd = pd.to_datetime(f["session"]).values.astype("datetime64[D]")
    ti = np.searchsorted(sessions, sd)
    keep = (ti < T) & (sd >= sessions[0])
    y = np.zeros((T, N), np.int16)
    yn = np.zeros((T, N), np.int16)
    for flag, arr in (("u", y), ("un", yn)):
        sel = keep & f[flag].values
        np.add.at(arr, (ti[sel], f["cik"].map(col).values[sel]), 1)

    out = dict(sessions=sessions, ciks=ciks, member=member, y=y, y_narrow=yn, industry=industry, sector=sector,
               unmapped=np.array(unmapped), n_8k=np.array(int(keep.sum())), **ohlc)
    np.savez_compressed(CACHE, **out)
    return out


def load():
    if not os.path.exists(CACHE):
        return build()
    z = np.load(CACHE, allow_pickle=True)
    return {k: z[k] for k in z.files}
