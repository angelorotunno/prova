"""Frontiera efficiente di Markowitz su un paniere di ETF.

Scarica i prezzi mensili aggiustati da Alpha Vantage, calcola rendimenti
e covarianze annualizzati, trova il portafoglio a varianza minima e quello
a massimo Sharpe (solo posizioni lunghe) e disegna la frontiera.

Uso:
    export ALPHAVANTAGE_API_KEY=la_tua_chiave
    python efficient_frontier.py --tickers SPY EFA AGG GLD VNQ --start 2010-01 --rf 0.02
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize

API_URL = "https://www.alphavantage.co/query"
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


# --------------------------------------------------------------------------- #
# Dati
# --------------------------------------------------------------------------- #
def fetch_monthly_adjusted(ticker: str, api_key: str) -> pd.Series:
    """Prezzi di chiusura mensili aggiustati (dividendi e split) di un ticker."""
    params = {"function": "TIME_SERIES_MONTHLY_ADJUSTED", "symbol": ticker, "apikey": api_key}
    payload = requests.get(API_URL, params=params, timeout=30).json()
    series = payload.get("Monthly Adjusted Time Series")
    if series is None:
        msg = payload.get("Note") or payload.get("Information") or payload.get("Error Message") or payload
        raise RuntimeError(f"{ticker}: risposta inattesa da Alpha Vantage -> {msg}")
    prices = pd.Series({pd.Timestamp(d): float(v["5. adjusted close"]) for d, v in series.items()}, name=ticker)
    return prices.sort_index()


def load_prices(tickers: list[str], api_key: str, refresh: bool = False) -> pd.DataFrame:
    """Usa la cache in data/prices.csv se contiene già tutti i ticker richiesti."""
    cache = DATA_DIR / "prices.csv"
    if cache.exists() and not refresh:
        cached = pd.read_csv(cache, index_col=0, parse_dates=True)
        if set(tickers).issubset(cached.columns):
            return cached[tickers]

    columns = []
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(13)  # piano gratuito: max 5 richieste al minuto
        print(f"Scarico {ticker}...")
        columns.append(fetch_monthly_adjusted(ticker, api_key))

    prices = pd.concat(columns, axis=1).dropna()
    DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(cache)
    return prices


# --------------------------------------------------------------------------- #
# Statistiche e ottimizzazione
# --------------------------------------------------------------------------- #
def annualized_stats(prices: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Rendimento atteso e matrice di covarianza annualizzati da dati mensili."""
    returns = prices.pct_change().dropna()
    return returns.mean() * 12, returns.cov() * 12


def portfolio_performance(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    return float(w @ mu), float(np.sqrt(w @ cov @ w))


def _optimize(objective, n: int, extra_constraints: list | None = None) -> np.ndarray:
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}] + (extra_constraints or [])
    result = minimize(
        objective,
        x0=np.full(n, 1 / n),
        bounds=[(0, 1)] * n,  # solo posizioni lunghe, niente leva
        constraints=constraints,
        method="SLSQP",
    )
    if not result.success:
        raise RuntimeError(f"Ottimizzazione fallita: {result.message}")
    return result.x


def min_variance(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return _optimize(lambda w: w @ cov @ w, len(mu))


def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float) -> np.ndarray:
    def neg_sharpe(w):
        ret, vol = portfolio_performance(w, mu, cov)
        return -(ret - rf) / vol

    return _optimize(neg_sharpe, len(mu))


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, points: int = 50) -> list[tuple[float, float]]:
    """Per ogni rendimento obiettivo, il portafoglio a volatilità minima."""
    w_min = min_variance(mu, cov)
    targets = np.linspace(portfolio_performance(w_min, mu, cov)[0], mu.max(), points)
    frontier = []
    for target in targets:
        w = _optimize(
            lambda w: w @ cov @ w,
            len(mu),
            [{"type": "eq", "fun": lambda w, t=target: w @ mu - t}],
        )
        ret, vol = portfolio_performance(w, mu, cov)
        frontier.append((vol, ret))
    return frontier


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def plot(mu, cov, tickers, frontier, w_mv, w_ms, rf, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    vols, rets = zip(*frontier)
    ax.plot(vols, rets, color="#1f4e79", lw=2, label="Frontiera efficiente")

    asset_vol = np.sqrt(np.diag(cov))
    ax.scatter(asset_vol, mu, color="#7f7f7f", zorder=3)
    for t, x, y in zip(tickers, asset_vol, mu):
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)

    for w, label, color in [(w_mv, "Varianza minima", "#2e8b57"), (w_ms, "Massimo Sharpe", "#c0392b")]:
        ret, vol = portfolio_performance(w, mu, cov)
        ax.scatter(vol, ret, s=120, marker="*", color=color, zorder=4, label=label)

    # Capital Market Line: dal tasso privo di rischio al portafoglio tangente
    ret_ms, vol_ms = portfolio_performance(w_ms, mu, cov)
    x = np.linspace(0, max(asset_vol) * 1.05, 50)
    ax.plot(x, rf + (ret_ms - rf) / vol_ms * x, ls="--", color="#c0392b", lw=1, label="Capital Market Line")

    ax.set_xlabel("Volatilità annua")
    ax.set_ylabel("Rendimento atteso annuo")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(left=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("Frontiera efficiente di Markowitz (solo posizioni lunghe)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_table(tickers, mu, cov, w_mv, w_ms, rf) -> pd.DataFrame:
    rows = {}
    for name, w in [("Varianza minima", w_mv), ("Massimo Sharpe", w_ms)]:
        ret, vol = portfolio_performance(w, mu, cov)
        rows[name] = {**dict(zip(tickers, np.round(w, 4))), "Rendimento": ret, "Volatilità": vol, "Sharpe": (ret - rf) / vol}
    return pd.DataFrame(rows).T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=["SPY", "EFA", "AGG", "GLD", "VNQ"])
    parser.add_argument("--start", default="2010-01", help="primo mese del campione (AAAA-MM)")
    parser.add_argument("--rf", type=float, default=0.02, help="tasso privo di rischio annuo")
    parser.add_argument("--refresh", action="store_true", help="riscarica i dati ignorando la cache")
    args = parser.parse_args()

    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key and not (DATA_DIR / "prices.csv").exists():
        raise SystemExit("Imposta la variabile ALPHAVANTAGE_API_KEY (chiave gratuita su alphavantage.co).")

    prices = load_prices(args.tickers, api_key or "", args.refresh).loc[args.start:]
    mu_s, cov_df = annualized_stats(prices)
    mu, cov = mu_s.values, cov_df.values

    w_mv, w_ms = min_variance(mu, cov), max_sharpe(mu, cov, args.rf)
    frontier = efficient_frontier(mu, cov)

    RESULTS_DIR.mkdir(exist_ok=True)
    plot(mu, cov, args.tickers, frontier, w_mv, w_ms, args.rf, RESULTS_DIR / "frontier.png")
    table = summary_table(args.tickers, mu, cov, w_mv, w_ms, args.rf)
    table.to_csv(RESULTS_DIR / "weights.csv")

    print(f"\nCampione: {prices.index[0]:%Y-%m} -> {prices.index[-1]:%Y-%m} ({len(prices)} mesi)")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nGrafico salvato in {RESULTS_DIR / 'frontier.png'}")


if __name__ == "__main__":
    main()
