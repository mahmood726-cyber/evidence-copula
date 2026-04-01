"""
EvidenceCopula — copula-based dependence modeling of trust and statistical significance
in Cochrane meta-analyses.

Marginals: U = empirical_CDF(final_score), V = empirical_CDF(|z_statistic|)
Copulas: Clayton (lower-tail), Frank (symmetric), Gumbel (upper-tail)
Model selection: AIC
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats


# ---------------------------------------------------------------------------
# Empirical CDF (pseudo-observations)
# ---------------------------------------------------------------------------

def empirical_cdf(x: np.ndarray) -> np.ndarray:
    """
    Map 1-D array to pseudo-observations in (0, 1) via ranks/(n+1).

    Returns array of the same length as x with values strictly in (0, 1).
    Ties are broken by average rank.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)
    ranks = stats.rankdata(x, method="average")
    return ranks / (n + 1)


# ---------------------------------------------------------------------------
# z-statistic from p-value
# ---------------------------------------------------------------------------

def z_from_p(p: float) -> float:
    """Convert two-sided p-value to absolute z-statistic."""
    p = float(p)
    # Clamp to avoid infinities
    p = max(1e-15, min(1.0 - 1e-15, p))
    return abs(stats.norm.ppf(p / 2.0))


def z_array_from_p(p_values: np.ndarray) -> np.ndarray:
    """Vectorised z_from_p."""
    p_values = np.asarray(p_values, dtype=float)
    p_clamped = np.clip(p_values, 1e-15, 1.0 - 1e-15)
    return np.abs(stats.norm.ppf(p_clamped / 2.0))


# ---------------------------------------------------------------------------
# Copula density / log-likelihood helpers
# ---------------------------------------------------------------------------

def _log_sum_exp(a: float, b: float) -> float:
    """Stable log(exp(a) + exp(b))."""
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


# ---------------------------------------------------------------------------
# Clayton copula
# ---------------------------------------------------------------------------

def clayton_logpdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """
    Log-density of Clayton copula.

    c(u,v;theta) = (theta+1)*(u*v)^(-theta-1) * (u^-theta + v^-theta - 1)^(-2-1/theta)
    Valid for theta > 0.
    """
    if theta <= 0:
        return np.full(len(u), -np.inf)
    lu = np.log(u)
    lv = np.log(v)
    inner = np.exp(-theta * lu) + np.exp(-theta * lv) - 1.0
    # Guard against non-positive inner
    mask = inner <= 0
    inner = np.where(mask, 1e-300, inner)
    log_density = (
        math.log(theta + 1)
        + (-theta - 1) * (lu + lv)
        + (-2.0 - 1.0 / theta) * np.log(inner)
    )
    log_density = np.where(mask, -1e15, log_density)
    return log_density


def fit_clayton(u: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """
    Fit Clayton copula via MLE. Returns (theta_hat, log_likelihood).

    theta > 0 (lower-tail dependence).
    Independence limit: theta -> 0.
    """
    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)

    def neg_loglik(log_theta: float) -> float:
        th = math.exp(log_theta)
        ll = np.sum(clayton_logpdf(u, v, th))
        return -ll if np.isfinite(ll) else 1e15

    # Scan over a range of starting points
    best_val = np.inf
    best_log_theta = 0.0
    for start in np.linspace(-3, 3, 13):
        try:
            res = optimize.minimize_scalar(neg_loglik, bounds=(start - 1, start + 1),
                                           method="bounded")
            if res.fun < best_val:
                best_val = res.fun
                best_log_theta = res.x
        except Exception:
            pass

    theta_hat = math.exp(best_log_theta)
    loglik = -best_val
    return theta_hat, loglik


def clayton_lower_tail(theta: float) -> float:
    """Lower tail dependence coefficient: lambda_L = 2^(-1/theta)."""
    if theta <= 0:
        return 0.0
    return 2.0 ** (-1.0 / theta)


# ---------------------------------------------------------------------------
# Frank copula
# ---------------------------------------------------------------------------

def frank_logpdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """
    Log-density of Frank copula.

    c(u,v;theta) = -theta*(e^{-theta}-1)*e^{-theta(u+v)} /
                   [(e^{-theta}-1) + (e^{-theta*u}-1)*(e^{-theta*v}-1)]^2

    Handles theta -> 0 (independence) with a fallback.

    Note: For theta > 0, A = exp(-theta)-1 < 0 and eu, ev < 0 so eu*ev > 0.
    The denominator denom = A + eu*ev can be negative — that is valid since
    c(u,v) = num / denom^2 and denom^2 is always positive. We use |denom|.
    """
    if abs(theta) < 1e-8:
        return np.zeros(len(u))

    A = math.exp(-theta) - 1.0
    # Numerator log-magnitude: -theta * A > 0 for theta > 0 (since A < 0)
    num_mag = abs(-theta * A)
    if num_mag <= 0:
        return np.full(len(u), -1e15)
    log_num = math.log(num_mag) + (-theta) * (u + v)

    # Denominator: can be negative for theta > 0 but we square it
    eu = np.exp(-theta * u) - 1.0
    ev = np.exp(-theta * v) - 1.0
    denom = A + eu * ev
    # Use |denom|; if |denom| is ~0, density explodes (cop boundary)
    abs_denom = np.abs(denom)
    too_small = abs_denom < 1e-300
    abs_denom_safe = np.where(too_small, 1e-300, abs_denom)
    log_density = log_num - 2.0 * np.log(abs_denom_safe)
    log_density = np.where(too_small, -1e15, log_density)
    return log_density


def fit_frank(u: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """
    Fit Frank copula via MLE. Returns (theta_hat, log_likelihood).

    theta can be any real number != 0.
    """
    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)

    def neg_loglik(theta: float) -> float:
        if abs(theta) < 1e-8:
            return 0.0  # Independence: loglik = 0
        ll = np.sum(frank_logpdf(u, v, theta))
        return -ll if np.isfinite(ll) else 1e15

    best_val = np.inf
    best_theta = 1.0
    for start in np.linspace(-8, 8, 17):
        try:
            lo = start - 1.0 if start > 0 else start - 1.0
            hi = start + 1.0
            if lo * hi < 0:
                lo = 0.01 if start > 0 else -0.01
            if abs(lo) < 1e-8:
                lo = 1e-6
            if abs(hi) < 1e-8:
                hi = 1e-6
            res = optimize.minimize_scalar(neg_loglik,
                                           bounds=(start - 0.8, start + 0.8),
                                           method="bounded")
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x
        except Exception:
            pass

    loglik = -best_val
    return best_theta, loglik


# Frank has no tail dependence
FRANK_TAIL_DEPENDENCE = 0.0


# ---------------------------------------------------------------------------
# Gumbel copula
# ---------------------------------------------------------------------------

def gumbel_logpdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """
    Log-density of Gumbel copula.

    C(u,v;theta) = exp(-((-ln u)^theta + (-ln v)^theta)^(1/theta))
    c(u,v) = C * (ln u * ln v)^(theta-1) * ... (full expression derived below)

    Valid for theta >= 1.
    """
    if theta < 1.0:
        theta = 1.0
    lu = -np.log(np.clip(u, 1e-12, 1 - 1e-12))
    lv = -np.log(np.clip(v, 1e-12, 1 - 1e-12))

    # inner = lu^theta + lv^theta
    inner = lu ** theta + lv ** theta
    log_inner = np.log(np.where(inner <= 0, 1e-300, inner))

    # C(u,v) = exp(-inner^(1/theta))
    log_C = -inner ** (1.0 / theta)

    # Full log-density:
    # log c = log_C + (1/theta - 2)*log_inner + (theta-1)*(log(lu)+log(lv))
    #         + log(inner^(1/theta - 1) + (theta-1)*inner^(1/theta-2))
    # Using the known formula:
    # c(u,v) = C(u,v) / (u*v) * inner^(1/theta-2) * (lu*lv)^(theta-1)
    #          * (inner^(1/theta) + theta - 1)
    log_lu = np.log(np.where(lu <= 0, 1e-300, lu))
    log_lv = np.log(np.where(lv <= 0, 1e-300, lv))

    log_density = (
        log_C
        + np.log(u) + np.log(v)    # division by u*v as negation since we use log(1/u)=log(lv) no...
        # Correct: divide by u*v means -log(u) - log(v) = +log(lu) + log(lv)  NO...
        # Let's be precise: log c = log C - log u - log v + (1/theta-2)*log_inner
        #                            + (theta-1)*(log lu + log lv) + log(inner^(1/theta) + theta-1)
    )
    # Recompute carefully
    term1 = log_C  # = -inner^(1/theta)
    term2 = np.log(np.clip(u, 1e-15, None)) + np.log(np.clip(v, 1e-15, None))
    # -log u = log lu, -log v = log lv  since lu = -log u => log u = -lu
    # So -log u - log v should be added but we want to divide... let's use:
    # log(1/u) = log lu? No: lu = -log(u) so log(u) = -lu
    # Thus -log(u) - log(v) = lu + lv (the raw values, not log)
    # We need log(lu) + log(lv) for the (theta-1) term
    inner_power = inner ** (1.0 / theta)
    log_factor = np.log(np.where(inner_power + theta - 1 <= 0, 1e-300,
                                  inner_power + theta - 1))

    log_density = (
        -inner_power                            # log C(u,v)
        + lu + lv                               # -log(u) - log(v)  = lu + lv
        + (1.0 / theta - 2.0) * log_inner      # inner^(1/theta - 2)
        + (theta - 1.0) * (log_lu + log_lv)    # (lu*lv)^(theta-1)
        + log_factor                            # (inner^(1/theta) + theta - 1)
    )
    return log_density


def fit_gumbel(u: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """
    Fit Gumbel copula via MLE. Returns (theta_hat, log_likelihood).

    theta >= 1 (theta=1 is independence).
    """
    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)

    def neg_loglik(log_theta_m1: float) -> float:
        """Parameterize as log(theta-1) to enforce theta >= 1."""
        th = 1.0 + math.exp(log_theta_m1)
        ll = np.sum(gumbel_logpdf(u, v, th))
        return -ll if np.isfinite(ll) else 1e15

    best_val = np.inf
    best_param = 0.0
    for start in np.linspace(-3, 3, 13):
        try:
            res = optimize.minimize_scalar(neg_loglik, bounds=(start - 1, start + 1),
                                           method="bounded")
            if res.fun < best_val:
                best_val = res.fun
                best_param = res.x
        except Exception:
            pass

    theta_hat = 1.0 + math.exp(best_param)
    loglik = -best_val
    return theta_hat, loglik


def gumbel_upper_tail(theta: float) -> float:
    """Upper tail dependence coefficient: lambda_U = 2 - 2^(1/theta)."""
    if theta < 1.0:
        return 0.0
    return 2.0 - 2.0 ** (1.0 / theta)


# ---------------------------------------------------------------------------
# AIC / model selection
# ---------------------------------------------------------------------------

def aic(loglik: float, k: int) -> float:
    """AIC = 2k - 2*logL."""
    return 2.0 * k - 2.0 * loglik


@dataclass
class CopulaResult:
    """Result for a single fitted copula."""
    family: str          # 'Clayton', 'Frank', 'Gumbel'
    theta: float
    loglik: float
    aic_val: float
    tail_lower: float    # lambda_L
    tail_upper: float    # lambda_U
    n: int


@dataclass
class DependenceResult:
    """Full dependence analysis for one group."""
    group: str
    n: int
    kendall_tau: float
    kendall_p: float
    spearman_rho: float
    spearman_p: float
    copulas: list[CopulaResult] = field(default_factory=list)

    @property
    def best_copula(self) -> Optional[CopulaResult]:
        if not self.copulas:
            return None
        return min(self.copulas, key=lambda c: c.aic_val)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class EvidenceCopulaEngine:
    """
    Fits Clayton, Frank, and Gumbel copulas to (trust_score, |z_statistic|)
    pseudo-observations, overall and per domain.
    """

    def __init__(self,
                 scores_path: str,
                 verdicts_path: str,
                 groups_path: str):
        self.scores_path = scores_path
        self.verdicts_path = verdicts_path
        self.groups_path = groups_path
        self.data: Optional[pd.DataFrame] = None
        self.results: dict[str, DependenceResult] = {}

    def load(self) -> pd.DataFrame:
        """Load and merge data, derive z-statistics, drop NaN p_values."""
        scores = pd.read_csv(self.scores_path)
        verdicts = pd.read_csv(self.verdicts_path)
        groups = pd.read_csv(self.groups_path)

        df = scores.merge(verdicts, on="ma_id", how="inner")
        df = df.merge(groups, on="ma_id", how="left")
        df = df.dropna(subset=["p_value"])
        df["z_stat"] = z_array_from_p(df["p_value"].values)
        self.data = df.reset_index(drop=True)
        return self.data

    def _fit_group(self, sub: pd.DataFrame, group_name: str) -> DependenceResult:
        """Fit all copulas + baselines for one subset."""
        n = len(sub)
        scores = sub["final_score"].values
        z_vals = sub["z_stat"].values

        # Pseudo-observations
        u = empirical_cdf(scores)
        v = empirical_cdf(z_vals)

        # Baseline correlations
        tau, tau_p = stats.kendalltau(scores, z_vals)
        rho, rho_p = stats.spearmanr(scores, z_vals)

        copulas: list[CopulaResult] = []

        # Clayton
        try:
            th_c, ll_c = fit_clayton(u, v)
            a_c = aic(ll_c, k=1)
            copulas.append(CopulaResult(
                family="Clayton", theta=th_c, loglik=ll_c, aic_val=a_c,
                tail_lower=clayton_lower_tail(th_c), tail_upper=0.0, n=n
            ))
        except Exception as exc:
            warnings.warn(f"Clayton fit failed for {group_name}: {exc}")

        # Frank
        try:
            th_f, ll_f = fit_frank(u, v)
            a_f = aic(ll_f, k=1)
            copulas.append(CopulaResult(
                family="Frank", theta=th_f, loglik=ll_f, aic_val=a_f,
                tail_lower=FRANK_TAIL_DEPENDENCE, tail_upper=FRANK_TAIL_DEPENDENCE, n=n
            ))
        except Exception as exc:
            warnings.warn(f"Frank fit failed for {group_name}: {exc}")

        # Gumbel
        try:
            th_g, ll_g = fit_gumbel(u, v)
            a_g = aic(ll_g, k=1)
            copulas.append(CopulaResult(
                family="Gumbel", theta=th_g, loglik=ll_g, aic_val=a_g,
                tail_lower=0.0, tail_upper=gumbel_upper_tail(th_g), n=n
            ))
        except Exception as exc:
            warnings.warn(f"Gumbel fit failed for {group_name}: {exc}")

        return DependenceResult(
            group=group_name, n=n,
            kendall_tau=float(tau), kendall_p=float(tau_p),
            spearman_rho=float(rho), spearman_p=float(rho_p),
            copulas=copulas
        )

    def fit_all(self) -> dict[str, DependenceResult]:
        """Fit copulas for overall dataset and each domain."""
        if self.data is None:
            self.load()

        df = self.data

        # Overall
        self.results["Overall"] = self._fit_group(df, "Overall")

        # Per domain
        if "domain" in df.columns:
            for domain, sub in df.groupby("domain"):
                if len(sub) >= 10:  # Need enough data
                    self.results[domain] = self._fit_group(sub, domain)

        return self.results

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising results for all groups."""
        if not self.results:
            self.fit_all()

        rows = []
        for group, res in self.results.items():
            best = res.best_copula
            rows.append({
                "group": group,
                "n": res.n,
                "kendall_tau": res.kendall_tau,
                "spearman_rho": res.spearman_rho,
                "best_copula": best.family if best else "N/A",
                "theta": best.theta if best else float("nan"),
                "aic": best.aic_val if best else float("nan"),
                "lambda_L": best.tail_lower if best else float("nan"),
                "lambda_U": best.tail_upper if best else float("nan"),
            })
        return pd.DataFrame(rows)
