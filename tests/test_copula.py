"""
25 tests for EvidenceCopula engine.

T1-T3:   empirical_cdf
T4-T8:   Clayton fit
T9-T12:  Frank fit
T13-T16: Gumbel fit
T17-T18: Model selection (AIC comparison)
T19-T20: Kendall tau, Spearman rho
T21-T25: Pipeline integration
"""

import math
import sys
import os
import warnings

import numpy as np
import pytest
from scipy import stats

# Make copula_engine importable from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copula_engine import (
    empirical_cdf,
    z_from_p,
    z_array_from_p,
    clayton_logpdf,
    fit_clayton,
    clayton_lower_tail,
    frank_logpdf,
    fit_frank,
    gumbel_logpdf,
    fit_gumbel,
    gumbel_upper_tail,
    aic,
    EvidenceCopulaEngine,
    CopulaResult,
    DependenceResult,
)

# ============================================================
# T1-T3: empirical_cdf
# ============================================================

class TestEmpiricalCDF:

    def test_t1_sorted_ascending(self):
        """T1: Output of empirical_cdf should be sorted when input is sorted."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        u = empirical_cdf(x)
        assert np.all(np.diff(u) > 0), "CDF should be strictly increasing for distinct values"

    def test_t2_bounds_strictly_open(self):
        """T2: All pseudo-observations must be strictly in (0, 1)."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        u = empirical_cdf(x)
        assert np.all(u > 0), "All pseudo-observations must be > 0"
        assert np.all(u < 1), "All pseudo-observations must be < 1"

    def test_t3_known_ranks(self):
        """T3: For [3, 1, 2] with n=3, ranks are [3,1,2], pseudo-obs = ranks/4."""
        x = np.array([3.0, 1.0, 2.0])
        u = empirical_cdf(x)
        expected = np.array([3.0, 1.0, 2.0]) / 4.0
        np.testing.assert_allclose(u, expected, rtol=1e-12)


# ============================================================
# T4-T8: Clayton fit
# ============================================================

class TestClayton:

    @staticmethod
    def _sample_clayton(theta: float, n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample from Clayton copula using Gamma frailty (Marshall-Olkin) method.

        U = (1 + E1/G)^{-1/theta}, V = (1 + E2/G)^{-1/theta}
        where G ~ Gamma(1/theta, 1) and E1, E2 ~ Exp(1).
        """
        rng = np.random.default_rng(seed)
        g = rng.gamma(shape=1.0 / theta, scale=1.0, size=n)
        e1 = rng.exponential(1.0, n)
        e2 = rng.exponential(1.0, n)
        u = (1.0 + e1 / g) ** (-1.0 / theta)
        v = (1.0 + e2 / g) ** (-1.0 / theta)
        u = np.clip(u, 0.01, 0.99)
        v = np.clip(v, 0.01, 0.99)
        return u, v

    def test_t4_known_theta(self):
        """T4: Fit on large Clayton sample recovers theta within ±0.4."""
        theta_true = 2.0
        n = 2000
        u, v = self._sample_clayton(theta_true, n)
        theta_hat, loglik = fit_clayton(u, v)
        assert abs(theta_hat - theta_true) < 0.4, \
            f"theta_hat={theta_hat:.3f} too far from true {theta_true}"

    def test_t5_positive_dependence(self):
        """T5: Clayton theta must be positive (lower-tail dependence)."""
        rng = np.random.default_rng(1)
        u = rng.uniform(0.05, 0.95, 300)
        v = 0.6 * u + 0.4 * rng.uniform(0.05, 0.95, 300)  # positive dependence
        v = np.clip(v, 0.01, 0.99)
        theta_hat, _ = fit_clayton(u, v)
        assert theta_hat > 0, f"Clayton theta must be positive, got {theta_hat}"

    def test_t6_independence_theta_near_zero(self):
        """T6: Independence (random pairs) should give theta near 0 (small)."""
        rng = np.random.default_rng(2)
        u = rng.uniform(0.01, 0.99, 500)
        v = rng.uniform(0.01, 0.99, 500)
        theta_hat, _ = fit_clayton(u, v)
        # Under independence, theta can be small but positive in MLE
        # We check it's in a reasonable range (not wildly large)
        assert theta_hat < 2.0, \
            f"Under independence, Clayton theta should be small, got {theta_hat:.3f}"

    def test_t7_lower_tail_dependence_formula(self):
        """T7: lambda_L = 2^(-1/theta) for theta=2 => 2^(-0.5) = 0.7071."""
        lam = clayton_lower_tail(2.0)
        assert abs(lam - 2.0 ** (-0.5)) < 1e-10

    def test_t8_logpdf_finite(self):
        """T8: Clayton log-density is finite and negative for valid (u,v,theta)."""
        rng = np.random.default_rng(3)
        u = rng.uniform(0.1, 0.9, 50)
        v = rng.uniform(0.1, 0.9, 50)
        ld = clayton_logpdf(u, v, theta=1.5)
        assert np.all(np.isfinite(ld)), "All log-densities should be finite"


# ============================================================
# T9-T12: Frank fit
# ============================================================

class TestFrank:

    @staticmethod
    def _sample_frank(theta: float, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample from Frank copula via conditional inversion.

        C(v|u) = p => v = -1/theta * log(1 + p*(exp(-theta)-1) / (exp(-theta*u)*(1-p)+p))
        """
        rng = np.random.default_rng(seed)
        u = rng.uniform(0.01, 0.99, n)
        p = rng.uniform(0.01, 0.99, n)
        et = math.exp(-theta)
        etu = np.exp(-theta * u)
        denom_cond = etu * (1.0 - p) + p
        # arg = 1 + p*(et-1)/denom_cond; must be >0 for valid samples
        arg = 1.0 + p * (et - 1.0) / denom_cond
        arg = np.clip(arg, 1e-10, None)
        v = -1.0 / theta * np.log(arg)
        v = np.clip(v, 0.01, 0.99)
        return u, v

    def test_t9_known_theta_positive(self):
        """T9: Frank fit on large positive-theta sample should recover theta within ±1.5."""
        theta_true = 4.0
        u, v = self._sample_frank(theta_true, 2000, seed=10)
        theta_hat, _ = fit_frank(u, v)
        assert abs(theta_hat - theta_true) < 1.5, \
            f"Frank theta_hat={theta_hat:.3f} too far from true {theta_true}"

    def test_t10_symmetry(self):
        """T10: Frank copula is symmetric: fit(u,v) == fit(v,u) theta."""
        rng = np.random.default_rng(20)
        u = rng.uniform(0.05, 0.95, 300)
        v = rng.uniform(0.05, 0.95, 300)
        th1, _ = fit_frank(u, v)
        th2, _ = fit_frank(v, u)
        # Due to symmetry of Frank copula, theta estimates should be very close
        assert abs(th1 - th2) < 0.5, f"Frank symmetry: {th1:.3f} vs {th2:.3f}"

    def test_t11_logpdf_near_zero_theta(self):
        """T11: Frank log-pdf with theta near 0 returns zeros (independence)."""
        rng = np.random.default_rng(30)
        u = rng.uniform(0.1, 0.9, 20)
        v = rng.uniform(0.1, 0.9, 20)
        ld = frank_logpdf(u, v, theta=1e-10)
        np.testing.assert_allclose(ld, 0.0, atol=1e-6)

    def test_t12_no_tail_dependence(self):
        """T12: Frank copula has no tail dependence (lambda_L = lambda_U = 0)."""
        from copula_engine import FRANK_TAIL_DEPENDENCE
        assert FRANK_TAIL_DEPENDENCE == 0.0


# ============================================================
# T13-T16: Gumbel fit
# ============================================================

class TestGumbel:

    @staticmethod
    def _sample_gumbel(theta: float, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample from Gumbel copula via Marshall-Olkin algorithm
        using stable distribution with Laplace-transform.
        Simplified: use correlated exponentials.
        """
        rng = np.random.default_rng(seed)
        # Use known sampling: generate V ~ Stable(1/theta, 1, ...) then
        # U1 = exp(-E1/V^(1/theta)), U2 = exp(-E2/V^(1/theta))
        # where E1, E2 ~ Exp(1), V ~ Stable
        # Approximate via: V = (cos(pi/(2*theta)) / X)^theta where X ~ Gamma(1-1/theta, 1)
        # For simplicity with theta >= 1, use a known exact approach:
        alpha = 1.0 / theta
        # Sample from positive stable(alpha) via Chambers-Mallows-Stuck
        phi = rng.uniform(0, math.pi, n)
        w = rng.exponential(1.0, n)
        s = (np.sin(alpha * phi) / np.sin(phi)) ** (1.0 / alpha) * \
            (np.sin((1.0 - alpha) * phi) / w) ** ((1.0 - alpha) / alpha)
        # Clip to avoid numerical issues
        s = np.clip(s, 1e-6, None)
        e1 = rng.exponential(1.0, n)
        e2 = rng.exponential(1.0, n)
        u = np.exp(-e1 / s)
        v = np.exp(-e2 / s)
        u = np.clip(u, 0.01, 0.99)
        v = np.clip(v, 0.01, 0.99)
        return u, v

    def test_t13_known_theta(self):
        """T13: Gumbel fit should recover theta >= 1 within ±0.5 for theta=2."""
        theta_true = 2.0
        u, v = self._sample_gumbel(theta_true, 2000, seed=40)
        theta_hat, _ = fit_gumbel(u, v)
        assert abs(theta_hat - theta_true) < 0.5, \
            f"Gumbel theta_hat={theta_hat:.3f} too far from true {theta_true}"

    def test_t14_theta_geq_1(self):
        """T14: Gumbel theta estimate must always be >= 1."""
        rng = np.random.default_rng(50)
        u = rng.uniform(0.05, 0.95, 200)
        v = rng.uniform(0.05, 0.95, 200)
        theta_hat, _ = fit_gumbel(u, v)
        assert theta_hat >= 1.0, f"Gumbel theta must be >= 1, got {theta_hat:.4f}"

    def test_t15_upper_tail_formula(self):
        """T15: lambda_U = 2 - 2^(1/theta) for theta=2 => 2 - 2^0.5 = 0.5858."""
        lam = gumbel_upper_tail(2.0)
        expected = 2.0 - 2.0 ** 0.5
        assert abs(lam - expected) < 1e-10

    def test_t16_logpdf_finite(self):
        """T16: Gumbel log-density is finite for valid inputs."""
        rng = np.random.default_rng(60)
        u = rng.uniform(0.1, 0.9, 50)
        v = rng.uniform(0.1, 0.9, 50)
        ld = gumbel_logpdf(u, v, theta=2.0)
        assert np.all(np.isfinite(ld)), "Gumbel log-density must be finite for valid inputs"


# ============================================================
# T17-T18: Model selection (AIC)
# ============================================================

class TestModelSelection:

    def test_t17_aic_formula(self):
        """T17: AIC = 2k - 2*logL."""
        assert aic(100.0, k=1) == pytest.approx(2 * 1 - 2 * 100.0)
        assert aic(-50.0, k=2) == pytest.approx(4 - 2 * (-50.0))

    def test_t18_lower_aic_wins(self):
        """T18: Model with lower AIC is selected as best."""
        results = [
            CopulaResult("Clayton", 1.5, loglik=200.0, aic_val=aic(200.0, 1),
                         tail_lower=0.35, tail_upper=0.0, n=100),
            CopulaResult("Frank", 3.0, loglik=210.0, aic_val=aic(210.0, 1),
                         tail_lower=0.0, tail_upper=0.0, n=100),
            CopulaResult("Gumbel", 2.0, loglik=205.0, aic_val=aic(205.0, 1),
                         tail_lower=0.0, tail_upper=0.41, n=100),
        ]
        best = min(results, key=lambda c: c.aic_val)
        assert best.family == "Frank", f"Expected Frank to win, got {best.family}"


# ============================================================
# T19-T20: Kendall tau, Spearman rho
# ============================================================

class TestCorrelations:

    def test_t19_kendall_tau_known(self):
        """T19: Kendall tau for perfectly monotone data = 1.0."""
        x = np.arange(1, 21, dtype=float)
        y = np.arange(1, 21, dtype=float)
        tau, p = stats.kendalltau(x, y)
        assert abs(tau - 1.0) < 1e-10

    def test_t20_spearman_rho_known(self):
        """T20: Spearman rho for perfectly monotone data = 1.0."""
        x = np.arange(1, 21, dtype=float)
        y = np.arange(1, 21, dtype=float)
        rho, p = stats.spearmanr(x, y)
        assert abs(rho - 1.0) < 1e-10


# ============================================================
# T21-T25: Pipeline integration
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@pytest.fixture(scope="module")
def engine():
    scores_path = os.path.join(DATA_DIR, "scores.csv")
    verdicts_path = os.path.join(DATA_DIR, "verdicts.csv")
    groups_path = os.path.join(DATA_DIR, "review_groups.csv")
    eng = EvidenceCopulaEngine(scores_path, verdicts_path, groups_path)
    eng.load()
    return eng


@pytest.fixture(scope="module")
def fitted_engine(engine):
    engine.fit_all()
    return engine


class TestPipeline:

    def test_t21_load_data_shape(self, engine):
        """T21: Loaded data has expected columns and enough rows."""
        df = engine.data
        assert df is not None
        assert "final_score" in df.columns
        assert "z_stat" in df.columns
        assert "p_value" in df.columns
        assert len(df) >= 100

    def test_t22_z_stat_nonneg(self, engine):
        """T22: All z-statistics are non-negative (absolute values)."""
        df = engine.data
        assert np.all(df["z_stat"].values >= 0), "z-statistics must be non-negative"

    def test_t23_overall_result_exists(self, fitted_engine):
        """T23: 'Overall' group must be in results."""
        assert "Overall" in fitted_engine.results

    def test_t24_all_copulas_fitted(self, fitted_engine):
        """T24: Overall result must contain Clayton, Frank, and Gumbel."""
        res = fitted_engine.results["Overall"]
        families = {c.family for c in res.copulas}
        assert "Clayton" in families
        assert "Frank" in families
        assert "Gumbel" in families

    def test_t25_summary_dataframe(self, fitted_engine):
        """T25: Summary DataFrame has expected columns and at least one row."""
        df = fitted_engine.summary()
        required = ["group", "n", "kendall_tau", "spearman_rho",
                    "best_copula", "theta", "aic", "lambda_L", "lambda_U"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
        assert len(df) >= 1
        # Best copula should be one of the three families
        best = df.loc[df["group"] == "Overall", "best_copula"].iloc[0]
        assert best in {"Clayton", "Frank", "Gumbel"}, f"Unexpected best copula: {best}"
