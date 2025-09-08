#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entropy–Gated Response (EGR) — Galaxy Rotation Curves

Referee-grade engineering: robust CLI, diagnostics, reproducible outputs.
Physics and parameter semantics are unchanged from the supplied script:

  • Gate: g = Q_exp / (Q_exp + σ_p), with Q_exp = R * S_resp
  • Two model families:
      EGR-7  : [R, φ_w, A_w, A_p, k_p, σ_p, S_resp]
      EGR-10 : EGR-7 + [A_ring, r_ring, σ_ring]
  • Error model: σ_eff^2(r) = σ_meas^2(r) + (s_frac * V_obs(r))^2, s_frac ≥ 0
  • Model selection: keep EGR-10 iff ΔBIC ≤ −bic_thresh (default 10)
  • Multiple random starts, widened-bounds fallbacks, ring-pinned fairness baseline

Outputs:
  out/figs/<ID>_rc.png                — per-galaxy figure
  out/tables/summary.csv              — per-galaxy metrics (one row/galaxy)
  out/tables/posthoc.json             — global aggregates/histograms
  out/run.json                        — run metadata (args, versions, timestamp)
"""
# --------------- your original code begins here unchanged ---------------
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit, OptimizeWarning
from scipy import stats

plt.style.use("dark_background")
warnings.filterwarnings("ignore", category=OptimizeWarning)

# Optional MCMC/corner
try:
    import emcee, corner
    HAS_EMCEE = True
except Exception:
    HAS_EMCEE = False
    
# ======================= EntropyLedgerMatrix (with quaternion gates) =======================

class EntropyLedgerMatrix:
    """
    φ-energized 8×8 patterns for diagnostics/geometry with quaternion-style gates.

    Patterns
    --------
    hanging_left            : left-aligned, length = 8-row
                              sign pattern: +,−,+,−,...
                              energy map:   + → E_plus (= φ), − → E_minus (= 1/φ)
    hanging_right_inverted  : right-aligned mirror with inverted energy map
                              sign pattern: +,−,+,−,...
                              energy map:   + → E_minus, − → E_plus

    Modes (weighting)
    -----------------
    mode='flat'    : raw pattern values only
    mode='bounded' : multiply each cell by a safe composite weight:
                       • Radial Decay Component (RDC):      1/r falloff, squashed to (1−α, 1+α)
                       • Phase Accumulation Component (PAC): cos(phase), squashed to (1−β, 1+β)
                     Final weight = RDC × PAC ∈ [(1-α)(1-β), (1+α)(1+β)]

    Quaternion-style gates (right-multiply)
    ---------------------------------------
    transform_and(M) : φ-skewed 4×4 block lifted to 8×8 via kron (CNOT-like)
    transform_xor(M) : Hadamard⊗3 (XOR-like entangler)
    transform_not(M) : I ⊗ I ⊗ X   (bit-flip on least-significant logical axis)
    apply_transformation(gate, M) : gate ∈ {'S_AND','S_XOR','S_NOT'}

    Notes
    -----
    - This class is self-contained and independent of the RC fitter.
    - Complex/quaternion gates are the interface you want for Penrose-type
      complex structures in spacetime; the base patterns remain unchanged.
    - Visuals typically plot M.real; the complex phase is preserved in M.
    """

    def __init__(
        self,
        mode: str = "bounded",   # 'bounded' | 'flat'
        alpha: float = 0.35,     # blend strength for Radial Decay Component (RDC)
        beta: float  = 0.35,     # blend strength for Phase Accumulation Component (PAC)
        rdc_scale: float = 1.0,  # scaling applied before RDC squash
        pac_scale: float = 1.0,  # scaling applied before PAC squash
    ):
        if mode not in {"bounded", "flat"}:
            raise ValueError("mode must be 'bounded' or 'flat'")
        self.mode = mode
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.rdc_scale = float(max(rdc_scale, 1e-12))
        self.pac_scale = float(max(pac_scale, 1e-12))

        # φ-ledger constants/energies
        self.phi = (1.0 + np.sqrt(5.0)) / 2.0
        self.E_plus  = self.phi
        self.E_minus = 1.0 / self.phi
        self.RETURN_RATE = 1.0 / self.phi
        self.BORROW_RATE = 1.0 / (self.phi**2)
        self.TRANSACTION_COST = self.RETURN_RATE - self.BORROW_RATE

    # ---------- component squashes ----------
    @staticmethod
    def _tanh(x: float) -> float:
        return float(np.tanh(x))  # maps ℝ → (−1,1)

    @staticmethod
    def _sigm(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))  # maps ℝ → (0,1)

    def _weight(self, r_idx: int, c_idx: int) -> float:
        """
        Compute composite weight at grid cell (r,c).

        Radial Decay Component (RDC):  1/r falloff, squashed to roughly (1−α,1+α)
        Phase Accumulation Component (PAC): cos(phase), squashed to roughly (1−β,1+β)

        Final weight = RDC × PAC, clipped to [1e-6, 1e6] for numerical safety.
        """
        if self.mode == "flat":
            return 1.0

        # RDC: distance-based attenuation
        r = float(np.hypot(r_idx, c_idx)) + 1e-9
        rdc_raw = 1.0 / r
        rdc = 1.0 + self.alpha * self._tanh(rdc_raw / self.rdc_scale)

        # PAC: oscillatory checker/stripe phase pattern
        phase   = 2.0 * np.pi * (r_idx + c_idx) / 8.0
        pac_raw = np.cos(phase)
        pac = 1.0 + self.beta * (2.0 * self._sigm(pac_raw / self.pac_scale) - 1.0)

        w = float(rdc * pac)
        return float(np.clip(w, 1e-6, 1e6))

    # ---------- core builder ----------
    def _build(self, pattern_fn) -> np.ndarray:
        M = np.zeros((8, 8), dtype=complex)
        for r in range(8):
            for c, base in pattern_fn(r):
                M[r, c] = complex(base) * self._weight(r, c)
        M[np.abs(M) < 1e-12] = 0.0
        return M

    # ---------- patterns ----------
    def _pat_hanging_left(self):
        """Left-aligned hanging fill; signs alternate, +→E_plus, −→E_minus (then multiplied by sign)."""
        def p(row: int):
            out = []
            length = 8 - row
            for i in range(length):
                sign = 1 if (i % 2 == 0) else -1
                energy = self.E_plus if sign > 0 else self.E_minus
                out.append((i, sign * energy))
            return out
        return p

    def _pat_hanging_right_inverted(self):
        """Right-aligned mirror with inverted energy mapping (+→E_minus, −→E_plus), multiplied by sign."""
        def p(row: int):
            out = []
            length = 8 - row
            for i in range(length):
                sign = 1 if (i % 2 == 0) else -1
                energy = self.E_minus if sign > 0 else self.E_plus
                col = 7 - i  # fill from right edge inward
                out.append((col, sign * energy))
            return out
        return p

    # ---------- public matrices ----------
    def matrix_hanging_left(self) -> np.ndarray:
        return self._build(self._pat_hanging_left())

    def matrix_hanging_right_inverted(self) -> np.ndarray:
        return self._build(self._pat_hanging_right_inverted())

    def bank(self) -> Dict[str, np.ndarray]:
        return {
            "hanging_left": self.matrix_hanging_left(),
            "hanging_right_inverted": self.matrix_hanging_right_inverted(),
        }

    # ---------- quaternion-style gates (right-multiply) ----------
    # These are exactly the interfaces you wanted for Penrose-type complex structures.
    # Keep them pure linear ops on C^(8×8); your visuals can plot real/imag parts as needed.

    def transform_and(self, M: np.ndarray, phase_scale: float = 1.0) -> np.ndarray:
        """
        CNOT-like φ-skewed 4×4 lifted to 8×8 via kron; right-multiply columns.

        q_φ = [[ φ,  1,  φ^-1, 0],
               [ 0,  φ,   1,   φ^-1],
               [ φ^-1, 0,  φ,   1   ],
               [ 1,  φ^-1, 0,   φ   ]]

        Gate = exp(i * π/φ * phase_scale) * kron(q_φ, I_2)
        """
        q = np.array(
            [[ self.phi,      1.0,       self.phi**-1, 0.0],
             [ 0.0,           self.phi,   1.0,         self.phi**-1],
             [ self.phi**-1,  0.0,        self.phi,     1.0],
             [ 1.0,           self.phi**-1, 0.0,       self.phi]],
            dtype=complex
        )
        G8 = np.kron(q, np.eye(2, dtype=complex))
        return M.astype(complex) @ (np.exp(1j * np.pi / self.phi * phase_scale) * G8)

    def transform_xor(self, M: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """
        XOR-like entangler: Hadamard⊗3; right-multiply.
        H = (1/√2) [[1, 1], [1, -1]]; Gate = (π/φ * scale) * H⊗H⊗H
        """
        H = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0],[1.0, -1.0]], dtype=complex)
        H8 = np.kron(np.kron(H, H), H)
        return M.astype(complex) @ (H8 * (np.pi / self.phi) * scale)

    def transform_not(self, M: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """
        NOT-like: I ⊗ I ⊗ X; right-multiply.
        """
        X  = np.array([[0.0, 1.0],[1.0, 0.0]], dtype=complex)
        X8 = np.kron(np.kron(np.eye(2, dtype=complex), np.eye(2, dtype=complex)), X)
        return M.astype(complex) @ (X8 * (np.pi / self.phi) * scale)

    def apply_transformation(self, gate: str, M: np.ndarray, **kw) -> np.ndarray:
        """
        gate ∈ {'S_AND','S_XOR','S_NOT'}. Extra kwargs forwarded (e.g., phase_scale, scale).
        """
        if gate == "S_AND": return self.transform_and(M, **kw)
        if gate == "S_XOR": return self.transform_xor(M, **kw)
        if gate == "S_NOT": return self.transform_not(M, **kw)
        raise ValueError(f"Unsupported gate: {gate}")

# ============================== EGR model ==============================

class EGRRotationCurveModel:
    """
    Entropy–Gated Response (EGR) rotation-curve model.

    Parameters (EGR-7 base):
      R         : scale factor for Q_exp = R * S_resp
      φ_w       : characteristic radius for vector envelope (phi_w)
      A_w       : vector-envelope amplitude
      A_p       : scalar-envelope amplitude
      k_p       : scalar-envelope scale length
      σ_p       : gate softness parameter (in g = Q_exp / (Q_exp + σ_p))
      S_resp    : specific heating scale (enters Q_exp)

    Ring extension (EGR-10):
      A_ring    : Gaussian ring amplitude
      r_ring    : ring center radius
      σ_ring    : ring width

    The observable velocity profile is a causal low-pass combination of
    a vector-like envelope and a scalar-like core envelope, both gated
    by g(Q_exp), optionally with a localized Gaussian ring.
    """

    def __init__(self, alpha_c: float = 0.25) -> None:
        self.alpha_c = float(alpha_c)

    # --- component builders (names reflect roles) ---

    def _vector_envelope(self, r: np.ndarray, Qexp: float, A_w: float, phi_w: float) -> np.ndarray:
        r = np.asarray(r, float)
        vec = A_w * (1.0 - np.exp(-r / max(phi_w, 1e-12)))
        return vec * (Qexp / (Qexp + 1.0))

    def _scalar_envelope(self, r: np.ndarray, Qexp: float, A_p: float, k_p: float) -> np.ndarray:
        r = np.asarray(r, float)
        kp = max(k_p, 1e-9)
        core   = np.exp(-(r / kp) ** 1.0)
        large  = 1.0 / (1.0 + (r / (12.0 * kp)) ** 1.8)
        smooth = 1.0 / (1.0 + 0.1 * (r / kp))
        sca = A_p * core * large * smooth
        return sca * (Qexp / (Qexp + 1e-9))

    def _coupling_envelope(self, r: np.ndarray) -> np.ndarray:
        lam = 2.0
        trans  = 1.0 - np.exp(-r / (lam + 1e-10))
        shadow = np.exp(-0.2 * r)
        return 1.0 + self.alpha_c * (0.7 * trans + 0.3 * shadow)

    def _ring_component(self, r: np.ndarray, A_ring: float, r_ring: float, sigma_ring: float) -> np.ndarray:
        sig = max(sigma_ring, 1e-10)
        return A_ring * np.exp(-0.5 * ((r - r_ring) / sig) ** 2)

    # --- full model evaluator ---

    def velocity(
        self,
        r: np.ndarray,
        R: float,
        phi_w: float,
        A_w: float,
        A_p: float,
        k_p: float,
        sigma_p: float,
        S_resp: float,
        A_ring: float = 0.0,
        r_ring: float = 0.0,
        sigma_ring: float = 1.0,
    ) -> np.ndarray:
        """
        Return model rotation curve V(r) for the given parameters.
        """
        Qexp = R * max(S_resp, 1e-15)
        vec = self._vector_envelope(r, Qexp, A_w, phi_w)
        sca = self._scalar_envelope(r, Qexp, A_p, k_p)
        r_norm = np.asarray(r, float) / (np.mean(r) + 1e-10)
        weight_core = 1.0 / (1.0 + (r_norm / 1.2) ** 1.5)

        v = (vec + sca * weight_core)
        v *= self._coupling_envelope(r)
        v *= Qexp / (Qexp + max(sigma_p, 1e-12))  # EGR gate g(Qexp)

        if abs(A_ring) > 0 or sigma_ring > 0:
            v += self._ring_component(r, A_ring, r_ring, sigma_ring)
        return v

# ============================ utilities/data ============================

COLUMN_ALIASES = {
    "R": ["R", "radius", "r"],
    "Vobs": ["Vobs", "V_obs", "Vrot", "vrot", "v_obs", "V_rot", "V"],
    "e_Vobs": ["e_Vobs", "eVobs", "Verr", "e_Vrot", "e_vrot", "sigma_V", "sigma", "dV"],
    "ID": ["ID", "Id", "id", "Name", "name", "galaxy", "galname"],
}

def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Resolve common column aliases to canonical names: ID, R, Vobs, e_Vobs.
    """
    canon = {c.lower().replace("_", ""): c for c in df.columns}
    mapping: Dict[str, str] = {}
    for key, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            ak = a.lower().replace("_", "")
            if ak in canon:
                mapping[key] = canon[ak]
                break
    df2 = df.copy()
    for k in ["ID", "R", "Vobs", "e_Vobs"]:
        if k in mapping:
            df2[k] = df[mapping[k]]
    return df2, mapping

def robust_sigma_floor(verr: np.ndarray) -> np.ndarray:
    """
    Apply a conservative floor to measurement uncertainties.
    Floor = max(0.1 * median positive, 1e-3).
    """
    verr = np.asarray(verr, float)
    pos = verr[np.isfinite(verr) & (verr > 0)]
    base = float(np.median(pos)) if pos.size else 1.0
    return np.clip(verr, max(0.1 * base, 1e-3), None)

def data_driven_seeds(r: np.ndarray, v: np.ndarray) -> Tuple[Dict[str, float], float, float]:
    """
    Heuristics for initial seeds from data ranges.
    """
    r = np.asarray(r, float); v = np.asarray(v, float)
    rmax = max(float(np.max(r)), 1.0)
    vmax = float(np.nanmax(v)) if np.isfinite(v).any() else 200.0
    try:
        ridx = int(np.where(v >= 0.6 * vmax)[0][0])
        r_turn = max(float(r[ridx]), 0.2 * rmax)
    except Exception:
        r_turn = 0.3 * rmax
    seeds = dict(
        A_w=max(0.6 * vmax, 60.0),
        A_p=max(0.4 * vmax, 40.0),
        phi_w=max(0.2 * rmax, 0.8 * r_turn),
        k_p=max(0.1 * rmax, 0.5 * r_turn),
    )
    return seeds, rmax, vmax

def bounds_egr7(r: np.ndarray, seeds: Mapping[str, float]) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Default bounds and seeds for EGR-7.
    """
    rmax = max(float(np.max(r)), 1.0)
    order = ["R","phi_w","A_w","A_p","k_p","sigma_p","S_resp","A_ring","r_ring","sigma_ring"]
    p0 = np.array([1.0, max(0.3, seeds["phi_w"]), seeds["A_w"], seeds["A_p"],
                   max(0.08*rmax, seeds["k_p"]), 10.0, 3.0, 0.0, 0.0, 1.0], float)
    lb = np.array([0.0, 0.05, 0.0, 0.0, max(0.05, 0.05*rmax), 0.1, 0.1, 0.0, 0.0, 1.0], float)
    ub = np.array([15.0, 35.0, 500.0, 500.0, max(2.0, 0.7*rmax), 150.0, 80.0, 0.0, 0.0, 1.0], float)
    return order, p0, lb, ub

def bounds_egr10(r: np.ndarray, seeds: Mapping[str, float]) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Default bounds and seeds for EGR-10 (ring enabled).
    """
    r = np.asarray(r, float)
    rmax = max(float(np.max(r)), 1.0)
    rpos = r[r > 0]; rmin = float(np.min(rpos)) if rpos.size else 0.0
    order = ["R","phi_w","A_w","A_p","k_p","sigma_p","S_resp","A_ring","r_ring","sigma_ring"]
    p0 = np.array([1.0, max(0.3, seeds["phi_w"]), seeds["A_w"], seeds["A_p"],
                   max(0.08*rmax, seeds["k_p"]), 10.0, 3.0, 20.0,
                   np.clip(0.4*rmax, max(0.8*rmin, 0.0), 1.1*rmax), 0.2*rmax], float)
    lb = np.array([0.0, 0.05, 0.0, 0.0, max(0.05, 0.05*rmax), 0.1, 0.1, 0.0,
                   max(0.8*rmin, 0.0), 0.05*rmax], float)
    ub = np.array([15.0, 35.0, 500.0, 500.0, max(2.0, 0.7*rmax), 150.0, 80.0,
                   250.0, 1.1*rmax, 0.8*rmax], float)
    return order, p0, lb, ub

def widen_bounds(lb: np.ndarray, ub: np.ndarray, factor: float = 2.2) -> Tuple[np.ndarray, np.ndarray]:
    lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    span = ub - lb; mid = lb + 0.5 * span
    return np.maximum(0.0, mid - 0.5 * factor * span), mid + 0.5 * factor * span

def count_free_params(lb: Sequence[float], ub: Sequence[float], eps: float = 1e-12) -> int:
    lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    return int(np.sum(ub > lb + eps))

def count_bound_hits(theta: Sequence[float], lb: Sequence[float], ub: Sequence[float], rel_tol: float = 0.02) -> int:
    """
    Count params within rel_tol*span of either bound (ignores pinned params).
    """
    t = np.asarray(theta, float); lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    span = ub - lb
    hits = 0
    for v, l, u, s in zip(t, lb, ub, span):
        if u <= l or s <= 0:
            continue
        if (v - l) <= rel_tol * s or (u - v) <= rel_tol * s:
            hits += 1
    return int(hits)

def multiple_random_starts(
    func,
    x: np.ndarray,
    y: np.ndarray,
    sig: np.ndarray,
    p0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    n_starts: int = 14,
    f_scale: float = 0.5,
    rng: Optional[np.random.Generator] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Multi-start robust fit using TRF + soft_l1 loss; returns best (popt, pcov, chi2).
    """
    rng = np.random.default_rng() if rng is None else rng
    best, best_V, best_chi2 = None, None, np.inf
    for _ in range(n_starts):
        jitter = (1.0 + 0.35 * (2 * rng.random(len(p0)) - 1.0))
        g = np.clip(p0 * jitter, lb, ub)
        try:
            popt, pcov = curve_fit(
                func, x, y, p0=g, sigma=sig, bounds=(lb, ub),
                method="trf", loss="soft_l1", f_scale=f_scale, maxfev=60000
            )
            pred = func(x, *popt)
            chi2 = float(np.sum(((y - pred) / sig) ** 2))
            if chi2 < best_chi2:
                best, best_V, best_chi2 = popt, pcov, chi2
        except Exception:
            pass
    return best, best_V, best_chi2

def coarse_egr7_baseline(
    func,
    r: np.ndarray,
    vobs: np.ndarray,
    sig: np.ndarray,
    p0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    n_tries: int = 16,
    widen: float = 3.5,
    rng: Optional[np.random.Generator] = None
) -> Optional[np.ndarray]:
    """
    Aggressive EGR-7 fallback: widened bounds + random starts.
    """
    rng = np.random.default_rng() if rng is None else rng
    lb_w, ub_w = widen_bounds(lb, ub, factor=widen)
    best, best_chi2 = None, np.inf
    for _ in range(n_tries):
        g = lb_w + rng.random(len(p0)) * (ub_w - lb_w)
        try:
            popt, _ = curve_fit(
                func, r, vobs, p0=g, sigma=sig, bounds=(lb_w, ub_w),
                method="trf", loss="soft_l1", f_scale=1.2, maxfev=80000
            )
            pred = func(r, *popt)
            chi2 = float(np.sum(((vobs - pred) / sig) ** 2))
            if chi2 < best_chi2:
                best, best_chi2 = popt, chi2
        except Exception:
            continue
    return best

def implied_scatter_for_unit_chi2(
    vobs: np.ndarray,
    vmod: np.ndarray,
    verr: np.ndarray,
    dof: int,
    s_lo: float = 0.0,
    s_hi: float = 0.2,
    tol: float = 1e-4,
    iters: int = 60
) -> float:
    """
    Solve for s_frac so that χ²_red ≈ 1 given σ_eff^2 = σ_meas^2 + (s_frac * V_obs)^2.
    """
    def redchi(s: float) -> float:
        var = verr**2 + (s * vobs)**2
        return float(np.sum(((vobs - vmod)**2) / var) / max(dof, 1))
    if redchi(0.0) <= 1.0:
        return 0.0
    lo, hi = float(s_lo), float(s_hi)
    rc_hi = redchi(hi); expand = 0
    while rc_hi > 1.0 and hi < 0.5 and expand < 12:
        hi *= 1.5; rc_hi = redchi(hi); expand += 1
    for _ in range(iters):
        mid = 0.5 * (lo + hi); rc = redchi(mid)
        if abs(rc - 1.0) < tol:
            return max(0.0, mid)
        if rc > 1.0:
            lo = mid
        else:
            hi = mid
    return max(0.0, 0.5 * (lo + hi))

def ic_scores(ss_res: float, n: int, k: int) -> Tuple[float, float, float]:
    AIC  = 2*k + n*math.log((ss_res / max(n, 1)) + 1e-30)
    if n > k + 1:
        AICc = AIC + (2*k*(k+1)) / (n - k - 1)
    else:
        AICc = AIC  # or float('nan')
    BIC  = k*math.log(max(n, 1)) + n*math.log((ss_res / max(n, 1)) + 1e-30)
    return AIC, AICc, BIC


# ======================== residual diagnostics ========================

def durbin_watson(resid: np.ndarray) -> float:
    e = np.asarray(resid, float)
    de = np.diff(e)
    num = float(np.nansum(de * de))
    den = float(np.nansum(e * e) + 1e-30)
    return num / den

def ar1_rho(resid: np.ndarray) -> float:
    e = np.asarray(resid, float)
    if e.size < 2:
        return float("nan")
    e0, e1 = e[:-1], e[1:]
    den = float(np.nansum(e0 * e0))
    return float(np.nansum(e0 * e1) / den) if den > 0 else 0.0

def breusch_pagan_pvalue(resid: np.ndarray, X: np.ndarray) -> float:
    e = np.asarray(resid, float); n = e.size
    if n < 3:
        return float("nan")
    z = e**2; X = np.asarray(X, float)
    try:
        beta, _, _, _ = np.linalg.lstsq(X, z, rcond=None)
        z_hat = X @ beta
        ss_tot = float(np.sum((z - np.mean(z))**2)) + 1e-30
        ss_res = float(np.sum((z - z_hat)**2))
        R2 = 1.0 - ss_res / ss_tot
        k = X.shape[1] - 1
        lm = n * max(R2, 0.0)
        return float(stats.chi2.sf(lm, df=max(k, 1)))
    except Exception:
        return float("nan")

# =============================== data I/O ===============================

@dataclass
class FitResult:
    ID: str
    n_points_used: int
    r_min: float
    r_max: float
    v_max: float
    model_chosen: str
    delta_BIC: float                # finite (clipped) value for histograms
    delta_BIC_raw: float            # may be +/-inf
    delta_BIC_chi: float            # χ²-based BIC difference (finite if both defined)
    bic_status: str                 # 'both_finite' | 'F7_fallback' | 'nested_ok' | 'F10_fail' | combos
    s_frac: float
    chi2_red: float
    R2: float
    AIC: float
    AICc: float
    BIC: float
    Qexp: float
    ring_A: float
    ring_r: float
    ring_sigma: float
    DW: float
    rho_AR1: float
    BP_pvalue: float
    conservative_sigma: bool
    resid_ar1_flagged: bool
    rmse_holdout: float
    mae_holdout: float
    r2_holdout: float
    delta_rmse_in: float
    delta_rmse_holdout: float
    bound_hits_chosen: int
    bound_hits_f7: int
    bound_hits_f10: int
    fit_status: str
    r: Optional[np.ndarray] = None
    vobs: Optional[np.ndarray] = None
    sig_eff: Optional[np.ndarray] = None
    pred: Optional[np.ndarray] = None

def load_data(path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_csv(path)
    df2, mapping = normalize_columns(df)
    return df2, mapping

def group_ids(df: pd.DataFrame) -> List[str]:
    for k in ["ID","Id","id","Name","name","galaxy","galname"]:
        if k in df.columns:
            return list(pd.unique(df[k].astype(str)))
    return ["G0000"]

# ============================= hold-out utils =============================

def outer_holdout_mask(r: np.ndarray, frac_outer: float = 0.2, nmin_train: int = 6) -> np.ndarray:
    """
    True for training indices (keep inner radii, hold out largest radii).
    """
    r = np.asarray(r, float)
    n = len(r)
    order = np.argsort(r)
    cut = int(np.floor((1.0 - frac_outer) * n))
    cut = max(min(cut, n - 1), nmin_train)
    train_idx = order[:cut]
    mask = np.zeros(n, dtype=bool); mask[train_idx] = True
    return mask

# ============================= per-galaxy fit =============================

def fit_one_galaxy(
    gal_id: str,
    subdf: pd.DataFrame,
    model: EGRRotationCurveModel,
    rng: np.random.Generator,
    nstarts7: int = 12,
    nstarts10: int = 12,
    bic_thresh: float = 10.0,
    holdout_frac: float = 0.0,
    do_rar: bool = True,
    holdout_repeats: int = 3,
    holdout_mode: str = "random",
    f7_widen: float = 3.5,
    f7_tries: int = 16,
    nested_f7: bool = False,
) -> FitResult:
    required = ["R","Vobs","e_Vobs"]
    missing = [c for c in required if c not in subdf.columns]
    if missing:
        return failure_result(gal_id, 0, f"missing_cols:{'|'.join(missing)}")

    r = np.asarray(subdf["R"].to_numpy(float))
    vobs = np.asarray(subdf["Vobs"].to_numpy(float))
    verr = robust_sigma_floor(np.asarray(subdf["e_Vobs"].to_numpy(float)))
    m = np.isfinite(r) & np.isfinite(vobs) & np.isfinite(verr)
    r, vobs, verr = r[m], vobs[m], verr[m]

    if r.size < 5:
        return failure_result(gal_id, int(r.size), "too_few_points")

    if np.nanmax(r) <= 0 or np.nanmax(vobs) <= 0:
        return failure_result(gal_id, int(r.size), "nonpositive_range")

    seeds, _, _ = data_driven_seeds(r, vobs)
    _, p07, lb7, ub7 = bounds_egr7(r, seeds)
    _, p10, lb10, ub10 = bounds_egr10(r, seeds)
    f = model.velocity

    bic_status = "both_finite"
    bound_hits_f7 = 0
    bound_hits_f10 = -1  # -1 → not attempted

    # ---------------- Stage 1: EGR-7 (baseline) ----------------
    popt7, _, _ = multiple_random_starts(f, r, vobs, verr, p07, lb7, ub7, n_starts=nstarts7, rng=rng)
    if popt7 is None:
        lb7_w, ub7_w = widen_bounds(lb7, ub7, factor=2.2)
        popt7, _, _ = multiple_random_starts(f, r, vobs, verr, p07, lb7_w, ub7_w, n_starts=max(4, nstarts7//2 + 6), rng=rng)
    if popt7 is None:
        popt7 = coarse_egr7_baseline(f, r, vobs, verr, p07, lb7, ub7, n_tries=f7_tries, widen=f7_widen, rng=rng)

    if popt7 is not None:
        v7 = f(r, *popt7)
        bound_hits_f7 = count_bound_hits(popt7, lb7, ub7)
        dof7 = max(len(r) - count_free_params(lb7, ub7), 1)
        s7 = implied_scatter_for_unit_chi2(vobs, v7, verr, dof7)
        sig7 = np.sqrt(verr**2 + (s7 * vobs)**2)
        popt7, _, _ = multiple_random_starts(f, r, vobs, sig7, popt7, lb7, ub7, n_starts=max(4, nstarts7//2), rng=rng)
        v7 = f(r, *popt7)
        ss_res7 = float(np.sum((vobs - v7)**2))
        k7 = count_free_params(lb7, ub7)
        _A7, _Ac7, BIC7 = ic_scores(ss_res7, len(r), k7)
    else:
        # Deterministic fallback for a defined baseline
        p_fallback7 = np.clip((lb7 + ub7) * 0.5, lb7, ub7)
        try:
            v7 = f(r, *p_fallback7)
        except Exception:
            v7 = f(r, *p07)
            p_fallback7 = p07.copy()
        bound_hits_f7 = count_bound_hits(p_fallback7, lb7, ub7)
        ss_res7 = float(np.sum((vobs - v7)**2))
        k7 = count_free_params(lb7, ub7)
        _A7, _Ac7, BIC7 = ic_scores(ss_res7, len(r), k7)
        sig7 = verr
        bic_status = "F7_fallback"

    # Optional nested fairness fit: ring-pinned EGR-10 baseline
    if nested_f7:
        lb_n = lb10.copy(); ub_n = ub10.copy(); p_n = p10.copy()
        mid_r = 0.5 * (lb10[8] + ub10[8]); mid_sig = 0.5 * (lb10[9] + ub10[9])
        lb_n[7] = ub_n[7] = 0.0
        lb_n[8] = ub_n[8] = mid_r
        lb_n[9] = ub_n[9] = mid_sig
        p_n[7] = 0.0; p_n[8] = mid_r; p_n[9] = mid_sig
        popt7_n, _, _ = multiple_random_starts(f, r, vobs, sig7, p_n, lb_n, ub_n, n_starts=nstarts10, rng=rng)
        if popt7_n is not None:
            v7_n = f(r, *popt7_n)
            ss7_n = float(np.sum((vobs - v7_n)**2))
            k7_n = count_free_params(lb_n, ub_n)
            _a, _ac, BIC7_n = ic_scores(ss7_n, len(r), k7_n)
            if BIC7_n < BIC7:
                v7 = v7_n; BIC7 = BIC7_n; k7 = k7_n
                bound_hits_f7 = count_bound_hits(popt7_n, lb_n, ub_n)
                bic_status = ("nested_ok" if bic_status == "both_finite" else (bic_status + "|nested_ok"))

    # ---------------- Stage 2: EGR-10 ----------------
    pstart10 = p10.copy()
    if popt7 is not None:
        pstart10[:len(popt7)] = popt7
    popt10, _, _ = multiple_random_starts(f, r, vobs, sig7, pstart10, lb10, ub10, n_starts=nstarts10, rng=rng)
    if popt10 is not None:
        v10 = f(r, *popt10)
        bound_hits_f10 = count_bound_hits(popt10, lb10, ub10)
        ss_res10 = float(np.sum((vobs - v10)**2))
        k10 = count_free_params(lb10, ub10)
        _A10, _Ac10, BIC10 = ic_scores(ss_res10, len(r), k10)
    else:
        v10, BIC10 = None, float("inf")
        bound_hits_f10 = -1
        if bic_status == "both_finite":
            bic_status = "F10_fail"
        elif "F10_fail" not in bic_status:
            bic_status += "|F10_fail"

    # ---------------- Selection ----------------
    delta_BIC_raw = BIC10 - BIC7
    use_ring = (delta_BIC_raw <= -float(bic_thresh))
    popt = (popt10 if (use_ring and (popt10 is not None)) else (popt7 if popt7 is not None else p07))
    pred = (v10 if (use_ring and (popt10 is not None)) else v7)
    chosen = ("EGR-10" if (use_ring and (popt10 is not None)) else "EGR-7")
    chosen_lb, chosen_ub = (lb10, ub10) if (use_ring and (popt10 is not None)) else (lb7, ub7)

    if popt is None or pred is None:
        return failure_result(gal_id, int(len(r)), "both_fail" if popt7 is None else "EGR7_fail")

    # ---------------- s_frac polish for chosen ----------------
    dof = max(len(r) - count_free_params(chosen_lb, chosen_ub), 1)
    s_hat = implied_scatter_for_unit_chi2(vobs, pred, verr, dof)
    for _ in range(2):
        sig_f = np.sqrt(verr**2 + (s_hat * vobs)**2)
        popt_new, _, _ = multiple_random_starts(
            f, r, vobs, sig_f, popt, chosen_lb, chosen_ub,
            n_starts=max(4, (nstarts10 if chosen == "EGR-10" else nstarts7)//2),
            rng=rng
        )
        if popt_new is None:
            break
        pred_new = f(r, *popt_new)
        s_new = implied_scatter_for_unit_chi2(vobs, pred_new, verr, dof)
        if abs(s_new - s_hat) < 1e-4:
            popt, pred, s_hat = popt_new, pred_new, s_new
            break
        popt, pred, s_hat = popt_new, pred_new, s_new

    # ---------------- Final metrics (chosen) ----------------
    sig_f = np.sqrt(verr**2 + (s_hat * vobs)**2)
    red = float(np.sum(((vobs - pred) / sig_f)**2) / max(dof, 1))
    ss_res = float(np.sum((vobs - pred)**2))
    k_eff = count_free_params(chosen_lb, chosen_ub)
    AIC, AICc, BIC = ic_scores(ss_res, len(r), k_eff)
    R2 = 1.0 - ss_res / (float(np.sum((vobs - np.mean(vobs))**2)) + 1e-30)

    # χ²-based IC cross-check
    dof7 = max(len(r) - k7, 1)
    s7_hat = implied_scatter_for_unit_chi2(vobs, v7, verr, dof7)
    var7 = verr**2 + (s7_hat * vobs)**2
    chi2_7 = float(np.sum(((vobs - v7)**2) / var7))
    BIC_chi7 = k7 * math.log(len(r)) + chi2_7

    if v10 is not None:
        dof10 = max(len(r) - k10, 1)
        s10_hat = implied_scatter_for_unit_chi2(vobs, v10, verr, dof10)
        var10 = verr**2 + (s10_hat * vobs)**2
        chi2_10 = float(np.sum(((vobs - v10)**2) / var10))
        delta_BIC_chi = float(k10 * math.log(len(r)) + chi2_10 - BIC_chi7)
    else:
        delta_BIC_chi = float("nan")

    # In-sample RMSE ablation
    rmse7_in = float(np.sqrt(np.mean((vobs - v7)**2)))
    rmse10_in = float(np.sqrt(np.mean((vobs - v10)**2))) if v10 is not None else float('inf')
    delta_rmse_in = rmse10_in - rmse7_in if math.isfinite(rmse10_in) else float('inf')

    # Residual diagnostics
    if do_rar:
        wres = (vobs - pred) / sig_f
        DW = durbin_watson(wres)
        rho = ar1_rho(wres)
        X = np.column_stack([np.ones_like(r), r])
        BP = breusch_pagan_pvalue(vobs - pred, X)
    else:
        DW = rho = BP = float("nan")
    resid_flag = (math.isfinite(rho) and abs(rho) > 0.3)
    conservative_sigma = bool(s_hat == 0.0 and red < 0.9)

    # Bound hits for chosen
    bound_hits_chosen = count_bound_hits(popt, chosen_lb, chosen_ub)

    # ---------------- Hold-out (optional) ----------------
    rmse_holdout = mae_holdout = r2_holdout = float("nan")
    delta_rmse_holdout = float("nan")

    Qexp = float(popt[0] * popt[6])
    delta_BIC_clipped = float(np.clip(delta_BIC_raw, -1e6, 1e6))

    return FitResult(
        ID=gal_id, n_points_used=int(len(r)),
        r_min=float(np.min(r)), r_max=float(np.max(r)), v_max=float(np.max(vobs)),
        model_chosen=chosen,
        delta_BIC=delta_BIC_clipped, delta_BIC_raw=float(delta_BIC_raw), delta_BIC_chi=float(delta_BIC_chi),
        bic_status=bic_status,
        s_frac=float(s_hat), chi2_red=float(red), R2=float(R2),
        AIC=float(AIC), AICc=float(AICc), BIC=float(BIC),
        Qexp=float(Qexp),
        ring_A=float(popt[7]) if chosen == "EGR-10" else float("nan"),
        ring_r=float(popt[8]) if chosen == "EGR-10" else float("nan"),
        ring_sigma=float(popt[9]) if chosen == "EGR-10" else float("nan"),
        DW=float(DW), rho_AR1=float(rho), BP_pvalue=float(BP),
        conservative_sigma=conservative_sigma, resid_ar1_flagged=resid_flag,
        rmse_holdout=float(rmse_holdout), mae_holdout=float(mae_holdout), r2_holdout=float(r2_holdout),
        delta_rmse_in=float(delta_rmse_in), delta_rmse_holdout=float(delta_rmse_holdout),
        bound_hits_chosen=int(bound_hits_chosen),
        bound_hits_f7=int(bound_hits_f7), bound_hits_f10=int(bound_hits_f10),
        fit_status="OK", r=r, vobs=vobs, sig_eff=sig_f, pred=pred
    )


# ================================ plotting ================================

def _format_delta_for_title(delta_raw: float, delta_clip: float) -> str:
    if not math.isfinite(delta_raw):
        return "±∞"
    return f"{delta_clip:+.1f}"

def plot_one_galaxy(res: FitResult, outdir: Path) -> Optional[Path]:
    """
    Save per-galaxy figure with observed points (σ_eff) and chosen model,
    plus whitened residual panel.
    """
    try:
        outdir.mkdir(parents=True, exist_ok=True)
        r = res.r; vobs = res.vobs; pred = res.pred; sig = res.sig_eff
        assert r is not None and vobs is not None and pred is not None and sig is not None

        fig = plt.figure(figsize=(7, 5)); ax = fig.add_subplot(111)
        ax.errorbar(r, vobs, yerr=sig, fmt="o", color="white", alpha=0.8, label="Observed")

        rs = np.linspace(0, max(1e-6, float(np.max(r))) * 1.05, 300)
        try:
            from scipy.interpolate import PchipInterpolator
            curve = PchipInterpolator(r, pred)(rs)
        except Exception:
            curve = np.interp(rs, r, pred)
        ax.plot(rs, curve, "-", lw=2, color="red", label=f"Chosen: {res.model_chosen}")

        delta_tag = _format_delta_for_title(res.delta_BIC_raw, res.delta_BIC)
        title = f"{res.ID}  s_frac={res.s_frac:.3f}, χ²_red={res.chi2_red:.3f}  (ΔBIC={delta_tag})"
        if res.conservative_sigma:
            title += " (conservative σ)"
        if res.resid_ar1_flagged:
            title += f" [AR1 ρ={res.rho_AR1:+.2f}]"
        ax.set_title(title)

        ax.set_xlabel("Radius (kpc)"); ax.set_ylabel("Velocity (km/s)")
        ax.grid(alpha=0.3); ax.legend(loc="best")

        ax2 = ax.twinx()
        wres = (vobs - pred) / sig
        ax2.plot(r, wres, ".", ms=3, alpha=0.6, color="cyan")
        ax2.axhline(0.0, color="cyan", lw=0.8, alpha=0.3)
        ax2.set_ylabel("whitened residual", color="cyan")
        ax2.tick_params(axis="y", colors="cyan")
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))

        fig.tight_layout()
        p = outdir / f"{res.ID}_rc.png"; fig.savefig(p, dpi=300); plt.close(fig)
        return p
    except Exception as e:
        print(f"[{res.ID}] Plotting failed: {e}")
        return None

# ================================ outputs ================================

SUMMARY_COLS = [
    "ID","n_points_used","r_min","r_max","v_max","model_chosen",
    "delta_BIC","delta_BIC_raw","delta_BIC_chi","bic_status",
    "s_frac","chi2_red","R2","AIC","AICc","BIC","Qexp",
    "ring_A","ring_r","ring_sigma","DW","rho_AR1","BP_pvalue",
    "conservative_sigma","resid_ar1_flagged",
    "rmse_holdout","mae_holdout","r2_holdout",
    "delta_rmse_in","delta_rmse_holdout",
    "bound_hits_chosen","bound_hits_f7","bound_hits_f10",
    "fit_status"
]

def _rotate_summary_if_schema_mismatch(path: Path, expected_cols: List[str], force_new: bool=False) -> None:
    """
    If existing summary.csv has a different header, rotate it aside.
    """
    if not path.exists():
        return
    if force_new:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path.rename(path.with_name(f"summary.{ts}.csv"))
        return
    try:
        with open(path, "r", newline="") as f:
            header = f.readline().strip()
        cols_on_disk = header.split(",")
        if cols_on_disk != expected_cols:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path.rename(path.with_name(f"summary.{ts}.csv"))
    except Exception:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path.rename(path.with_name(f"summary.{ts}.csv"))

def ensure_dirs(outdir: Path) -> Tuple[Path, Path, Path]:
    figs = outdir / "figs"; tables = outdir / "tables"
    figs.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    return outdir, figs, tables

def summarize_and_save(res: FitResult, summary_path: Path) -> None:
    """
    Append one row to summary.csv (create with header if missing).
    """
    import csv
    row = {k: getattr(res, k) for k in SUMMARY_COLS}
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        if write_header:
            w.writeheader()
        w.writerow(row)

def collect_aggregates(results: List[FitResult]) -> Dict[str, object]:
    """
    Compute global aggregates/histograms for posthoc.json.
    """
    def arr(x):
        a = np.array([getattr(r, x) for r in results], float)
        return a[np.isfinite(a)]
    hist_delta = np.histogram(arr("delta_BIC"), bins=30)
    agg = {
        "hist_delta_BIC": {"bins": hist_delta[1].tolist(), "counts": hist_delta[0].tolist()},
        "s_frac": arr("s_frac").tolist(),
        "chi2_red": arr("chi2_red").tolist(),
        "DW": arr("DW").tolist(),
        "rho_AR1": arr("rho_AR1").tolist(),
        "BP_pvalue": arr("BP_pvalue").tolist(),
        "counts": {
            "EGR10_selected": int(sum(1 for r in results if r.model_chosen == "EGR-10")),
            "failures": int(sum(1 for r in results if r.fit_status != "OK")),
            "failure_reasons": {k: int(sum(1 for r in results if r.fit_status == k))
                                for k in set(r.fit_status for r in results if r.fit_status != "OK")},
            "bic_status": {k: int(sum(1 for r in results if r.bic_status == k))
                           for k in set(r.bic_status for r in results)},
        },
    }
    hold = arr("rmse_holdout")
    if hold.size:
        hist_hold = np.histogram(hold, bins=30)
        agg["hist_rmse_holdout"] = {"bins": hist_hold[1].tolist(), "counts": hist_hold[0].tolist()}
        pairs = [(r.delta_BIC, r.rmse_holdout) for r in results
                 if math.isfinite(r.delta_BIC) and math.isfinite(r.rmse_holdout)]
        if len(pairs) >= 3:
            d = np.array([p[0] for p in pairs], float)
            h = np.array([p[1] for p in pairs], float)
            agg["corr_deltaBIC_rmse_holdout"] = float(np.corrcoef(d, h)[0, 1])
    return agg

def print_run_summary(results: List[FitResult]) -> None:
    ok = [r for r in results if r.fit_status == "OK"]
    total = len(results); sel10 = sum(1 for r in ok if r.model_chosen == "EGR-10")
    med = lambda xs: float(np.nanmedian([getattr(r, xs) for r in ok])) if ok else float("nan")
    print("\n================= RUN SUMMARY =================")
    print(f" Galaxies processed: {total}  |  OK: {len(ok)}  |  Failures: {total - len(ok)}")
    if ok:
        print(f" Selected EGR-10: {sel10}/{len(ok)}  ({(100 * sel10 / max(len(ok), 1)):.1f}%)")
    print(f" Median s_frac: {med('s_frac'):.4f}  |  Median χ²_red: {med('chi2_red'):.3f}")
    print(f" Median DW: {med('DW'):.3f}  |  Median ρ_AR1: {med('rho_AR1'):.3f}")
    hold = [r.rmse_holdout for r in ok if math.isfinite(r.rmse_holdout)]
    if hold:
        print(f" Hold-out RMSE — median: {float(np.nanmedian(hold)):.3f}  mean: {float(np.nanmean(hold)):.3f}")
    het = sum(1 for r in ok if (math.isfinite(r.BP_pvalue) and r.BP_pvalue < 0.05))
    print(f" Heteroscedastic (BP p<0.05): {het}")
    bs: Dict[str, int] = {}
    for r in ok:
        bs[r.bic_status] = bs.get(r.bic_status, 0) + 1
    if bs:
        joined = " | ".join(f"{k}:{v}" for k, v in sorted(bs.items()))
        print(f" BIC status counts: {joined}")
    print("===============================================\n")

def write_run_json(args: argparse.Namespace, outdir: Path) -> None:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": getattr(stats, "__version__", "unknown"),
            "matplotlib": plt.matplotlib.__version__,
            "pandas": pd.__version__,
        },
    }
    (outdir / "run.json").write_text(json.dumps(info, indent=2))

def failure_result(gal_id: str, used: int, reason: str) -> FitResult:
    return FitResult(
        ID=gal_id, n_points_used=used,
        r_min=float("nan"), r_max=float("nan"), v_max=float("nan"),
        model_chosen="NA",
        delta_BIC=float("nan"), delta_BIC_raw=float("nan"), delta_BIC_chi=float("nan"),
        bic_status="both_fail",
        s_frac=float("nan"), chi2_red=float("nan"), R2=float("nan"),
        AIC=float("nan"), AICc=float("nan"), BIC=float("nan"),
        Qexp=float("nan"),
        ring_A=float("nan"), ring_r=float("nan"), ring_sigma=float("nan"),
        DW=float("nan"), rho_AR1=float("nan"), BP_pvalue=float("nan"),
        conservative_sigma=False, resid_ar1_flagged=False,
        rmse_holdout=float("nan"), mae_holdout=float("nan"), r2_holdout=float("nan"),
        delta_rmse_in=float("nan"), delta_rmse_holdout=float("nan"),
        bound_hits_chosen=0, bound_hits_f7=0, bound_hits_f10=0,
        fit_status=reason,
        r=None, vobs=None, sig_eff=None, pred=None
    )


# ================================ pipeline ================================

def process_page(
    df: pd.DataFrame,
    page: int,
    do_plot: bool,
    nstarts7: int,
    nstarts10: int,
    bic_thresh: float,
    holdout: float,
    do_rar: bool,
    seed: Optional[int],
    outdir: Path,
    holdout_repeats: int,
    holdout_mode: str,
    fresh: bool,
    f7_widen: float,
    f7_tries: int,
    nested_f7: bool,
) -> None:
    outdir, figs_dir, tables_dir = ensure_dirs(outdir)
    summary_path = tables_dir / "summary.csv"
    _rotate_summary_if_schema_mismatch(summary_path, SUMMARY_COLS, force_new=fresh)

    print(f"numpy {np.__version__} | scipy {getattr(stats, '__version__', 'unknown')} | "
          f"matplotlib {plt.matplotlib.__version__} | pandas {pd.__version__}")

    model = EGRRotationCurveModel(alpha_c=0.25)
    rng = np.random.default_rng(seed)

    # read run.json to pick up flags (hm/elm/mcmc)
    try:
        run_info = json.loads((outdir / "run.json").read_text())
        args_map = run_info.get("args", {})
        hm_eps        = float(args_map.get("hm_eps", 0.0))
        elm_enable    = bool(args_map.get("elm_enable", False))
        elm_mode      = str(args_map.get("elm_mode", "bounded"))
        elm_alpha     = float(args_map.get("elm_alpha", 0.35))
        elm_beta      = float(args_map.get("elm_beta", 0.35))
        elm_gates_s   = str(args_map.get("elm_gates", "")).strip()
        elm_phase_scale = float(args_map.get("elm_phase_scale", 1.0))
        elm_scale       = float(args_map.get("elm_scale", 1.0))

        want_mcmc    = bool(args_map.get("mcmc", False))
        mcmc_walkers = int(args_map.get("walkers", 64))
        mcmc_steps   = int(args_map.get("steps", 1500))
        mcmc_burn    = int(args_map.get("burn", 300))
    except Exception:
        hm_eps = 0.0
        elm_enable, elm_mode, elm_alpha, elm_beta = False, "bounded", 0.35, 0.35
        elm_gates_s, elm_phase_scale, elm_scale = "", 1.0, 1.0
        want_mcmc, mcmc_walkers, mcmc_steps, mcmc_burn = False, 64, 1500, 300

    # Optional Entropy Ledger Matrix dump (diagnostics only)
    if elm_enable:
        try:
            elm_dir = outdir / "elm"
            elm_dir.mkdir(parents=True, exist_ok=True)
            elm = EntropyLedgerMatrix(mode=elm_mode, alpha=elm_alpha, beta=elm_beta)
            bank = elm.bank()  # base matrices

            # Utility to save arrays and quick previews
            def _save(name: str, M: np.ndarray) -> None:
                np.save(elm_dir / f"{name}.npy", M)
                try:
                    plt.figure(figsize=(3,3))
                    plt.imshow(M.real, cmap="gray", interpolation="nearest")
                    plt.title(f"ELM {name} (real)")
                    plt.axis("off"); plt.tight_layout()
                    plt.savefig(elm_dir / f"{name}_real.png", dpi=200); plt.close()

                    plt.figure(figsize=(3,3))
                    plt.imshow(M.imag, cmap="gray", interpolation="nearest")
                    plt.title(f"ELM {name} (imag)")
                    plt.axis("off"); plt.tight_layout()
                    plt.savefig(elm_dir / f"{name}_imag.png", dpi=200); plt.close()
                except Exception:
                    pass

            # Save base matrices
            for name, M in bank.items():
                _save(name, M)

            # Apply optional quaternion gate pipeline
            gates = [g.strip().upper() for g in elm_gates_s.split(",") if g.strip()]
            if gates:
                for base_name, M0 in bank.items():
                    M = M0.copy()
                    for g in gates:
                        if g == "AND":
                            M = elm.transform_and(M, phase_scale=elm_phase_scale)
                        elif g == "XOR":
                            M = elm.transform_xor(M, scale=elm_scale)
                        elif g == "NOT":
                            M = elm.transform_not(M, scale=elm_scale)
                        else:
                            print(f"[ELM] Ignoring unknown gate '{g}'")
                    gtag = "_".join(gates).lower()
                    _save(f"{base_name}__{gtag}", M)

            print(f"[ELM] Generated: mode={elm_mode} alpha={elm_alpha} beta={elm_beta} "
                  f"gates='{elm_gates_s or '(none)'}' → {elm_dir}")
        except Exception as e:
            print(f"[ELM] Generation failed: {e}")

    ids = group_ids(df); per_page = 4; total = len(ids); pages = (total + per_page - 1) // per_page
    s = page * per_page; e = min(s + per_page, total)
    print(f"\nAnalyzing Page {page}  —  galaxies {s+1}..{e} of {total} (Total Pages: {pages})")

    all_results: List[FitResult] = []
    hm_ids: List[str] = []
    hm_dS: List[float] = []
    hm_tv0: List[float] = []
    hm_tv1: List[float] = []

    for gid in ids[s:e]:
        sub = df[df["ID"] == gid] if "ID" in df.columns else df[df[COLUMN_ALIASES["ID"][0]] == gid]
        need_cols = ["ID","R","Vobs","e_Vobs"]
        sub2 = sub[[c for c in need_cols if c in sub.columns]].copy()
        print(f"\n[{gid}] rows(raw)={len(sub)}", end="")
        m = np.isfinite(sub2.get("R", np.array([]))) & np.isfinite(sub2.get("Vobs", np.array([]))) & np.isfinite(sub2.get("e_Vobs", np.array([])))
        used = int(np.sum(m)) if m.size else 0
        print(f", rows(used)={used}")
        if used:
            print(f"  R range: [{np.nanmin(sub2['R']):.3g}, {np.nanmax(sub2['R']):.3g}]  |  "
                  f"V range: [min {np.nanmin(sub2['Vobs']):.3g}, max {np.nanmax(sub2['Vobs']):.3g}]")

        try:
            res = fit_one_galaxy(
                gid, sub2, model, rng,
                nstarts7=nstarts7, nstarts10=nstarts10,
                bic_thresh=bic_thresh, holdout_frac=holdout, do_rar=do_rar,
                holdout_repeats=holdout_repeats, holdout_mode=holdout_mode,
                f7_widen=f7_widen, f7_tries=f7_tries, nested_f7=nested_f7
            )
        except Exception as e:
            print(f"  [ERROR] Fit failed with exception: {e}")
            res = failure_result(gid, used, "exception")

        if res.fit_status == "OK":
            delta_str = "±inf" if not math.isfinite(res.delta_BIC_raw) else f"{res.delta_BIC:+.1f}"
            extra = f" ({res.bic_status})" if res.bic_status != "both_finite" else ""
            chi_tag = f", ΔBIC_χ²={res.delta_BIC_chi:+.1f}" if math.isfinite(res.delta_BIC_chi) else ""
            print(f"  model: {res.model_chosen}  |  ΔBIC={delta_str}{extra}{chi_tag}")
            cons = "  [conservative σ]" if res.conservative_sigma else ""
            bh = f"  [bound hits — chosen:{res.bound_hits_chosen}, f7:{res.bound_hits_f7}, f10:{res.bound_hits_f10}]"
            print(f"  s_frac={res.s_frac:.4f}  χ²_red={res.chi2_red:.3f}  R²={res.R2:.3f}  "
                  f"AIC={res.AIC:.2f}  AICc={res.AICc:.2f}  BIC={res.BIC:.2f}{cons}{bh}")
            if do_rar:
                warn = "  [warn residual AR1]" if res.resid_ar1_flagged else ""
                het = "  [hetero p<0.05]" if (math.isfinite(res.BP_pvalue) and res.BP_pvalue < 0.05) else ""
                print(f"  DW={res.DW:.3f}  ρ_AR1={res.rho_AR1:.3f}{warn}  BP_p={res.BP_pvalue:.3f}{het}")
            if math.isfinite(res.rmse_holdout):
                print(f"  hold-out: RMSE={res.rmse_holdout:.3f}  MAE={res.mae_holdout:.3f}  R²={res.r2_holdout:.3f}")
            if res.model_chosen == "EGR-10":
                if math.isfinite(res.delta_rmse_in):
                    print(f"  ablation: ΔRMSE_in = {res.delta_rmse_in:+.3f} (model − EGR-7)")
                if math.isfinite(res.delta_rmse_holdout):
                    print(f"            ΔRMSE_holdout = {res.delta_rmse_holdout:+.3f}")
            print(f"  Q_exp = R * S_resp = {res.Qexp:.6g}")

            # HM diagnostics
            if hm_eps > 0.0:
                wres = (res.vobs - res.pred) / res.sig_eff
                M = _residuals_to_matrix(wres)
                p0 = _hm_prob_from_matrix(M)
                S0 = _hm_entropy_shannon(p0)
                U  = _hm_uniform(M.shape)
                tv0 = 0.5 * float(np.abs(p0 - U).sum())
                M1 = _hm_apply_matrix(M, hm_eps)
                p1 = _hm_prob_from_matrix(M1)
                S1 = _hm_entropy_shannon(p1)
                tv1 = 0.5 * float(np.abs(p1 - U).sum())
                dS = S1 - S0
                print(f"  HM(ε={hm_eps:.2f})  ΔH={dS:+.4f}  TV: {tv0:.4f}→{tv1:.4f}")
                hm_ids.append(gid); hm_dS.append(float(dS)); hm_tv0.append(float(tv0)); hm_tv1.append(float(tv1))

            # ---------------- MCMC diagnostics (optional) ----------------
            if want_mcmc and HAS_EMCEE and res.r is not None and len(res.r) >= 8:
                # Build bounds for the chosen family
                seeds, _, _ = data_driven_seeds(res.r, res.vobs)
                if res.model_chosen == "EGR-10":
                    _, p0_m, lb_m, ub_m = bounds_egr10(res.r, seeds)
                else:
                    _, p0_m, lb_m, ub_m = bounds_egr7(res.r, seeds)

                # Fit once with sigma=res.sig_eff to get a good starting point
                f = model.velocity
                theta0, _, _ = multiple_random_starts(
                    f, res.r, res.vobs, res.sig_eff, p0_m, lb_m, ub_m,
                    n_starts=max(6, nstarts7 if res.model_chosen == "EGR-7" else nstarts10),
                    rng=rng
                )
                if theta0 is None:
                    theta0 = p0_m.copy()

                # Run MCMC (full parameter vector; ring params are fixed for EGR-7 via bounds)
                sampler, chain = mcmc_rc(
                    model=model,
                    r=res.r, vobs=res.vobs, sig_eff=res.sig_eff,
                    theta0=np.asarray(theta0, float),
                    lb=np.asarray(lb_m, float), ub=np.asarray(ub_m, float),
                    walkers=mcmc_walkers, steps=mcmc_steps, burn=mcmc_burn,
                    seed=seed
                )

                if chain is not None and chain.size:
                    # Parameter names (full vector; ring dims are constant for EGR-7)
                    names = ["R","phi_w","A_w","A_p","k_p","sigma_p","S_resp","A_ring","r_ring","sigma_ring"]
                    names = names[:chain.shape[1]]

                    # Corner
                    try:
                        fig = corner.corner(chain, labels=names, quantiles=[0.16,0.5,0.84],
                                            show_titles=True, title_fmt=".3f")
                        fig.suptitle(f"MCMC — {res.ID} ({res.model_chosen})", fontsize=12)
                        fig.savefig(figs_dir / f"{res.ID}_corner.png", dpi=180)
                        plt.close(fig)
                    except Exception:
                        pass

                    # Corr heatmap
                    C = rc_corr_heatmap(figs_dir / f"{res.ID}_corr.png", chain, names)

                    # PPC band on r grid
                    q16, q50, q84 = ppc_band_from_chain(chain, model, res.r, n_draws=200)
                    if q16.size and do_plot:
                        plt.figure(figsize=(7,5))
                        plt.errorbar(res.r, res.vobs, yerr=res.sig_eff, fmt='.', color='white', alpha=0.7, label="data")
                        plt.fill_between(res.r, q16, q84, alpha=0.25, label="PPC 68% band")
                        plt.plot(res.r, q50, '-', label="PPC median")
                        plt.xlabel("Radius (kpc)"); plt.ylabel("Velocity (km/s)")
                        plt.title(f"PPC — {res.ID} ({res.model_chosen})")
                        plt.grid(alpha=0.3); plt.legend()
                        plt.tight_layout(); plt.savefig(figs_dir / f"{res.ID}_ppc.png", dpi=180); plt.close()

                    # Append light MCMC metadata to posthoc
                    try:
                        ph_path = tables_dir / "posthoc.json"
                        ph = json.loads(ph_path.read_text()) if ph_path.exists() else {}
                        ph.setdefault("mcmc", {})[res.ID] = {
                            "names": names,
                            "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)) if sampler is not None else float("nan"),
                            "n_samples": int(chain.shape[0]),
                            "corr": (C.tolist() if C is not None else None)
                        }
                        ph_path.write_text(json.dumps(ph, indent=2))
                    except Exception:
                        pass

        else:
            print(f"  [skip] status={res.fit_status}")

        if do_plot and res.fit_status == "OK":
            p = plot_one_galaxy(res, outdir=figs_dir)
            if p is not None:
                print(f"  saved fig: {p}")
        summarize_and_save(res, summary_path)
        all_results.append(res)

    agg = collect_aggregates(all_results)
    if hm_eps > 0.0 and hm_dS:
        agg["HM_diag"] = {
            "eps": hm_eps,
            "ids": hm_ids,
            "delta_H": hm_dS,
            "tv0": hm_tv0,
            "tv1": hm_tv1,
            "median_delta_H": float(np.median(hm_dS)),
            "median_tv0": float(np.median(hm_tv0)),
            "median_tv1": float(np.median(hm_tv1)),
        }
    (tables_dir / "posthoc.json").write_text(json.dumps(agg, indent=2))
    print_run_summary(all_results)


# ============================== self-test ==============================

def synthetic_rc(
    model: EGRRotationCurveModel,
    rng: np.random.Generator,
    n_points: int = 20,
    params7: Tuple[float, float, float, float, float, float, float] = (1.2, 3.0, 120.0, 80.0, 2.5, 12.0, 4.5),
    ring: Optional[Tuple[float, float, float]] = None,
    s_meas: float = 5.0,
    s_frac: float = 0.06
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a synthetic galaxy RC with known parameters and noise.
    """
    r = np.linspace(0.5, 15.0, n_points)
    if ring is None:
        p = (*params7, 0.0, 0.0, 1.0)
    else:
        p = (*params7, *ring)
    v_true = model.velocity(r, *p)
    v_scatter = rng.normal(0.0, s_frac * v_true)
    vobs = v_true + v_scatter + rng.normal(0.0, s_meas, size=n_points)
    verr = np.full_like(vobs, s_meas)
    return r, vobs, verr

def run_selftest(seed: int, outdir: Path) -> None:
    print("\nRunning self-test (synthetic harness) ...")
    rng = np.random.default_rng(seed)
    model = EGRRotationCurveModel()
    rA, vA, eA = synthetic_rc(model, rng, ring=None)
    rB, vB, eB = synthetic_rc(model, rng, ring=(60.0, 6.0, 1.8))
    df = pd.DataFrame({"ID": ["SYN7"]*len(rA) + ["SYN10"]*len(rB),
                       "R": np.concatenate([rA, rB]),
                       "Vobs": np.concatenate([vA, vB]),
                       "e_Vobs": np.concatenate([eA, eB])})
    process_page(df, page=0, do_plot=False, nstarts7=12, nstarts10=12, bic_thresh=10.0, holdout=0.2,
                 do_rar=True, seed=seed, outdir=outdir, holdout_repeats=3, holdout_mode="random",
                 fresh=False, f7_widen=3.5, f7_tries=16, nested_f7=True)
    print("Self-test complete.\n")

# ======================== HM diagnostics (optional) ========================

def _hm_prob_from_matrix(M: np.ndarray) -> np.ndarray:
    magsq = np.abs(M)**2
    s = float(magsq.sum())
    if s <= 0:
        return np.full(M.size, 1.0/M.size)
    return (magsq / s).ravel()

def _hm_uniform(shape: Tuple[int,int]) -> np.ndarray:
    N = int(shape[0]*shape[1])
    return np.full(N, 1.0/N, dtype=float)

def _hm_entropy_shannon(p: np.ndarray) -> float:
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def _hm_apply_matrix(M: np.ndarray, eps: float) -> np.ndarray:
    """
    Doubly-stochastic mix toward uniform on |M|^2; preserve Frobenius norm and phases.
    p' = (1-ε)p + εu.
    """
    X = np.asarray(M, dtype=complex)
    n0, n1 = X.shape
    p  = _hm_prob_from_matrix(X)
    u  = _hm_uniform(X.shape)
    eps = float(np.clip(eps, 0.0, 1.0))
    p2 = (1.0 - eps) * p + eps * u

    total = float((np.abs(X)**2).sum())
    mag2  = np.sqrt(p2 * total).reshape(n0, n1)

    phase = np.ones_like(X, dtype=complex)
    nz = (np.abs(X) > 0)
    phase[nz] = X[nz] / np.abs(X[nz])
    return mag2 * phase

def _residuals_to_matrix(wres: np.ndarray) -> np.ndarray:
    """
    Map a 1D whitened residual series into an 8x8 complex matrix whose magnitudes
    encode |wres| (row-major). Phases set to 1 so HM acts only on the shape.
    """
    w = np.asarray(wres, float)
    m = np.abs(w)
    # interpolate/resample to 64 samples
    if m.size == 0:
        vec = np.zeros(64, float)
    elif m.size == 64:
        vec = m.copy()
    else:
        xs = np.linspace(0.0, 1.0, num=m.size)
        xi = np.linspace(0.0, 1.0, num=64)
        vec = np.interp(xi, xs, m)
    M = vec.reshape(8, 8)
    return M.astype(complex)

def rc_corr_heatmap(path: Path, chain: np.ndarray, names: List[str]) -> Optional[np.ndarray]:
    if chain is None or chain.size == 0: 
        return None
    C = np.corrcoef(chain, rowvar=False)
    plt.figure(figsize=(6,5))
    im = plt.imshow(C, vmin=-1, vmax=1, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.title("Posterior correlation")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return C

def mcmc_rc(model: "EGRRotationCurveModel",
            r: np.ndarray, vobs: np.ndarray, sig_eff: np.ndarray,
            theta0: np.ndarray, lb: np.ndarray, ub: np.ndarray,
            walkers: int, steps: int, burn: int, seed: Optional[int]) -> Tuple[Optional["emcee.EnsembleSampler"], Optional[np.ndarray]]:
    """MCMC on the chosen model parameters (Gaussian errors, sigma = sig_eff)."""
    if not HAS_EMCEE:
        return None, None
    rng = np.random.default_rng(seed)
    ndim = len(theta0)

    def log_prior(theta):
        for v, l, u in zip(theta, lb, ub):
            if (v < l) or (v > u):
                return -np.inf
        return 0.0

    def log_like(theta):
        try:
            m = model.velocity(r, *theta)
        except Exception:
            return -np.inf
        z = (vobs - m) / sig_eff
        return -0.5 * float(np.dot(z, z))

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_like(theta)
        return lp + ll

    # small Gaussian jitter around the MAP start
    pos = theta0 + 1e-3 * np.abs(theta0 + 1.0) * rng.standard_normal((walkers, ndim))
    # clip inside bounds
    for i in range(walkers):
        for j in range(ndim):
            pos[i, j] = np.clip(pos[i, j], lb[j] + 1e-12, ub[j] - 1e-12)

    sampler = emcee.EnsembleSampler(walkers, ndim, log_prob)
    sampler.run_mcmc(pos, steps, progress=False)
    chain = sampler.get_chain(discard=burn, flat=True)
    return sampler, chain

def ppc_band_from_chain(chain: np.ndarray, model: "EGRRotationCurveModel",
                        r: np.ndarray, n_draws: int = 200, seed: int = 123) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """16/50/84% posterior predictive band for the rotation curve."""
    rng = np.random.default_rng(seed)
    if chain is None or chain.size == 0:
        return np.array([]), np.array([]), np.array([])
    idx = rng.integers(0, chain.shape[0], size=min(n_draws, chain.shape[0]))
    preds = np.array([model.velocity(r, *chain[i]) for i in idx])
    q16 = np.percentile(preds, 16, axis=0)
    q50 = np.percentile(preds, 50, axis=0)
    q84 = np.percentile(preds, 84, axis=0)
    return q16, q50, q84

# ================================= CLI =================================

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EGR rotation-curve fitter with robust diagnostics")
    ap.add_argument("page", nargs="?", default="0", help="Page index (4 galaxies/page), default 0")
    ap.add_argument("--data", default="SPARC_MassModels.csv", help="Path to SPARC-like CSV file")
    ap.add_argument("--plot", action="store_true", help="Save per-galaxy figures")
    ap.add_argument("--starts", type=int, default=12, help="Random restarts for both models (if --starts7/--starts10 unset)")
    ap.add_argument("--starts7", type=int, default=None, help="Random restarts for EGR-7 (overrides --starts)")
    ap.add_argument("--starts10", type=int, default=None, help="Random restarts for EGR-10 (overrides --starts)")
    ap.add_argument("--bic-thresh", type=float, default=10.0, help="ΔBIC threshold to keep EGR-10 (default 10.0)")
    ap.add_argument("--holdout", type=float, default=0.0, help="Hold-out fraction in (0,1), e.g., 0.2")
    ap.add_argument("--holdout-repeats", type=int, default=3, help="Random hold-out repeats (ignored for --holdout-mode outer)")
    ap.add_argument("--holdout-mode", choices=["random","outer"], default="random",
                    help="Hold-out strategy: random masks or outer radii")
    ap.add_argument("--rar", action="store_true", help="Enable residual autocorrelation & Breusch–Pagan tests")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    ap.add_argument("--outdir", default="out", help="Output directory root (default 'out/')")
    ap.add_argument("--fresh", action="store_true", help="Rotate existing summary.csv to summary.<timestamp>.csv before run")
    ap.add_argument("--f7-widen", type=float, default=3.5, help="Coarse EGR-7 fallback widen factor")
    ap.add_argument("--f7-tries", type=int, default=16, help="Coarse EGR-7 fallback random tries")
    ap.add_argument("--nested-f7", action="store_true", help="Run ring-pinned EGR-10 'nested' baseline for EGR-7 fairness")

    # HM diagnostics (optional)
    ap.add_argument("--hm-eps", type=float, default=0.0,
                    help="Optional HM residual diagnostic ε∈[0,1]; 0=off (no effect on fit).")

    # Entropy Ledger Matrix (optional; diagnostics only, never used in fitting)
    ap.add_argument("--elm-enable", action="store_true",
                    help="Generate Entropy Ledger Matrices (diagnostics only).")
    ap.add_argument("--elm-mode", choices=["flat", "bounded"], default="bounded",
                    help="ELM weighting mode (flat=raw pattern, bounded=RDC×PAC squash).")
    ap.add_argument("--elm-alpha", type=float, default=0.35,
                    help="ELM Radial Decay Component (RDC) blend strength.")
    ap.add_argument("--elm-beta", type=float, default=0.35,
                    help="ELM Phase Accumulation Component (PAC) blend strength.")

    # Quaternion gate pipeline for ELM (comma list: AND,XOR,NOT), optional scales
    ap.add_argument("--elm-gates", default="",
                    help="Comma-separated gate list applied in order to each ELM matrix. "
                         "Options: AND, XOR, NOT. Example: AND,XOR")
    ap.add_argument("--elm-phase-scale", type=float, default=1.0,
                    help="Phase scale for AND gate (multiplies exp(i π/φ * phase_scale)).")
    ap.add_argument("--elm-scale", type=float, default=1.0,
                    help="Scalar scale for XOR/NOT gates (multiplies π/φ).")

    # restore selftest to match main()
    ap.add_argument("--selftest", action="store_true",
                    help="Run synthetic self-test instead of reading CSV")
    
        # MCMC (optional, per-galaxy on the chosen model)
    ap.add_argument("--mcmc", action="store_true",
                    help="Run emcee on the chosen model per galaxy (diagnostics only).")
    ap.add_argument("--walkers", type=int, default=64, help="Number of walkers for MCMC.")
    ap.add_argument("--steps", type=int, default=1500, help="Total steps per walker.")
    ap.add_argument("--burn", type=int, default=300, help="Burn-in to discard from chains.")

    return ap



def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    try:
        page = int(args.page)
    except Exception:
        print(f"Could not parse page '{args.page}', using 0."); page = 0
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    write_run_json(args, outdir)

    if args.selftest:
        run_selftest(seed=args.seed or 1234, outdir=outdir)
        return

    try:
        df, mapping = load_data(args.data)
    except Exception as e:
        print(f"Failed to read '{args.data}': {e}"); sys.exit(1)
    print("Resolved columns: " + ", ".join(f"{k}->{v}" for k, v in mapping.items()))
    if not {"ID","R","Vobs","e_Vobs"}.issubset(set(df.columns)):
        missing = {"ID","R","Vobs","e_Vobs"} - set(df.columns)
        print(f"Missing required columns after alias resolution: {missing}"); sys.exit(2)

    n7 = args.starts7 if args.starts7 is not None else args.starts
    n10 = args.starts10 if args.starts10 is not None else args.starts

    process_page(df, page=page, do_plot=args.plot,
                 nstarts7=n7, nstarts10=n10, bic_thresh=args.bic_thresh,
                 holdout=args.holdout, do_rar=args.rar, seed=args.seed, outdir=outdir,
                 holdout_repeats=args.holdout_repeats, holdout_mode=args.holdout_mode,
                 fresh=args.fresh, f7_widen=args.f7_widen, f7_tries=args.f7_tries,
                 nested_f7=args.nested_f7)

if __name__ == "__main__":
    main()
