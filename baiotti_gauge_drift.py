#!/usr/bin/env python3
"""
Quadratic gauge-drift removal for gravitational-wave strain.

This procedure is necessary before gravitational-wave analysis. Double
integration of Psi4 (or any equivalent recovery of h from curvature) amplifies
low-frequency gauge content, so the raw strain typically carries a slow
quadratic floor. That floor biases the GW phase phi = arctan(h_x / h_+) and
its derivative, and therefore f_GW(t), merger-time f_max, and related
diagnostics. Subtract the floor first, then analyse h_+' and h_x'.

Following Baiotti, Bernuzzi, Corvino, De Pietri & Nagar, Phys. Rev. D 79,
024002 (2009), Eq. (38) and Figs. 10--11:

    r h^{lm}(t) = r tilde{h}^{lm}(t, r) + Q_0 + Q_1 t + Q_2(r) t^2

Operationally a least-squares quadratic P(t) = Q_0 + Q_1 t + Q_2 t^2 is fitted
to each polarization and subtracted so the corrected strain oscillates about
zero:

    h_+' = h_+ - P_+(t),   h_x' = h_x - P_x(t)

doi:10.1103/PhysRevD.79.024002
arXiv:0808.4002
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fit window used for the binary quark-star analysis (geometric units, M_sun).
# Public callers should set t_fit_min / t_fit_max for their own data; these
# constants are the thesis defaults, not a universal prescription.
THESIS_FIT_T_MIN_M = 0.0
THESIS_FIT_T_MAX_M = 3000.0


@dataclass(frozen=True)
class BaiottiEq38Floor:
    """Polynomial floor P(t) = Q0 + Q1 t + Q2 t^2 (Baiotti et al. 2009, Eq. 38)."""

    Q0: float
    Q1: float
    Q2: float

    def drift(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return self.Q0 + self.Q1 * t + self.Q2 * (t**2)


@dataclass(frozen=True)
class BaiottiEq38Drift:
    """Eq. (38) floors for r(h_+ - i h_x) at extraction radius r."""

    h_plus: BaiottiEq38Floor
    h_cross: BaiottiEq38Floor
    r_extract: float
    method: str
    t_fit_min: float | None = None
    t_fit_max: float | None = None


def _fit_window_mask(
    t: np.ndarray,
    y: np.ndarray,
    *,
    t_fit_min: float | None,
    t_fit_max: float | None,
) -> np.ndarray:
    mask = np.isfinite(t) & np.isfinite(y)
    if t_fit_min is not None:
        mask &= t >= float(t_fit_min)
    if t_fit_max is not None:
        mask &= t <= float(t_fit_max)
    return mask


def baiotti_eq38_floor(
    t: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "fit",
    t0: float | None = None,
    t_fit_min: float | None = None,
    t_fit_max: float | None = None,
) -> BaiottiEq38Floor:
    """
    Coefficients for P(t) = Q0 + Q1 t + Q2 t^2.

    method ``fit`` (default): least-squares quadratic floor
    (Baiotti et al. 2009, Figs. 10--11). This is the procedure that must be
    applied before gravitational-wave phase and frequency analysis.

    method ``initial``: Q0 = y(t0), Q1 = dy/dt at t0 (their Eq. 36), Q2 from
    least squares on the residual.

    ``t_fit_min`` / ``t_fit_max`` restrict where Q_i are fitted; P(t) is still
    evaluated on the full series.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = _fit_window_mask(t, y, t_fit_min=t_fit_min, t_fit_max=t_fit_max)
    if np.count_nonzero(mask) < 3:
        return BaiottiEq38Floor(0.0, 0.0, 0.0)

    if method == "fit":
        tm = t[mask]
        design = np.column_stack([np.ones(tm.size), tm, tm**2])
        q0, q1, q2 = np.linalg.lstsq(design, y[mask], rcond=None)[0]
        return BaiottiEq38Floor(Q0=float(q0), Q1=float(q1), Q2=float(q2))

    if method == "initial":
        if t0 is not None:
            t_ref = float(t0)
        elif t_fit_min is not None:
            t_ref = float(t_fit_min)
        else:
            t_ref = float(t[mask][0])
        i0 = int(np.argmin(np.abs(t - t_ref)))
        q0 = float(y[i0])
        q1 = float(np.gradient(y, t)[i0])
        resid = y - q0 - q1 * t
        tm = t[mask]
        q2, *_ = np.linalg.lstsq(
            np.column_stack([tm**2]),
            resid[mask],
            rcond=None,
        )
        return BaiottiEq38Floor(Q0=q0, Q1=q1, Q2=float(q2[0]))

    raise ValueError(f"unknown Baiotti Eq. (38) method: {method!r}")


def quadratic_floor(
    t: np.ndarray,
    y: np.ndarray,
    *,
    t_fit_min: float | None = None,
    t_fit_max: float | None = None,
) -> np.ndarray:
    """Least-squares P(t) fitted to a real strain polarization."""
    floor = baiotti_eq38_floor(
        t,
        np.asarray(y, dtype=float),
        method="fit",
        t_fit_min=t_fit_min,
        t_fit_max=t_fit_max,
    )
    return floor.drift(t)


def detrend_hp_hx_quadratic(
    t: np.ndarray,
    hp: np.ndarray,
    hx: np.ndarray,
    *,
    t_fit_min: float | None = None,
    t_fit_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove quadratic gauge drift from both polarizations.

    This step is necessary before gravitational-wave analysis: use h_+' and
    h_x' for phase, instantaneous frequency, and peak-frequency measurements.

        h_+' = h_+ - P_+(t),  h_x' = h_x - P_x(t)

    with separate least-squares quadratics (Baiotti et al. 2009, Eq. 38).
    """
    t = np.asarray(t, dtype=float)
    hp = np.asarray(hp, dtype=float)
    hx = np.asarray(hx, dtype=float)
    p_plus = quadratic_floor(t, hp, t_fit_min=t_fit_min, t_fit_max=t_fit_max)
    p_cross = quadratic_floor(t, hx, t_fit_min=t_fit_min, t_fit_max=t_fit_max)
    return hp - p_plus, hx - p_cross


def detrend_rhp_rhc_quadratic(
    t: np.ndarray,
    rhp: np.ndarray,
    rhc: np.ndarray,
    r_extract: float,
    *,
    t_fit_min: float | None = None,
    t_fit_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Detrend dimensionless h_+, h_x; return r h_+' and r h_x'."""
    r = float(r_extract)
    hp, hx = detrend_hp_hx_quadratic(
        t,
        np.asarray(rhp, dtype=float) / r,
        np.asarray(rhc, dtype=float) / r,
        t_fit_min=t_fit_min,
        t_fit_max=t_fit_max,
    )
    return r * hp, r * hx


def fit_subtract_baiotti_eq38_rh(
    t: np.ndarray,
    rh_raw: np.ndarray,
    r_extract: float,
    *,
    method: str = "fit",
    t_fit_min: float | None = None,
    t_fit_max: float | None = None,
) -> tuple[np.ndarray, BaiottiEq38Drift]:
    """
    Apply Baiotti et al. (2009) Eq. (38) to complex strain r(h_+ - i h_x).

    ``rh_raw`` is the double-integrated (FFI) multipole r tilde{h}. Separate
    quadratic floors are subtracted from r h_+ and r h_x so both polarizations
    oscillate about zero. Do this before any gravitational-wave analysis of
    the strain.
    """
    t = np.asarray(t, dtype=float)
    rh_raw = np.asarray(rh_raw, dtype=np.complex128)
    r = float(r_extract)

    fit_kw = dict(
        method=method,
        t0=t_fit_min if t_fit_min is not None else t[0],
        t_fit_min=t_fit_min,
        t_fit_max=t_fit_max,
    )
    floor_plus = baiotti_eq38_floor(t, rh_raw.real, **fit_kw)
    floor_cross = baiotti_eq38_floor(t, -rh_raw.imag, **fit_kw)

    re_corr = rh_raw.real - floor_plus.drift(t)
    im_corr = -((-rh_raw.imag) - floor_cross.drift(t))
    rh = re_corr + 1j * im_corr

    coeff = BaiottiEq38Drift(
        h_plus=floor_plus,
        h_cross=floor_cross,
        r_extract=r,
        method=method,
        t_fit_min=t_fit_min,
        t_fit_max=t_fit_max,
    )
    return rh, coeff


def _synthetic_demo() -> None:
    """Recover a known quadratic floor from a mock (h_+, h_x) pair."""
    t = np.linspace(0.0, 3000.0, 4001)
    hp_phys = 0.02 * np.sin(2.0 * np.pi * t / 80.0)
    hx_phys = 0.02 * np.cos(2.0 * np.pi * t / 80.0)
    p_plus = -1.5e-3 + 2.0e-6 * t + 4.0e-10 * (t**2)
    p_cross = 8.0e-4 - 1.1e-6 * t + 2.5e-10 * (t**2)
    hp_p, hx_p = detrend_hp_hx_quadratic(t, hp_phys + p_plus, hx_phys + p_cross)
    print("Baiotti et al. (2009) quadratic gauge-drift removal")
    print("This procedure is necessary before gravitational-wave analysis.")
    print(f"  mean(h_+') = {hp_p.mean():.3e}")
    print(f"  mean(h_x') = {hx_p.mean():.3e}")
    print(f"  max |h_+' - h_+^phys| = {np.max(np.abs(hp_p - hp_phys)):.3e}")
    print(f"  max |h_x' - h_x^phys| = {np.max(np.abs(hx_p - hx_phys)):.3e}")


if __name__ == "__main__":
    _synthetic_demo()
