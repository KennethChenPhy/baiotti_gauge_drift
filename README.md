# baiotti_gauge_drift

Quadratic gauge-drift removal for gravitational-wave strain, following [Baiotti, Bernuzzi, Corvino, De Pietri & Nagar, Phys. Rev. D **79**, 024002 (2009)](https://doi.org/10.1103/PhysRevD.79.024002).

**This procedure is necessary before gravitational-wave analysis.** Recovering \(h_+\) and \(h_\times\) from \(\Psi_4\) requires two time integrations. That double integral amplifies low-frequency gauge content, so the raw strain \(\tilde h\) typically sits on a slow quadratic floor. The floor contaminates the GW phase \(\phi=\arctan(h_\times/h_+)\) and \(\mathrm{d}\phi/\mathrm{d}t\). Instantaneous frequency \(f_{\rm GW}(t)\), merger-time \(f_{\rm max}\), and any diagnostic built from the phase are unreliable until the floor is subtracted.

Baiotti et al. remove the artefact at fixed extraction radius by a quadratic polynomial in coordinate time [their Eq. (38)]:

\[
r\,h^{\ell m}(t)
= r\,\tilde h^{\ell m}(t,r) + Q_0 + Q_1 t + Q_2(r)\,t^2.
\]

This code implements the operational step used in their Figs. 10–11: fit \(P(t)=Q_0+Q_1 t+Q_2 t^2\) by least squares on each polarization and subtract it so the corrected strain oscillates about zero,

\[
h_+'=h_+-P_+(t),\qquad h_\times'=h_\times-P_\times(t).
\]

Use \(h_+'\) and \(h_\times'\) for all subsequent gravitational-wave analysis.

## Requirements

- Python 3.10+
- NumPy

```bash
pip install -r requirements.txt
```

## Usage

After you have strain at a fixed extraction radius (for example from fixed-frequency double integration of \(\Psi_{4,22}\)):

```python
from baiotti_gauge_drift import detrend_hp_hx_quadratic

# h_plus, h_cross: raw polarizations after Psi4 → h
hp_prime, hx_prime = detrend_hp_hx_quadratic(t, h_plus, h_cross)

# necessary before phase / frequency analysis
# phi = arctan(hx_prime / hp_prime)
# f_GW = (1 / 2π) d phi / dt
```

Complex multipole \(r(h_+-i h_\times)\):

```python
from baiotti_gauge_drift import fit_subtract_baiotti_eq38_rh

rh_corr, coeff = fit_subtract_baiotti_eq38_rh(t, rh_raw, r_extract)
```

Optional fit window (geometric units \(M_\odot\)). The binary quark-star analysis used \(0\le t\le 3000\,M_\odot\); choose a window appropriate to your run:

```python
from baiotti_gauge_drift import THESIS_FIT_T_MIN_M, THESIS_FIT_T_MAX_M

hp_prime, hx_prime = detrend_hp_hx_quadratic(
    t, h_plus, h_cross,
    t_fit_min=THESIS_FIT_T_MIN_M,
    t_fit_max=THESIS_FIT_T_MAX_M,
)
```

Synthetic check (no simulation data required):

```bash
python baiotti_gauge_drift.py
```

## Files

| File | Role |
|------|------|
| `baiotti_gauge_drift.py` | Least-squares quadratic floor on \(h_+\) and \(h_\times\); Eq. (38) on \(r(h_+-i h_\times)\) |

## Reference

L. Baiotti, S. Bernuzzi, G. Corvino, R. De Pietri, and A. Nagar, *Gravitational-wave extraction from neutron-star oscillations: Comparing linear and nonlinear techniques*, [Phys. Rev. D **79**, 024002 (2009)](https://doi.org/10.1103/PhysRevD.79.024002).  
[doi:10.1103/PhysRevD.79.024002](https://doi.org/10.1103/PhysRevD.79.024002) · [arXiv:0808.4002](https://arxiv.org/abs/0808.4002)

If you use this code, please cite Baiotti et al. (2009). The quadratic subtraction must be applied before gravitational-wave analysis of strain obtained from \(\Psi_4\).
