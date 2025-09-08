# EGR Rotation Curves — Docker Reproducibility Suite

Entropy–Gated Response (EGR) rotation-curve fitter with ΔBIC model selection (EGR-7 vs EGR-10), robust diagnostics (DW/AR1/Breusch–Pagan), optional MCMC (emcee), and per-galaxy figures. **Docker-only** for zero-drama reproducibility.

---

## TL;DR

```bash
# from repo root
docker build -t egr-rc .

# Full run on your CSV (everything on: plots + diagnostics + MCMC)
docker run --rm \
  -v "$PWD/data:/work/data" \
  -v "$PWD/out:/work/out" \
  egr-rc egr-rc-run-all \
    --data /work/data/sparc_mass_models.csv \
    --outdir /work/out
````

Outputs land in `out/`:

```
out/
  figs/         <ID>_rc.png, <ID>_corner.png, <ID>_corr.png, <ID>_ppc.png
  tables/       summary.csv, posthoc.json
  run.json      args + versions (enables plots/RAR/MCMC for the session)
```

To smoke-test without data:

```bash
docker run --rm -v "$PWD/out:/work/out" egr-rc egr-rc-selftest --outdir /work/out
```

---

## Data format

Place a SPARC-like CSV in `data/` (or mount from elsewhere). The loader resolves common aliases; minimally you need:

* **ID** — galaxy label
* **R** — radius (kpc)
* **Vobs** — observed rotation speed (km/s)
* **e\_Vobs** — uncertainty (km/s)

Example: `data/sparc_mass_models.csv`

---

## What the pipeline does

* **Two families**

  * **EGR-7**: `[R, φ_w, A_w, A_p, k_p, σ_p, S_resp]`
  * **EGR-10** (EGR-7 + ring): `[A_ring, r_ring, σ_ring]`
* **Gate**: $g(Q_\mathrm{exp}) = \frac{Q_\mathrm{exp}}{Q_\mathrm{exp} + \sigma_p}$ with $Q_\mathrm{exp} = R \cdot S_\mathrm{resp}$
* **Error model**: $\sigma_\mathrm{eff}^2 = \sigma_\mathrm{meas}^2 + (s_\mathrm{frac} \, V_\mathrm{obs})^2$
* **Model selection**: keep EGR-10 iff **ΔBIC ≤ −10** (configurable)
* **Diagnostics**:
  DW, AR(1) ρ, Breusch–Pagan heteroscedasticity p-value, bound-hits counters, in-sample/hold-out RMSE deltas
* **MCMC (optional)**: emcee on the chosen model per galaxy; corner, correlation heatmap, and PPC (median + 68% band)

Everything above is enabled by `egr-rc-run-all` and written to one output directory.

---

## Repository layout

```
src/egrrc/
  rc.py         # the fitter & CLI (your original logic, packaged)
  run_all.py    # run all pages end-to-end (writes run.json to enable plots+RAR+MCMC)
  __main__.py   # python -m egrrc
tests/
  test_selftest.py
data/
  sparc_mass_models.csv   # your data lives here (or mount via -v)
out/
  ...                     # generated artifacts (gitignored)
Dockerfile
pyproject.toml
requirements.txt
Makefile
README.md
```

---

## Common runs

**1) End-to-end on your CSV (recommended)**

```bash
docker run --rm \
  -v "$PWD/data:/work/data" \
  -v "$PWD/out:/work/out" \
  egr-rc egr-rc-run-all \
    --data /work/data/sparc_mass_models.csv \
    --outdir /work/out
```

**2) Original one-page CLI** (four galaxies per page; plots+RAR+MCMC explicitly requested)

```bash
docker run --rm \
  -v "$PWD/data:/work/data" \
  -v "$PWD/out:/work/out" \
  egr-rc egr-rc 0 \
    --data /work/data/sparc_mass_models.csv \
    --plot --rar --mcmc \
    --outdir /work/out
```

**3) Quick self-test** (synthetic)

```bash
docker run --rm -v "$PWD/out:/work/out" egr-rc egr-rc-selftest --outdir /work/out
```

---

## Tuning knobs (optional)

* Restarts:

  * `--starts` (both) or `--starts7`, `--starts10` (per family)
* ΔBIC threshold: `--bic-thresh` (default 10)
* Hold-out: `--holdout 0.2` and `--holdout-mode random|outer`
* Residual tests: `--rar`
* MCMC: `--mcmc --walkers 64 --steps 1500 --burn 300`

For `egr-rc-run-all`, edit arguments right on the command line or (advanced) tweak the `run.json` it writes in `out/` and re-run pages.

---

## Determinism

* Seeds passed via `--seed` are used everywhere feasible.
* Multi-start jitter and MCMC are stochastic—set `--seed` to make runs repeatable.
* Existing `summary.csv` is rotated to `summary.<timestamp>.csv` when `--fresh` is set (run-all handles this at the first page).

---

## License / citation

If you use this code in a publication, please cite this repository and the accompanying paper.

MIT license (see file headers).

---

## Troubleshooting

* **No figures?** Ensure you bind-mount `out/` so PNGs persist:

  ```
  -v "$PWD/out:/work/out"
  ```
* **Column mismatch**: the loader prints resolved columns. If required columns are missing, add aliases or rename your CSV headers to `ID, R, Vobs, e_Vobs`.
* **Docker memory**: MCMC can be memory-heavy for very long chains; reduce `--steps` or turn off `--mcmc`.
