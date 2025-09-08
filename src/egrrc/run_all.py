"""
Run all pages end-to-end into a single output directory, with plots, RAR, and MCMC on.
This is a thin wrapper around egrrc.rc.process_page that writes run.json once.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
from math import ceil
import numpy as np
import pandas as pd

from .rc import (
    load_data, group_ids, process_page, write_run_json
)

def _default_args(data:str, outdir:Path, seed:int|None):
    # Mirror rc.write_run_json() structure
    return argparse.Namespace(
        page=0, data=data, plot=True,
        starts=12, starts7=None, starts10=None,
        bic_thresh=10.0, holdout=0.0, holdout_repeats=3, holdout_mode="random",
        rar=True, seed=seed, outdir=str(outdir), fresh=False,
        f7_widen=3.5, f7_tries=16, nested_f7=True,
        # Diagnostics toggles consumed in process_page via run.json:
        hm_eps=0.0,
        elm_enable=False, elm_mode="bounded", elm_alpha=0.35, elm_beta=0.35,
        elm_gates="", elm_phase_scale=1.0, elm_scale=1.0,
        mcmc=True, walkers=64, steps=1500, burn=300,
        selftest=False
    )

def run_all(data:str, outdir:Path, seed:int|None, page_size:int=4) -> None:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    # Write run.json once to turn on plotting, RAR, and MCMC
    args_ns = _default_args(data, outdir, seed)
    write_run_json(args_ns, outdir)

    df, _ = load_data(data)
    ids = group_ids(df)
    pages = ceil(len(ids) / max(1, page_size))
    for page in range(pages):
        process_page(
            df=df, page=page, do_plot=True,
            nstarts7=args_ns.starts if args_ns.starts7 is None else args_ns.starts7,
            nstarts10=args_ns.starts if args_ns.starts10 is None else args_ns.starts10,
            bic_thresh=args_ns.bic_thresh,
            holdout=args_ns.holdout, do_rar=True,
            seed=seed, outdir=outdir,
            holdout_repeats=args_ns.holdout_repeats,
            holdout_mode=args_ns.holdout_mode,
            fresh=(page == 0),  # rotate summary only once at start if requested
            f7_widen=args_ns.f7_widen, f7_tries=args_ns.f7_tries,
            nested_f7=args_ns.nested_f7
        )

def main():
    ap = argparse.ArgumentParser(description="Run all pages with plots+RAR+MCMC enabled")
    ap.add_argument("--data", required=False, default="data/SPARC_MassModels.csv",
                    help="Path to SPARC-like CSV (ID,R,Vobs,e_Vobs).")
    ap.add_argument("--outdir", default="out", help="Output directory (will be created).")
    ap.add_argument("--seed", type=int, default=137, help="RNG seed.")
    ap.add_argument("--page-size", type=int, default=4, help="Galaxies per page (default=4).")
    ap.add_argument("--selftest", action="store_true", help="Use synthetic self-test instead of reading CSV.")
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    if args.selftest:
        # Reuse the self-test harness in rc.py but push outputs to out/
        from .rc import run_selftest
        run_selftest(seed=args.seed, outdir=out)
    else:
        run_all(data=args.data, outdir=out, seed=args.seed, page_size=args.page_size)

if __name__ == "__main__":
    main()
