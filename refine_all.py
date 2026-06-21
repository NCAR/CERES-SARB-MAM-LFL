"""Rebuild every fine per-band LUT by sweeping refine_lut.py over all bands/modes.

After the base tables are (re)generated (see generate_lut.py), the production
lookup consumes the *fine* per-band tables produced by refine_lut.py. This driver
runs refine_lut.py for every (mode, band) in one command, in parallel.

    python refine_all.py                                  # all modes, all SW+LW bands
    python refine_all.py --aerosol aerosol_ceres.yaml     # production paths
    python refine_all.py --modes a3 --bands sw5 lw7       # explicit subset
    python refine_all.py --processes 4

Band counts default to the shortwave/longwave band dimensions of the base table
named in the config (falling back to 14 SW / 12 LW). Exits nonzero if any band
fails.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def _base_band_counts(aerosol, scheme, mode):
    """(n_sw, n_lw) from the base table dims; fall back to (14, 12)."""
    try:
        import xarray as xr

        with open(aerosol, "r") as stream:
            config = yaml.safe_load(stream)
        path = os.path.expandvars(config[scheme][mode]["filename_sarb"])
        with xr.open_dataset(path) as ds:
            return int(ds.sizes.get("sw_band", 14)), int(ds.sizes.get("lw_band", 12))
    except Exception:
        return 14, 12


def _bands(args):
    if args.bands:
        return list(args.bands)
    n_sw = args.sw if args.sw is not None else _base_band_counts(args.aerosol, args.scheme, args.modes[0])[0]
    n_lw = args.lw if args.lw is not None else _base_band_counts(args.aerosol, args.scheme, args.modes[0])[1]
    return ["sw%d" % i for i in range(1, n_sw + 1)] + ["lw%d" % i for i in range(1, n_lw + 1)]


def _run_one(aerosol, scheme, mode, band):
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "refine_lut.py"),
         "--aerosol", aerosol, "--scheme", scheme, "--mode", mode, "--band", band],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    return mode, band, proc.returncode, proc.stderr


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aerosol", default="aerosol.yaml")
    parser.add_argument("--scheme", default="MAM4")
    parser.add_argument("--modes", nargs="+", default=["a1", "a2", "a3", "a4"])
    parser.add_argument("--bands", nargs="+", default=None,
                        help="explicit band tokens (e.g. sw5 lw7); default = all SW+LW")
    parser.add_argument("--sw", type=int, default=None, help="number of SW bands (default from base table)")
    parser.add_argument("--lw", type=int, default=None, help="number of LW bands (default from base table)")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args(argv)

    bands = _bands(args)
    tasks = [(m, b) for m in args.modes for b in bands]
    print("scheme=%s modes=%s bands=%d (%s) tasks=%d processes=%d"
          % (args.scheme, ",".join(args.modes), len(bands), bands[0] + ".." + bands[-1],
             len(tasks), args.processes))

    ok, failures = 0, []
    with ThreadPoolExecutor(max_workers=args.processes) as pool:
        futures = {pool.submit(_run_one, args.aerosol, args.scheme, m, b): (m, b) for m, b in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            mode, band, code, stderr = future.result()
            if code == 0:
                ok += 1
            else:
                failures.append((mode, band, stderr))
                print("  FAIL %s %s" % (mode, band))
            if done % 20 == 0 or done == len(tasks):
                print("  %d / %d done (ok=%d fail=%d)" % (done, len(tasks), ok, len(failures)), flush=True)

    print("TOTAL ok=%d fail=%d" % (ok, len(failures)))
    for mode, band, stderr in failures:
        tail = "\n".join((stderr or "").strip().splitlines()[-3:])
        print("---- %s %s ----\n%s" % (mode, band, tail))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
