"""Command-line interface for the global OTL pipeline.

Usage:
    gotl [OPTIONS] COMMAND [ARGS]...

Environment variables (alternative to CLI options):
    GOTL_RESULTS_DIR
    GOTL_TDP_ROOT
    GOTL_HARDISP_ROOT
    GOTL_OTL_PARAMS_DIR
    GOTL_REGISTRY
    GOTL_HARDISP_EXE
    GOTL_TIDE_MODELS      (comma-separated, e.g. "TPXO9,FES2014b")

    GipsyX RINEX processing (Phase 0):
    GOTL_RINEX_ROOT        Root directory for RINEX files
    GOTL_GIPSYX_RC         Path to GipsyX rc_GipsyX.sh
    GOTL_JPL_PRODUCTS      JPL GNSS orbit/clock products directory
    GOTL_VMF1_DIR          VMF1 troposphere data directory
    GOTL_GIPSY_MAX_WORKERS Max parallel GipsyX subprocesses (default: 4)

Examples:
    # Process a single station
    gotl --results-dir /path/results --tdp-root /data/tdp \\
         --hardisp-root /data/hardisp --otl-params-dir /data/otl_params run aboa

    # Process all stations
    gotl run-all

    # Show completion status
    gotl status

    # Compare GPS vs models for one station
    gotl compare aboa
"""

from __future__ import annotations

import logging
import sys

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


def _make_config(results_dir, tdp_root, hardisp_root, otl_params_dir,
                 registry, hardisp_exe, tide_models,
                 rinex_root=None, gipsyx_rc=None, jpl_products=None,
                 vmf1_dir=None, gipsy_max_workers=4, gipsy_otl_model="FES2014"):
    from gotl.config import Config
    models = [m.strip() for m in tide_models.split(",")] if tide_models else ["TPXO9"]
    return Config(
        results_dir=results_dir,
        tdp_root=tdp_root,
        hardisp_root=hardisp_root,
        otl_params_dir=otl_params_dir,
        registry_path=registry,
        hardisp_exe=hardisp_exe,
        tide_models=models,
        rinex_root=rinex_root or None,
        gipsyx_rc=gipsyx_rc or None,
        jpl_products_dir=jpl_products or None,
        vmf1_dir=vmf1_dir or None,
        gipsy_max_workers=int(gipsy_max_workers),
        gipsy_otl_model=gipsy_otl_model,
    )


@click.group()
@click.option("--results-dir", envvar="GOTL_RESULTS_DIR", required=True,
              help="Directory for .h5 cache files.")
@click.option("--tdp-root", envvar="GOTL_TDP_ROOT", default="",
              help="Root of GipsyX TDP output ({tdp_root}/{stnm}_results/*.tdp).")
@click.option("--hardisp-root", envvar="GOTL_HARDISP_ROOT", default="",
              help="Root of hardisp time series files.")
@click.option("--otl-params-dir", envvar="GOTL_OTL_PARAMS_DIR", default="data/otl_params",
              show_default=True, help="Directory of {stnm}_{model}.otl files.")
@click.option("--registry", envvar="GOTL_REGISTRY", default="data/station_registry.csv",
              show_default=True, help="Station registry CSV.")
@click.option("--hardisp-exe", envvar="GOTL_HARDISP_EXE", default="hardisp/hardisp_exe",
              show_default=True, help="Path to compiled hardisp binary.")
@click.option("--tide-models", envvar="GOTL_TIDE_MODELS", default="TPXO9",
              show_default=True, help="Comma-separated tide model names.")
@click.option("--rinex-root", envvar="GOTL_RINEX_ROOT", default="",
              help="Root directory for RINEX files ({rinex_root}/{stnm}/*.Z).")
@click.option("--gipsyx-rc", envvar="GOTL_GIPSYX_RC", default="",
              help="Path to GipsyX rc_GipsyX.sh.")
@click.option("--jpl-products", envvar="GOTL_JPL_PRODUCTS", default="",
              help="JPL GNSS products directory ({jpl_products}/{YYYY}/).")
@click.option("--vmf1-dir", envvar="GOTL_VMF1_DIR", default="",
              help="VMF1 troposphere data directory.")
@click.option("--gipsy-max-workers", envvar="GOTL_GIPSY_MAX_WORKERS", default="4",
              show_default=True, help="Max parallel GipsyX subprocesses.")
@click.option("--gipsy-otl-model", envvar="GOTL_GIPSY_OTL_MODEL", default="FES2014",
              show_default=True, help="Tide model for GipsyX OTL correction.")
@click.pass_context
def main(ctx, results_dir, tdp_root, hardisp_root, otl_params_dir,
         registry, hardisp_exe, tide_models,
         rinex_root, gipsyx_rc, jpl_products, vmf1_dir, gipsy_max_workers,
         gipsy_otl_model):
    """Global OTL pipeline: GPS ocean tidal loading analysis."""
    ctx.ensure_object(dict)
    ctx.obj.update(dict(
        results_dir=results_dir, tdp_root=tdp_root, hardisp_root=hardisp_root,
        otl_params_dir=otl_params_dir, registry=registry,
        hardisp_exe=hardisp_exe, tide_models=tide_models,
        rinex_root=rinex_root, gipsyx_rc=gipsyx_rc, jpl_products=jpl_products,
        vmf1_dir=vmf1_dir, gipsy_max_workers=gipsy_max_workers,
        gipsy_otl_model=gipsy_otl_model,
    ))


@main.command()
@click.argument("station")
@click.option("--force", is_flag=True, help="Re-run all steps even if cached.")
@click.pass_context
def run(ctx, station, force):
    """Process a single STATION through the full pipeline."""
    from gotl.registry import load_registry
    cfg = _make_config(**ctx.obj)
    registry = load_registry(cfg.registry_path)
    if station not in registry:
        click.echo(f"ERROR: {station} not in registry {cfg.registry_path}", err=True)
        sys.exit(1)
    from gotl.pipeline import process_station
    try:
        process_station(registry[station], cfg, force=force)
        click.echo(f"Station {station}: complete.")
    except Exception as e:
        click.echo(f"ERROR: {station}: {e}", err=True)
        sys.exit(1)


@main.command("run-all")
@click.option("--force", is_flag=True, help="Re-run all steps even if cached.")
@click.option("--stations", default=None,
              help="Comma-separated subset of stations (default: all in registry).")
@click.pass_context
def run_all(ctx, force, stations):
    """Process all stations, continuing past per-station errors."""
    cfg = _make_config(**ctx.obj)
    stnlist = [s.strip() for s in stations.split(",")] if stations else None
    from gotl.runall import run_all as _run_all
    results = _run_all(cfg, stations=stnlist, force=force)
    n_ok = sum(v is None for v in results.values())
    n_fail = len(results) - n_ok
    click.echo(f"\nDone: {n_ok} succeeded, {n_fail} failed.")
    if n_fail > 0:
        for stn, err in results.items():
            if err is not None:
                click.echo(f"  FAIL  {stn}: {err}", err=True)
        sys.exit(1)


@main.command()
@click.pass_context
def status(ctx):
    """Show pipeline completion status for all stations."""
    from pathlib import Path
    from gotl.io.cache import StationCache
    from gotl.registry import load_registry
    cfg = _make_config(**ctx.obj)
    registry = load_registry(cfg.registry_path)
    steps = ["tdp", "hardisp", "combined", "otl_coeff"]
    header = f"{'Station':<8}" + "".join(f"  {s:<12}" for s in steps)
    click.echo(header)
    click.echo("-" * len(header))
    n_complete = 0
    for stnm in sorted(registry.keys()):
        cache = StationCache(Path(cfg.results_dir) / f"{stnm}.h5")
        marks = []
        all_done = True
        for step in steps:
            done = cache.has(step)
            marks.append("✓" if done else "✗")
            if not done:
                all_done = False
        if all_done:
            n_complete += 1
        row = f"{stnm:<8}" + "  ".join(f"  {m:<12}" for m in marks)
        click.echo(row)
    click.echo(f"\n{n_complete}/{len(registry)} stations complete.")


@main.command()
@click.argument("station")
@click.pass_context
def compare(ctx, station):
    """Compare GPS-estimated OTL vs tide model(s) for STATION."""
    from pathlib import Path
    from gotl.compare.residuals import compare_station
    cfg = _make_config(**ctx.obj)
    try:
        df = compare_station(
            station,
            cfg.results_dir,
            cfg.otl_params_dir,
            cfg.tide_models,
        )
    except FileNotFoundError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    click.echo(f"\nStation: {station}")
    for model in cfg.tide_models:
        click.echo(f"\n--- Model: {model} ---")
        click.echo(f"{'Const':<6} {'Comp':<5} {'obs(mm)':>9} {'mod(mm)':>9} "
                   f"{'dAmp(mm)':>9} {'vmfit(mm)':>10}")
        click.echo("-" * 55)
        for (const, comp), row in df.iterrows():
            click.echo(
                f"{const:<6} {comp:<5} "
                f"{row['obs_amp']*1e3:>9.2f} "
                f"{row.get(f'{model}_amp', float('nan'))*1e3:>9.2f} "
                f"{row.get(f'{model}_damp', float('nan'))*1e3:>9.2f} "
                f"{row.get(f'{model}_vmfit', float('nan'))*1e3:>10.2f}"
            )


@main.command("gen-hardisp")
@click.argument("station")
@click.option("--years", default=None,
              help="Comma-separated years (default: data_start to data_end from registry).")
@click.option("--model", default="TPXO9", show_default=True,
              help="Tide model name (selects {stnm}_{model}.otl file).")
@click.option("--force", is_flag=True, help="Re-generate even if cached.")
@click.pass_context
def gen_hardisp(ctx, station, years, model, force):
    """Generate hardisp time series for STATION."""
    from gotl.hardisp_gen.batch import generate_hardisp_station
    from gotl.registry import load_registry
    cfg = _make_config(**ctx.obj)
    registry = load_registry(cfg.registry_path)
    if station not in registry:
        click.echo(f"ERROR: {station} not in registry.", err=True)
        sys.exit(1)
    meta = registry[station]
    if years:
        yr_list = [int(y.strip()) for y in years.split(",")]
    else:
        yr_list = list(range(meta.data_start.year, meta.data_end.year + 1))
    try:
        results = generate_hardisp_station(
            cfg.hardisp_exe, cfg.otl_params_dir, station,
            yr_list, cfg.hardisp_root, model=model, force=force,
        )
        click.echo(f"Generated {len(results)} hardisp files for {station}.")
    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)


@main.command("process-rinex")
@click.argument("station")
@click.option("--start", default=None,
              help="Start date (YYYY-MM-DD). Default: all available RINEX files.")
@click.option("--end", default=None,
              help="End date (YYYY-MM-DD). Default: all available RINEX files.")
@click.option("--max-workers", type=int, default=None,
              help="Max parallel GipsyX processes (default: from config).")
@click.option("--force", is_flag=True, help="Re-process even if TDP files exist.")
@click.option("--dry-run", is_flag=True,
              help="Show what would be processed without running GipsyX.")
@click.pass_context
def process_rinex(ctx, station, start, end, max_workers, force, dry_run):
    """Process RINEX files through GipsyX PPP to produce TDP files (Phase 0).

    Converts daily RINEX observation files into a merged data record,
    then processes through GipsyX kinematic PPP with overlapping 30-hour
    windows. Output TDP files are written to {tdp_root}/{station}_results/.

    Requires: GipsyX installed, JPL products downloaded, RINEX files.
    Set GOTL_RINEX_ROOT, GOTL_GIPSYX_RC, and GOTL_JPL_PRODUCTS env vars.

    \b
    Examples:
        gotl process-rinex aboa
        gotl process-rinex aboa --start 2010-01-01 --end 2010-12-31
        gotl process-rinex aboa --dry-run
    """
    from datetime import date as date_type

    cfg = _make_config(**ctx.obj)

    if not cfg.rinex_root:
        click.echo("ERROR: --rinex-root or GOTL_RINEX_ROOT is required.", err=True)
        sys.exit(1)
    if not cfg.jpl_products_dir:
        click.echo("ERROR: --jpl-products or GOTL_JPL_PRODUCTS is required.", err=True)
        sys.exit(1)
    if not cfg.tdp_root:
        click.echo("ERROR: --tdp-root or GOTL_TDP_ROOT is required.", err=True)
        sys.exit(1)

    start_date = date_type.fromisoformat(start) if start else None
    end_date = date_type.fromisoformat(end) if end else None

    if dry_run:
        _process_rinex_dry_run(cfg, station, start_date, end_date)
        return

    from gotl.gipsy.batch import process_station_rinex
    try:
        result = process_station_rinex(
            station, cfg,
            start_date=start_date,
            end_date=end_date,
            max_workers=max_workers,
            force=force,
        )
        click.echo(
            f"\nStation {station}: {result.n_windows_ok} windows succeeded, "
            f"{result.n_windows_fail} failed."
        )
        if result.tdp_dir:
            click.echo(f"TDP output: {result.tdp_dir}")
        if result.error:
            click.echo(f"ERROR: {result.error}", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"ERROR: {station}: {e}", err=True)
        sys.exit(1)


def _process_rinex_dry_run(cfg, station, start_date, end_date):
    """Show what would be processed without running GipsyX."""
    from gotl.gipsy.rinex import discover_rinex
    from gotl.gipsy.products import check_products_available

    click.echo(f"Dry run for station: {station}")
    click.echo(f"RINEX root: {cfg.rinex_root}")
    click.echo(f"JPL products: {cfg.jpl_products_dir}")
    click.echo(f"TDP output: {cfg.tdp_root}/{station}_results/")

    try:
        rinex_files = discover_rinex(cfg.rinex_root, station, start_date, end_date)
    except FileNotFoundError as e:
        click.echo(f"\nNo RINEX files found: {e}", err=True)
        return

    click.echo(f"\nRINEX files: {len(rinex_files)}")
    click.echo(f"  Date range: {rinex_files[0].date} to {rinex_files[-1].date}")
    click.echo(f"  First: {rinex_files[0].path.name}")
    click.echo(f"  Last:  {rinex_files[-1].path.name}")

    dates = [r.date for r in rinex_files]
    available, missing = check_products_available(cfg.jpl_products_dir, dates)
    click.echo(f"\nJPL products: {len(available)} available, {len(missing)} missing")
    if missing:
        click.echo(f"  Missing years: {sorted(set(d.year for d in missing))}")


# ── RINEX download ───────────────────────────────────────────────


@main.command("download-rinex")
@click.argument("station")
@click.option("--start", required=True,
              help="Start date (YYYY-MM-DD).")
@click.option("--end", required=True,
              help="End date (YYYY-MM-DD).")
@click.option("--archives", default="cddis,bkg", show_default=True,
              help="Comma-separated archive names to try, in priority order.")
@click.option("--force", is_flag=True,
              help="Re-download even if files already exist.")
@click.option("--dry-run", is_flag=True,
              help="Show URLs without downloading.")
@click.pass_context
def download_rinex(ctx, station, start, end, archives, force, dry_run):
    """Download RINEX observation files for STATION from GNSS archives.

    Downloads daily Hatanaka-compressed RINEX 2 files (.{YY}d.Z) to
    {rinex_root}/{station}/. Tries archives in the specified order
    (default: CDDIS, then BKG).

    CDDIS requires NASA Earthdata authentication. Set one of:
      - EARTHDATA_TOKEN env var
      - ~/.netrc entry for urs.earthdata.nasa.gov
      - EARTHDATA_USER + EARTHDATA_PASSWORD env vars

    \b
    Examples:
        gotl download-rinex brux --start 2024-01-01 --end 2024-01-31
        gotl download-rinex aboa --start 2020-01-01 --end 2020-12-31 --dry-run
        gotl download-rinex brux --start 2024-01-01 --end 2024-01-03 --archives bkg
    """
    from datetime import date as date_type
    from gotl.download import download_rinex_range

    cfg = _make_config(**ctx.obj)
    if not cfg.rinex_root or str(cfg.rinex_root) == "":
        click.echo("ERROR: --rinex-root or GOTL_RINEX_ROOT is required.", err=True)
        sys.exit(1)

    start_date = date_type.fromisoformat(start)
    end_date = date_type.fromisoformat(end)
    archive_list = [a.strip() for a in archives.split(",")]

    click.echo(f"Downloading RINEX for {station}: {start_date} to {end_date}")
    click.echo(f"Archives: {', '.join(archive_list)}")
    click.echo(f"Destination: {cfg.rinex_root}/{station}/")

    result = download_rinex_range(
        stnm=station,
        start_date=start_date,
        end_date=end_date,
        rinex_root=cfg.rinex_root,
        archives=archive_list,
        skip_existing=not force,
        dry_run=dry_run,
    )

    click.echo(
        f"\nDone: {result.dates_downloaded} downloaded, "
        f"{result.dates_skipped} skipped, "
        f"{result.dates_failed} failed "
        f"(of {result.dates_requested} requested)."
    )
    if result.failures:
        for d, err in sorted(result.failures.items()):
            click.echo(f"  FAIL {d}: {err}", err=True)
        sys.exit(1)


# ── LoadDef OTL parameter generation ────────────────────────────────


def _make_loaddef_env(ctx):
    """Build LoadDefEnv from CLI context."""
    from gotl.loaddef.env import load_loaddef_env

    obj = ctx.obj
    loaddef_root = obj.get("loaddef_root") or None
    loaddef_data_dir = obj.get("loaddef_data_dir") or None

    if not loaddef_root:
        click.echo("ERROR: --loaddef-root or GOTL_LOADDEF_ROOT is required.", err=True)
        sys.exit(1)
    if not loaddef_data_dir:
        click.echo("ERROR: --loaddef-data-dir or GOTL_LOADDEF_DATA_DIR is required.", err=True)
        sys.exit(1)

    return load_loaddef_env(loaddef_root, loaddef_data_dir)


@main.command("gen-otl-params")
@click.argument("station", required=False)
@click.option("--all", "all_stations", is_flag=True,
              help="Generate for all stations in registry.")
@click.option("--stations", default=None,
              help="Comma-separated list of stations.")
@click.option("--tide-model", default="FES2014", show_default=True,
              help="Tide model (FES2014, TPXO9, EOT20).")
@click.option("--earth-model", default="PREM", show_default=True,
              help="Earth model (PREM or custom name).")
@click.option("--mpi-np", type=int, default=None,
              help="MPI processes (default: from config).")
@click.option("--force", is_flag=True, help="Regenerate existing files.")
@click.option("--loaddef-root", envvar="GOTL_LOADDEF_ROOT", default="",
              help="Path to LoadDef-main/ directory.")
@click.option("--loaddef-data-dir", envvar="GOTL_LOADDEF_DATA_DIR", default="",
              help="Path to LoadDef data (Green's functions, grids).")
@click.pass_context
def gen_otl_params(ctx, station, all_stations, stations, tide_model, earth_model,
                   mpi_np, force, loaddef_root, loaddef_data_dir):
    """Generate OTL parameter files (.otl) using LoadDef.

    Runs the LoadDef pipeline: Love Numbers → Green's Functions →
    Convolution → Harpos .otl format. Pre-computed Green's functions
    are reused automatically.

    \b
    Examples:
        gotl gen-otl-params aboa --tide-model FES2014
        gotl gen-otl-params --all --tide-model TPXO9
        gotl gen-otl-params --stations aboa,brux --earth-model PREM
    """
    # Store loaddef options in context for _make_loaddef_env
    ctx.obj["loaddef_root"] = loaddef_root
    ctx.obj["loaddef_data_dir"] = loaddef_data_dir

    env = _make_loaddef_env(ctx)
    cfg = _make_config(**{k: v for k, v in ctx.obj.items()
                          if k not in ("loaddef_root", "loaddef_data_dir")})

    from gotl.loaddef.batch import generate_otl_params as gen_otl
    from gotl.registry import load_registry

    registry = load_registry(cfg.registry_path)
    mpi_np = mpi_np or 4

    # Determine station list
    if all_stations:
        stn_list = list(registry.keys())
    elif stations:
        stn_list = [s.strip() for s in stations.split(",")]
    elif station:
        stn_list = [station]
    else:
        click.echo("ERROR: Provide STATION, --stations, or --all.", err=True)
        sys.exit(1)

    ok_count = 0
    fail_count = 0
    for stnm in stn_list:
        if stnm not in registry:
            click.echo(f"WARNING: {stnm} not in registry, skipping.", err=True)
            fail_count += 1
            continue

        meta = registry[stnm]
        result = gen_otl(
            stnm=stnm,
            lat=meta.lat,
            lon=meta.lon,
            alt=meta.alt_m,
            env=env,
            otl_params_dir=cfg.otl_params_dir,
            tide_model=tide_model,
            earth_model=earth_model,
            mpi_np=mpi_np,
            force=force,
        )

        if result.error:
            click.echo(f"FAIL {stnm}: {result.error}", err=True)
            fail_count += 1
        else:
            click.echo(f"  OK {stnm}: {result.otl_path}")
            ok_count += 1

    click.echo(f"\nDone: {ok_count} succeeded, {fail_count} failed.")
    if fail_count:
        sys.exit(1)


@main.group()
def loaddef():
    """LoadDef sub-commands (Love Numbers, Green's Functions, Convolution)."""
    pass


# ── Cloud (Coiled) commands ──────────────────────────────────────


@main.group()
def cloud():
    """Cloud pipeline commands (Coiled/Dask on AWS)."""
    pass


@cloud.command("local-test")
@click.option("--stations", default="aboa",
              help="Comma-separated stations to test (default: aboa).")
@click.option("--force", is_flag=True, help="Re-run all steps even if cached.")
@click.pass_context
def cloud_local_test(ctx, stations, force):
    """Test the cloud workflow locally using Dask LocalCluster (no AWS cost).

    Validates the full data flow: serialize metadata, stage files,
    process in subprocess workers, collect results.

    \b
    Examples:
        gotl cloud local-test
        gotl cloud local-test --stations brux
        gotl cloud local-test --stations aboa,brux --force
    """
    try:
        from gotl.cloud.local import run_local_test
    except ImportError as e:
        click.echo(
            f"ERROR: Cloud dependencies not installed. "
            f"Run: poetry install --with cloud\n  ({e})",
            err=True,
        )
        sys.exit(1)

    cfg = _make_config(**ctx.obj)
    stn_list = [s.strip() for s in stations.split(",")]

    click.echo(f"Local cloud test: stations={stn_list}")
    results = run_local_test(cfg, stn_list, force=force)

    n_ok = sum(1 for r in results.values() if r["status"] == "ok")
    n_fail = len(results) - n_ok
    click.echo(f"\nDone: {n_ok} succeeded, {n_fail} failed.")
    for stnm, r in sorted(results.items()):
        status = "OK" if r["status"] == "ok" else "FAIL"
        elapsed = r.get("elapsed_sec", 0)
        msg = f"  {status}  {stnm}  ({elapsed:.1f}s)"
        if r.get("error"):
            msg += f"  — {r['error']}"
        click.echo(msg)
    if n_fail > 0:
        sys.exit(1)


@cloud.command("run")
@click.option("--stations", default=None,
              help="Comma-separated stations (default: all in registry).")
@click.option("--s3-bucket", envvar="GOTL_S3_BUCKET", required=True,
              help="S3 bucket for staging data.")
@click.option("--s3-prefix", default="cloud-runs",
              show_default=True, help="S3 key prefix.")
@click.option("--aws-region", default="us-east-1",
              show_default=True, help="AWS region.")
@click.option("--n-workers", default=3, type=int,
              show_default=True, help="Number of Coiled workers.")
@click.option("--vm-type", default="m6i.large",
              show_default=True, help="EC2 instance type.")
@click.option("--use-spot/--no-spot", default=True,
              show_default=True, help="Use spot instances.")
@click.option("--dry-run", is_flag=True,
              help="Stage data to S3 without launching workers.")
@click.option("--force", is_flag=True, help="Re-run all steps even if cached.")
@click.pass_context
def cloud_run(ctx, stations, s3_bucket, s3_prefix, aws_region,
              n_workers, vm_type, use_spot, dry_run, force):
    """Run pipeline on Coiled cloud workers.

    Stages station data to S3, distributes process_station() across
    Coiled workers, and downloads results.

    \b
    Examples:
        gotl cloud run --stations brux --s3-bucket gotl-pipeline --dry-run
        gotl cloud run --stations brux,aboa --s3-bucket gotl-pipeline
        gotl cloud run --s3-bucket gotl-pipeline --n-workers 10
    """
    try:
        from gotl.cloud import run_cloud
        from gotl.cloud.config import CloudConfig
    except ImportError as e:
        click.echo(
            f"ERROR: Cloud dependencies not installed. "
            f"Run: poetry install --with cloud\n  ({e})",
            err=True,
        )
        sys.exit(1)

    cfg = _make_config(**ctx.obj)
    cloud_cfg = CloudConfig(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        aws_region=aws_region,
        use_spot=use_spot,
        worker_vm_type=vm_type,
        n_workers=n_workers,
        tide_models=cfg.tide_models,
        force=force,
    )

    stn_list = [s.strip() for s in stations.split(",")] if stations else None

    results = run_cloud(cfg, cloud_cfg, stations=stn_list, dry_run=dry_run)

    if dry_run:
        click.echo(f"\nDry run complete. {len(results)} stations staged to S3.")
        return

    n_ok = sum(1 for r in results.values() if r["status"] == "ok")
    n_fail = len(results) - n_ok
    click.echo(f"\nDone: {n_ok} succeeded, {n_fail} failed.")
    for stnm, r in sorted(results.items()):
        status = "OK" if r["status"] == "ok" else "FAIL"
        elapsed = r.get("elapsed_sec", 0)
        msg = f"  {status}  {stnm}  ({elapsed:.1f}s)"
        if r.get("error"):
            msg += f"  — {r['error']}"
        click.echo(msg)
    if n_fail > 0:
        sys.exit(1)


@cloud.command("setup-env")
@click.option("--s3-bucket", envvar="GOTL_S3_BUCKET", required=True,
              help="S3 bucket name.")
@click.option("--env-name", default="gotl-env",
              show_default=True, help="Coiled software environment name.")
def cloud_setup_env(s3_bucket, env_name):
    """Create/update the Coiled software environment.

    Builds a gotl wheel and registers it with Coiled so cloud workers
    have the gotl package and all dependencies available.
    """
    try:
        from gotl.cloud import ensure_software_env
        from gotl.cloud.config import CloudConfig
    except ImportError as e:
        click.echo(
            f"ERROR: Cloud dependencies not installed. "
            f"Run: poetry install --with cloud\n  ({e})",
            err=True,
        )
        sys.exit(1)

    cloud_cfg = CloudConfig(
        s3_bucket=s3_bucket,
        software_env_name=env_name,
    )
    ensure_software_env(cloud_cfg)
    click.echo(f"Software environment '{env_name}' is ready.")


@cloud.command("package-gipsyx")
@click.option("--gipsyx-root", envvar="GOTL_GIPSYX_ROOT",
              default="/path/to/GipsyX/GipsyX-2.0",
              help="Path to GipsyX-2.0/ directory.")
@click.option("--venv39", default="/path/to/GipsyX/venv39",
              help="Path to Python 3.9 venv.")
@click.option("--goa-var", default="/path/to/GipsyX/goa-var",
              help="Path to goa-var/ directory.")
@click.option("--python39", default=None,
              help="Path to standalone Python 3.9 install (e.g. UV cpython-3.9 dir). "
                   "Included so cloud workers don't need Python 3.9 installed.")
@click.option("--output", default="/path/to/gipsyx-cloud.tar.gz",
              help="Output archive path.")
@click.option("--s3-bucket", envvar="GOTL_S3_BUCKET", default=None,
              help="Upload to S3 bucket after creating archive.")
def cloud_package_gipsyx(gipsyx_root, venv39, goa_var, python39, output, s3_bucket):
    """Package GipsyX for cloud deployment.

    Creates a minimal tar.gz archive (~1-2 GB) containing GipsyX binaries,
    Python 3.9 venv, standalone Python 3.9 interpreter, and trimmed goa-var
    support data. Optionally uploads to S3.

    \b
    Examples:
        gotl cloud package-gipsyx --python39 ~/.local/share/uv/python/cpython-3.9-linux-x86_64-gnu
        gotl cloud package-gipsyx --s3-bucket YOUR_S3_BUCKET
    """
    from pathlib import Path
    from gotl.cloud.gipsyx import create_gipsyx_archive, upload_gipsyx_archive

    archive = create_gipsyx_archive(
        Path(gipsyx_root), Path(venv39), Path(goa_var), Path(output),
        python39_root=Path(python39) if python39 else None,
    )
    click.echo(f"Archive created: {archive} ({archive.stat().st_size / 1e6:.0f} MB)")

    if s3_bucket:
        uri = upload_gipsyx_archive(archive, s3_bucket)
        click.echo(f"Uploaded to: {uri}")


@cloud.command("stage-products")
@click.option("--gcore-dir", required=True,
              help="Local directory with GCORE-format JPL products (GNSS.pos, etc).")
@click.option("--year", required=True, type=int,
              help="Year label for S3 path.")
@click.option("--s3-bucket", envvar="GOTL_S3_BUCKET", required=True,
              help="S3 bucket.")
def cloud_stage_products(gcore_dir, year, s3_bucket):
    """Upload GCORE JPL products to S3.

    Products must already be in GCORE format (use fetchGNSSproducts.py
    to create from per-day files).

    \b
    Examples:
        gotl cloud stage-products --gcore-dir /path/to/jpl_gcore --year 2024 --s3-bucket YOUR_S3_BUCKET
    """
    from pathlib import Path
    from gotl.cloud.gipsyx import upload_jpl_gcore

    n = upload_jpl_gcore(Path(gcore_dir), s3_bucket, year)
    click.echo(f"Uploaded {n} product files to s3://{s3_bucket}/gipsyx/jpl_gcore/{year}/")


@cloud.command("run-gipsyx")
@click.option("--stations", required=True,
              help="Comma-separated stations to process.")
@click.option("--year", required=True, type=int,
              help="Processing year.")
@click.option("--s3-bucket", envvar="GOTL_S3_BUCKET", required=True,
              help="S3 bucket.")
@click.option("--s3-prefix", default="cloud-runs",
              show_default=True, help="S3 key prefix for run data.")
@click.option("--s3-rinex-prefix", default=None,
              help="Shared RINEX prefix (e.g. 'rinex'). When set, workers "
                   "read RINEX from s3://BUCKET/PREFIX/{stnm}/ instead of "
                   "the per-run inputs path, and the CLI skips re-uploading "
                   "RINEX. OTL files still stage per-run.")
@click.option("--aws-region", default="us-east-1",
              show_default=True, help="AWS region.")
@click.option("--n-workers", default=1, type=int,
              show_default=True, help="Number of Coiled workers.")
@click.option("--batch-size", default=1, type=int,
              show_default=True, help="Stations processed sequentially per worker. "
                                      "Amortizes the ~15 min GipsyX install + JPL "
                                      "download + Python 3.9 setup across N stations. "
                                      "Use 1 for single-station-per-worker mode.")
@click.option("--vm-type", default="c6i.2xlarge",
              show_default=True, help="EC2 instance type (need 8 vCPU for parallel windows).")
@click.option("--max-workers-per-station", default=8, type=int,
              show_default=True,
              help="Parallel rnxEditGde + gd2e workers per station. Should "
                   "match the VM's vCPU count: 8 for *.2xlarge, 16 for "
                   "*.4xlarge, 32 for *.8xlarge.")
@click.option("--use-spot/--no-spot", default=True,
              show_default=True, help="Use spot instances.")
@click.option("--skip-full-merge/--no-skip-full-merge", default=False,
              show_default=True,
              help="Skip the full-year drMerge in process_station_rinex. "
                   "Per-window mini-DRs (and single-DR fallback for edge "
                   "windows) cover all data; the full merge is wasted work.")
@click.option("--dry-run", is_flag=True,
              help="Stage RINEX to S3 without launching workers.")
@click.pass_context
def cloud_run_gipsyx(ctx, stations, year, s3_bucket, s3_prefix, s3_rinex_prefix,
                     aws_region, n_workers, batch_size, vm_type,
                     max_workers_per_station, use_spot,
                     skip_full_merge, dry_run):
    """Run GipsyX PPP on Coiled cloud workers.

    Prerequisites:
        1. gotl cloud package-gipsyx --s3-bucket BUCKET
        2. gotl cloud stage-products --gcore-dir DIR --year YEAR --s3-bucket BUCKET

    Stages RINEX files to S3, then launches Coiled workers to run
    the full RINEX → TDP pipeline.

    \b
    Examples:
        gotl cloud run-gipsyx --stations brux --year 2024 --s3-bucket YOUR_S3_BUCKET
        gotl cloud run-gipsyx --stations brux,aboa --year 2024 --s3-bucket YOUR_S3_BUCKET --n-workers 2
    """
    try:
        import boto3
        import coiled
    except ImportError as e:
        click.echo(f"ERROR: Cloud deps not installed: {e}", err=True)
        sys.exit(1)

    from pathlib import Path
    from gotl.cloud.gipsyx import process_station_batch_remote
    from gotl.cloud.s3 import generate_run_id
    from gotl.cloud.worker import serialize_meta
    from gotl.registry import load_registry

    cfg = _make_config(**ctx.obj)
    registry = load_registry(cfg.registry_path)
    stn_list = [s.strip() for s in stations.split(",")]

    unknown = [s for s in stn_list if s not in registry]
    if unknown:
        click.echo(f"WARNING: Not in registry: {unknown}", err=True)
    stn_list = [s for s in stn_list if s in registry]

    run_id = generate_run_id()
    s3_run_prefix = f"{s3_prefix}/{run_id}"
    s3_client = boto3.client("s3", region_name=aws_region)

    click.echo(f"Run ID: {run_id}")
    click.echo(f"S3 path: s3://{s3_bucket}/{s3_run_prefix}/")

    # OTL files are required (GipsyX runs with OCEANLOAD == On)
    otl_model = cfg.gipsy_otl_model
    otl_params_dir = Path(cfg.otl_params_dir)

    # Stage RINEX + OTL file for each station
    staged_stations = []
    for stnm in stn_list:
        rinex_dir = cfg.rinex_root / stnm if cfg.rinex_root else None
        if not rinex_dir or not rinex_dir.is_dir():
            click.echo(f"WARNING: No RINEX directory for {stnm}, skipping", err=True)
            continue

        otl_file = otl_params_dir / f"{stnm}_{otl_model}.otl"
        if not otl_file.is_file():
            click.echo(
                f"WARNING: No OTL file for {stnm} at {otl_file}, skipping. "
                f"Generate it with: gotl gen-otl-params {stnm} --tide-model {otl_model}",
                err=True,
            )
            continue

        count = 0
        if s3_rinex_prefix:
            # Shared RINEX prefix — assume pre-staged via `aws s3 sync`.
            # Caller is responsible for keeping s3://BUCKET/PREFIX/{stnm}/ current.
            pass
        else:
            for f in sorted(rinex_dir.iterdir()):
                key = f"{s3_run_prefix}/inputs/{stnm}/rinex/{stnm}/{f.name}"
                s3_client.upload_file(str(f), s3_bucket, key)
                count += 1

        otl_key = f"{s3_run_prefix}/inputs/{stnm}/otl/{otl_file.name}"
        s3_client.upload_file(str(otl_file), s3_bucket, otl_key)

        if s3_rinex_prefix:
            click.echo(
                f"Staged {stnm}: OTL ({otl_file.name}); "
                f"RINEX reused from s3://{s3_bucket}/{s3_rinex_prefix}/{stnm}/"
            )
        else:
            click.echo(f"Staged {stnm}: {count} RINEX files + OTL ({otl_file.name})")
        staged_stations.append(stnm)

    stn_list = staged_stations
    if not stn_list:
        click.echo("ERROR: No stations staged (missing RINEX or OTL).", err=True)
        sys.exit(1)

    if dry_run:
        click.echo("Dry run — RINEX + OTL staged, no workers launched.")
        return

    # Submit to Coiled. Workers run process_station_batch_remote, which
    # downloads the GipsyX archive + JPL products + Python 3.9 once per
    # worker and processes the batch sequentially. batch_size=1 emulates
    # the single-station-per-worker mode (no amortization).
    @coiled.function(
        name="gotl-gipsyx",
        vm_type=vm_type,
        spot_policy="spot_with_fallback" if use_spot else "on-demand",
        idle_timeout="10 minutes",
        region=aws_region,
        n_workers=n_workers,
        disk_size="100 GiB",
    )
    def remote_fn(batch):
        return process_station_batch_remote(
            station_args=batch,
            s3_bucket=s3_bucket,
            s3_prefix=s3_run_prefix,
            year=year,
            s3_rinex_prefix=s3_rinex_prefix,
            max_workers=max_workers_per_station,
            otl_model=otl_model,
            skip_full_merge=skip_full_merge,
        )

    meta_dicts = {s: serialize_meta(registry[s]) for s in stn_list}
    batches = [
        [(s, meta_dicts[s]) for s in stn_list[i:i + batch_size]]
        for i in range(0, len(stn_list), batch_size)
    ]

    click.echo(
        f"Submitting {len(stn_list)} stations to Coiled "
        f"({len(batches)} batches × {batch_size} stations, {n_workers} workers)..."
    )
    # retries=2: if a worker dies (typical: spot reclaim), dask re-submits
    # the batch task to a fresh worker. Combined with the S3-existence check
    # inside process_station_batch_remote, the retry skips already-uploaded
    # stations and only redoes the in-flight one.
    nested_results = list(remote_fn.map(batches, retries=2))
    results = [r for batch_results in nested_results for r in batch_results]

    for r in results:
        status = "OK" if r["status"] == "ok" else "FAIL"
        click.echo(
            f"  {status}  {r['stnm']}  "
            f"({r.get('n_windows_ok', 0)} windows ok, "
            f"{r.get('elapsed_sec', 0):.0f}s)"
        )
        if r.get("error"):
            click.echo(f"         Error: {r['error']}", err=True)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    click.echo(f"\nDone: {n_ok}/{len(results)} stations succeeded.")
    if n_ok < len(results):
        sys.exit(1)


@cloud.command("run-loaddef")
@click.option("--stations", required=True,
              help="Comma-separated stations to process.")
@click.option("--tide-model", default="TPXO9", show_default=True,
              help="Tide model (TPXO9, FES2014, EOT20).")
@click.option("--earth-model", default="PREM", show_default=True)
@click.option("--mpi-np", default=4, type=int, show_default=True,
              help="MPI ranks per station's convolution.")
@click.option("--s3-bucket", envvar="GOTL_S3_BUCKET", required=True)
@click.option("--s3-otl-prefix", default="otl_params/TPXO9",
              show_default=True,
              help="S3 prefix where output .otl files will be uploaded.")
@click.option("--aws-region", default="us-east-1", show_default=True)
@click.option("--n-workers", default=2, type=int, show_default=True,
              help="Number of Coiled workers (cluster size).")
@click.option("--batch-size", default=30, type=int, show_default=True,
              help="Stations per worker per task. The 5-min worker setup "
                   "amortizes; larger batches reduce setup overhead.")
@click.option("--vm-type", default="r6i.4xlarge", show_default=True,
              help="High-RAM instance for TPXO9 convolution (~19 GB peak).")
@click.option("--use-spot/--no-spot", default=True, show_default=True)
@click.pass_context
def cloud_run_loaddef(ctx, stations, tide_model, earth_model, mpi_np,
                      s3_bucket, s3_otl_prefix, aws_region,
                      n_workers, batch_size, vm_type, use_spot):
    """Run LoadDef OTL parameter generation on Coiled workers.

    Prerequisites:
        1. gotl cloud package-gipsyx           (provides venv39 + mpi4py)
        2. bash scripts/cloud/build_loaddef_bundle.sh
           aws s3 cp /tmp/loaddef-cloud.tar.gz s3://BUCKET/gipsyx/loaddef-cloud.tar.gz

    Output .otl files are uploaded to s3://BUCKET/{s3-otl-prefix}/
    as each station finishes; the worker skips stations whose .otl
    already exists in S3 (resumable across spot reclaims).
    """
    try:
        import boto3  # noqa: F401
        import coiled
    except ImportError as e:
        click.echo(f"ERROR: Cloud deps not installed: {e}", err=True)
        sys.exit(1)

    from gotl.cloud.loaddef import process_loaddef_batch_remote

    stn_list = [s.strip() for s in stations.split(",") if s.strip()]
    batches = [stn_list[i:i + batch_size]
               for i in range(0, len(stn_list), batch_size)]

    click.echo(
        f"Submitting {len(stn_list)} stations to Coiled "
        f"({len(batches)} batches × ≤{batch_size} stations, "
        f"{n_workers} workers, {vm_type})..."
    )

    @coiled.function(
        name="gotl-loaddef",
        vm_type=vm_type,
        spot_policy="spot_with_fallback" if use_spot else "on-demand",
        idle_timeout="10 minutes",
        region=aws_region,
        n_workers=n_workers,
        disk_size="80 GiB",
    )
    def remote_fn(batch):
        return process_loaddef_batch_remote(
            stations=batch,
            s3_bucket=s3_bucket,
            s3_otl_prefix=s3_otl_prefix,
            tide_model=tide_model,
            earth_model=earth_model,
            mpi_np=mpi_np,
        )

    nested = list(remote_fn.map(batches, retries=2))
    results = [r for batch_results in nested for r in batch_results]

    for r in results:
        status = "OK" if r["status"] == "ok" else (
            "SKIP" if r["status"] == "skipped" else "FAIL"
        )
        click.echo(f"  {status:<4} {r['stnm']}  ({r.get('elapsed_sec', 0):.0f}s)")
        if r.get("error"):
            click.echo(f"         Error: {r['error']}", err=True)

    n_ok = sum(1 for r in results if r["status"] in ("ok", "skipped"))
    click.echo(f"\nDone: {n_ok}/{len(results)} stations succeeded.")
    if n_ok < len(results):
        sys.exit(1)


# ── Web dashboard ────────────────────────────────────────────────


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True,
              help="Host to bind to.")
@click.option("--port", default=8050, show_default=True, type=int,
              help="Port to listen on.")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development.")
def serve(host, port, reload):
    """Start the web dashboard (requires 'poetry install --with web')."""
    try:
        from gotl.web import run_server
    except ImportError as e:
        click.echo(
            f"ERROR: Web dependencies not installed. "
            f"Run: poetry install --with web\n  ({e})",
            err=True,
        )
        sys.exit(1)
    click.echo(f"Starting gotl dashboard at http://{host}:{port}")
    run_server(host=host, port=port, reload=reload)


@loaddef.command("love-numbers")
@click.option("--earth-model", default="PREM", show_default=True)
@click.option("--mpi-np", type=int, default=4, show_default=True)
@click.option("--loaddef-root", envvar="GOTL_LOADDEF_ROOT", required=True)
@click.option("--loaddef-data-dir", envvar="GOTL_LOADDEF_DATA_DIR", required=True)
@click.pass_context
def loaddef_love_numbers(ctx, earth_model, mpi_np, loaddef_root, loaddef_data_dir):
    """Compute Love Numbers from an Earth model."""
    ctx.obj["loaddef_root"] = loaddef_root
    ctx.obj["loaddef_data_dir"] = loaddef_data_dir
    env = _make_loaddef_env(ctx)

    from gotl.loaddef.runner import run_love_numbers
    from pathlib import Path

    result = run_love_numbers(env, earth_model, Path(loaddef_data_dir), mpi_np=mpi_np)
    if result.success:
        click.echo(f"Love numbers computed ({result.elapsed_sec:.1f}s)")
        for f in result.output_files:
            click.echo(f"  {f}")
    else:
        click.echo(f"FAILED: {result.error}", err=True)
        sys.exit(1)


@loaddef.command("greens-functions")
@click.option("--earth-model", default="PREM", show_default=True)
@click.option("--mpi-np", type=int, default=4, show_default=True)
@click.option("--loaddef-root", envvar="GOTL_LOADDEF_ROOT", required=True)
@click.option("--loaddef-data-dir", envvar="GOTL_LOADDEF_DATA_DIR", required=True)
@click.pass_context
def loaddef_greens_functions(ctx, earth_model, mpi_np, loaddef_root, loaddef_data_dir):
    """Compute Green's Functions from Love Numbers."""
    ctx.obj["loaddef_root"] = loaddef_root
    ctx.obj["loaddef_data_dir"] = loaddef_data_dir
    env = _make_loaddef_env(ctx)

    from gotl.loaddef.runner import run_greens_functions
    from pathlib import Path

    lln_file = Path(loaddef_data_dir) / "Love_Numbers" / "LLN" / f"lln_{earth_model}.txt"
    if not lln_file.exists():
        click.echo(f"ERROR: LLN file not found: {lln_file}", err=True)
        click.echo("Run 'gotl loaddef love-numbers' first.", err=True)
        sys.exit(1)

    result = run_greens_functions(env, lln_file, earth_model,
                                  Path(loaddef_data_dir), mpi_np=mpi_np)
    if result.success:
        click.echo(f"Green's functions computed ({result.elapsed_sec:.1f}s)")
        for f in result.output_files:
            click.echo(f"  {f}")
    else:
        click.echo(f"FAILED: {result.error}", err=True)
        sys.exit(1)
