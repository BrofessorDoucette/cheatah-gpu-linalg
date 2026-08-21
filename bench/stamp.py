"""Provenance stamps for cheatah-gpu-linalg's generated benchmark tables.

The cheatah repo's scripts/bench_table.purr reads these to answer one question a Markdown
table cannot answer about itself: has the code this table measured changed since it was
measured? The fields that matter to it are `suite:` (which BENCH region this belongs to),
`commit:` and `watch:` (together: is it stale?) and `publishable:`.

Kept in one module so compare.py, crossover.py and emit_tables.py cannot drift into
disagreeing about what a stamp looks like — the same reason the cheatah side has
scripts/bench/stamp.purr.
"""
import json
import os
import platform
import subprocess
import time


def _cap(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def commit():
    """`(dirty)` is recorded rather than hidden: a number taken from an uncommitted tree cannot
    be traced back to the source that produced it, and the gate refuses to publish one."""
    c = _cap("git rev-parse --short HEAD") or "unknown"
    # SOURCE dirtiness, excluding docs/bench: regenerating an artifact rewrites a tracked file
    # there, which must not make the measurement look untraceable. The refresh clears stale
    # stat info, which on its own makes --quiet report a difference that is not there.
    subprocess.run("git update-index --refresh", shell=True, capture_output=True)
    if subprocess.run("git diff --quiet -- . ':!docs/bench'", shell=True).returncode != 0:
        c += " (dirty)"
    return c


def gpu():
    """The GPU, its driver, and the Vulkan loader version — the three things that decide
    whether a number is reproducible. A table that names only "RTX 3070 Ti" cannot be."""
    name = _cap("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
    if name:
        return name.replace(", ", " (driver ") + ")"
    return _cap("vulkaninfo --summary 2>/dev/null | awk -F'= ' '/deviceName/{print $2; exit}'") \
        or "unknown GPU"


def host():
    cpu = _cap("awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo") or platform.machine()
    return f"{cpu}, {os.cpu_count()} CPUs, {platform.system()} {platform.release()}"


def build(suite, watch, competitors, harness, statistic, produced_by, kind=None,
          publishable="true"):
    """The region body's stamp. `kind='historical'` marks a table that records what past
    changes did at the time — a change-log, not a current-state measurement. Re-running cannot
    reproduce it, so the staleness check skips it by design rather than by omission."""
    lines = [
        "<!-- cheatah-bench-stamp v1",
        f"     suite:        {suite}",
        f"     generated:    {time.strftime('%Y-%m-%d')}",
        f"     commit:       {commit()}",
        f"     gpu:          {gpu()}",
        f"     host:         {host()}",
        f"     competitors:  {competitors}",
        f"     harness:      {harness}",
        f"     statistic:    {statistic}",
        f"     watch:        {watch}",
        f"     publishable:  {publishable}",
    ]
    if kind:
        lines.insert(1, f"     kind:         {kind}")
    lines += ["", "     PRODUCED BY:", f"       {produced_by}", "-->", ""]
    return "\n".join(lines) + "\n"


def write_region(path, stamp_text, table):
    with open(path, "w") as f:
        f.write(stamp_text + table.rstrip("\n") + "\n")
    print(f"wrote {path}")


def gb_medians(binary, filt, reps=5, min_time="0.15s"):
    """Run a Google Benchmark binary and return {case_name: median_real_time}.

    Repetitions AND interleaving, both deliberately. GPU clocks ramp and throttle across a
    run, so measuring every size as its own consecutive block puts the sizes that ran late on
    hotter, slower silicon — which reads as a shape in the data that is really a shape in the
    schedule. Interleaving scatters the repetitions; the median absorbs what is left.
    """
    out = subprocess.run(
        f"{binary} --benchmark_filter={filt!r} --benchmark_min_time={min_time} "
        f"--benchmark_repetitions={reps} --benchmark_enable_random_interleaving=true "
        f"--benchmark_report_aggregates_only=true --benchmark_format=json",
        shell=True, check=True, capture_output=True, text=True).stdout
    med = {}
    for b in json.loads(out)["benchmarks"]:
        if b.get("aggregate_name") != "median":
            continue
        name = b["run_name"]
        med[name] = b["real_time"]
    return med
