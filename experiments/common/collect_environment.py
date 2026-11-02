#!/usr/bin/env python3
import json
import platform
import subprocess
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]

info = {
    "os": platform.platform(),
    "python": platform.python_version(),
    "cpu": platform.processor() or None,
    "ram_gb": None,
    "container_memory_max": None,
    "container_memory_peak_bytes": None,
    "postgresql_native": None,
    "postgresql_postgresml": None,
    "pgml_extension_commit": None,
    "library_versions": {},
    "threads_note": "experiment scripts generally pin n_jobs/thread_count=1 where applicable",
    "repeats_policy": "single_run unless file documents warmup+5",
}
try:
    import psutil

    info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
except Exception as e:
    info["ram_gb_error"] = str(e)

for mod in ("numpy", "sklearn", "lightgbm", "catboost", "duckdb", "psycopg2"):
    try:
        m = __import__(mod if mod != "sklearn" else "sklearn")
        info["library_versions"][mod] = getattr(m, "__version__", None)
    except Exception:
        info["library_versions"][mod] = None

try:
    out = subprocess.check_output(
        [
            "docker",
            "exec",
            "pgml",
            "sh",
            "-c",
            "cat /sys/fs/cgroup/memory.max; cat /sys/fs/cgroup/memory.peak; postgres --version",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )
    lines = [x.strip() for x in out.splitlines() if x.strip()]
    info["container_memory_max"] = lines[0] if lines else None
    if len(lines) > 1 and lines[1].isdigit():
        info["container_memory_peak_bytes"] = int(lines[1])
    info["postgresql_native"] = next((l for l in lines if "PostgreSQL" in l or l.startswith("postgres")), None)
except Exception as e:
    info["docker_native_error"] = str(e)

try:
    out = subprocess.check_output(
        [
            "docker",
            "exec",
            "postgresml",
            "psql",
            "-U",
            "postgresml",
            "-d",
            "postgresml",
            "-tAc",
            "SELECT version(); SELECT extversion FROM pg_extension WHERE extname='pgml';",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )
    info["postgresql_postgresml"] = out.strip()
except Exception as e:
    info["docker_pml_error"] = str(e)

try:
    info["pgml_extension_commit"] = subprocess.check_output(
        ["git", "-C", str(_REPO), "rev-parse", "HEAD"], text=True
    ).strip()
except Exception:
    info["pgml_extension_commit"] = None

out = Path(_REPO.joinpath('results', 'environment.json'))
out.write_text(json.dumps(info, indent=2), encoding="utf-8")
print(json.dumps(info, indent=2))
