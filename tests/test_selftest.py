import subprocess, sys, json
from pathlib import Path

def test_selftest_smoke(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    # Run the packaged "run-all" in selftest mode (quick)
    cmd = [sys.executable, "-m", "egrrc.run_all", "--selftest", "--outdir", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Must produce tables/summary.csv and posthoc.json
    summary = out / "tables" / "summary.csv"
    posthoc  = out / "tables" / "posthoc.json"
    assert summary.exists(), "summary.csv not created"
    assert posthoc.exists(), "posthoc.json not created"
    # posthoc.json should be parseable JSON
    json.loads(posthoc.read_text())
