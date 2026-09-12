"""Step 7L - Stage B live supervisor: keep the live system running.

Launches and auto-restarts three long-running components:
  * microstructure collector  - Bybit WebSocket order-flow/trade flow (--minutes 0 = forever)
  * unified live dashboard    - event model + baseline ensemble, refresh each minute
  * paper trader              - simulated fills with fees/slippage, equity logged

stdout/stderr of each component -> logs/live_<name>.{log,err}; a heartbeat row is
appended to logs/live_heartbeat.csv every 30s.  A component that dies is restarted
(unless it exits cleanly with code 0).

Usage:
    python src/run_live.py                  # all three, forever
    python src/run_live.py --components micro dashboard   # subset
    python src/run_live.py --once           # stop when all components have exited
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LOGS = ROOT / "logs"

COMPONENTS = {
    "micro": ("microstructure.py", ["--symbols", "DOGEUSDT", "BTCUSDT", "XAUTUSDT",
                                    "--depth", "50", "--minutes", "0"]),
    "dashboard": ("event_live.py", ["--live"]),
    "paper": ("paper.py", ["--loop", "--minutes", "100000"]),
}


def spawn(name: str) -> subprocess.Popen:
    LOGS.mkdir(parents=True, exist_ok=True)
    script, args = COMPONENTS[name]
    cmd = [sys.executable, str(SRC / script), *args]
    out = open(LOGS / ("live_%s.log" % name), "ab")
    err = open(LOGS / ("live_%s.err" % name), "ab")
    p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=out, stderr=err)
    print("started %s pid=%s" % (name, p.pid), flush=True)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Live supervisor (Stage B)")
    ap.add_argument("--components", nargs="+", default=list(COMPONENTS))
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    procs = {name: spawn(name) for name in args.components}
    hb = open(LOGS / "live_heartbeat.csv", "a")
    try:
        while True:
            time.sleep(30)
            cells = [time.strftime("%Y-%m-%d %H:%M:%S")]
            for name in list(procs):
                p = procs[name]
                if p.poll() is None:
                    cells.append("%s=alive" % name)
                elif p.returncode != 0:
                    print("%s exited code=%s -> restart" % (name, p.returncode), flush=True)
                    procs[name] = spawn(name)
                    cells.append("%s=restarted" % name)
                else:
                    procs.pop(name)
                    cells.append("%s=done" % name)
            hb.write(",".join(cells) + "\n")
            hb.flush()
            if args.once and not procs:
                break
    except KeyboardInterrupt:
        print("supervisor stopping...", flush=True)
    finally:
        for p in procs.values():
            try:
                p.terminate()
            except Exception:
                pass
    print("supervisor stopped", flush=True)


if __name__ == "__main__":
    main()