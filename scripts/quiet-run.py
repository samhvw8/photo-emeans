#!/usr/bin/env python3
"""Run a heavy job on the shared Mac only while the user is not using the CPU.

    scripts/quiet-run.py [--max-busy 40] [--lock /tmp/emeans-mac-gpu.lock] [--nice 15] -- CMD [ARGS...]

Waits until CPU busy (user + sys, from `top`) has stayed at or below --max-busy percent for
--quiet-checks samples in a row, then takes the lock file (flock(2), the same lock as
/tmp/emeans-flock.py) if given, and runs CMD at the given nice level. Exits with CMD's exit code.
The load average is not used: on macOS it also counts threads waiting on disk.
"""
import argparse
import fcntl
import os
import re
import subprocess
import sys
import time


def cpu_busy() -> float:
    """Percent of CPU in user + sys over a 1 s sample (the second `top` sample)."""
    out = subprocess.run(["top", "-l", "2", "-n", "0", "-s", "1"], capture_output=True, text=True, check=True).stdout
    lines = [l for l in out.splitlines() if l.startswith("CPU usage")]
    m = re.search(r"([\d.]+)% user, ([\d.]+)% sys", lines[-1])
    return float(m.group(1)) + float(m.group(2))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-busy", type=float, default=40.0, help="CPU busy percent at or below which the Mac counts as quiet")
    p.add_argument("--quiet-checks", type=int, default=3, help="quiet samples in a row before starting")
    p.add_argument("--wait-secs", type=float, default=30.0, help="pause between checks while busy")
    p.add_argument("--lock", help="lock file to hold while CMD runs (e.g. /tmp/emeans-mac-gpu.lock)")
    p.add_argument("--nice", type=int, default=15)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        p.error("no command given")
    quiet = 0
    while quiet < a.quiet_checks:
        busy = cpu_busy()
        quiet = quiet + 1 if busy <= a.max_busy else 0
        if quiet < a.quiet_checks:
            print(f"quiet-run: CPU busy {busy:.0f}% (limit {a.max_busy:.0f}%), waiting", file=sys.stderr, flush=True)
            time.sleep(a.wait_secs if quiet == 0 else 2)
    if a.lock:
        fd = os.open(a.lock, os.O_RDWR | os.O_CREAT, 0o666)
        fcntl.flock(fd, fcntl.LOCK_EX)
    return subprocess.call(["nice", "-n", str(a.nice), *cmd])


if __name__ == "__main__":
    sys.exit(main())
