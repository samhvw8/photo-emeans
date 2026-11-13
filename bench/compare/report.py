"""Fill the generated blocks of RESULTS.md from results/<host>/table.md and embed.md.

  python3 report.py

A block is the text between `<!-- BEGIN <host>:<part> -->` and `<!-- END <host>:<part> -->`.
Parts: matrices (all metric matrices of table.md), long (the per-run table), embed.
Everything outside the blocks is hand-written and kept.
"""

import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent


def part(host, name):
    if name == "embed":
        f = HERE / "results" / host / "embed.md"
        return f.read_text() if f.exists() else "(not run yet)"
    t = (HERE / "results" / host / "table.md").read_text()
    head, _, rest = t.partition("\nTrain s, median")
    return head if name == "long" else "Train s, median" + rest


def main():
    p = HERE / "RESULTS.md"
    s = p.read_text()

    def fill(m):
        host, name = m.group(1), m.group(2)
        return f"<!-- BEGIN {host}:{name} -->\n{part(host, name).strip()}\n<!-- END {host}:{name} -->"

    s = re.sub(r"<!-- BEGIN (\w+):(\w+) -->.*?<!-- END \1:\2 -->", fill, s, flags=re.S)
    p.write_text(s)


if __name__ == "__main__":
    main()
