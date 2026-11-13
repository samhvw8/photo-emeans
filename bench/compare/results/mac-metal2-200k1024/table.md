| n | k | impl | objective mean ± sd | train s median (IQR) | load s | peak RSS GiB | peak GPU GiB (Linux) / footprint GiB (macOS) | ARI vs py-numpy | runs |
|---|---|---|---|---|---|---|---|---|---|
| 200,000 | 1024 | rs-cpu | 0.284947 ± 0.000321 | 1.13 (0.08) | 0.15 | 1.31 | 1.30 | - | 5/5 |
| 200,000 | 1024 | rs-metal-fused | 0.284947 ± 0.000321 | 0.74 (0.02) | 0.15 | 1.25 | 1.24 | - | 5/5 |

Train s, median of 5 seeds:

| n | k | rs-cpu | rs-metal-fused |
|---|---|---|---|
| 200,000 | 1024 | 1.13 | 0.74 |

Objective, mean of 5 seeds:

| n | k | rs-cpu | rs-metal-fused |
|---|---|---|---|
| 200,000 | 1024 | 0.28495 | 0.28495 |

ARI vs py-numpy (same seed), mean of 5 seeds:

| n | k | rs-cpu | rs-metal-fused |
|---|---|---|---|
| 200,000 | 1024 | - | - |

ARI noise floor (py-numpy seed s vs seed s+1, mean of 4 pairs):

| n | k | ARI |
|---|---|---|

Peak memory GiB, RSS / GPU (Linux, nvidia-smi) or footprint (macOS):

| n | k | rs-cpu | rs-metal-fused |
|---|---|---|---|
| 200,000 | 1024 | 1.31 / 1.30 | 1.25 / 1.24 |
