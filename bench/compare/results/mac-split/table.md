| n | k | impl | objective mean ± sd | train s median (IQR) | load s | peak RSS GiB | peak GPU GiB (Linux) / footprint GiB (macOS) | ARI vs py-numpy | runs |
|---|---|---|---|---|---|---|---|---|---|
| 200,000 | 1024 | rs-cpu | 0.284892 ± 0.000439 | 1.48 (0.10) | 0.17 | 1.31 | 1.30 | - | 3/3 |
| 200,000 | 1024 | rs-metal-fused | 0.284892 ± 0.000439 | 0.66 (0.01) | 0.17 | 1.24 | 1.24 | - | 3/3 |

Train s, median of 5 seeds:

| n | k | rs-cpu | rs-metal-fused |
|---|---|---|---|
| 200,000 | 1024 | 1.48 | 0.66 |

Objective, mean of 5 seeds:

| n | k | rs-cpu | rs-metal-fused |
|---|---|---|---|
| 200,000 | 1024 | 0.28489 | 0.28489 |

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
| 200,000 | 1024 | 1.31 / 1.30 | 1.24 / 1.24 |
