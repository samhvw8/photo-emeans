| runtime | kernel | b | k | ms (median of 7) | GFLOP/s | first pass ms (incl. JIT) | mismatches | non-tie | max gap | max score err |
|---|---|---|---|---|---|---|---|---|---|---|
| metal | MmaF32 | 4096 | 64 | 1.36 | 296 | 503 | - | - | - | - |
| metal | MmaF32 | 65536 | 64 | 5.24 | 1230 | 13 | - | - | - | - |
| metal | MmaF32 | all (200000) | 64 | 14.80 | 1328 | 149 | - | - | - | - |
| metal | MmaF32 | 4096 | 256 | 1.67 | 963 | 2 | - | - | - | - |
| metal | MmaF32 | 65536 | 256 | 15.55 | 1658 | 17 | - | - | - | - |
| metal | MmaF32 | all (200000) | 256 | 47.42 | 1658 | 47 | - | - | - | - |
| metal | MmaF32 | 4096 | 1024 | 5.07 | 1271 | 9 | - | - | - | - |
| metal | MmaF32 | 65536 | 1024 | 61.87 | 1666 | 63 | - | - | - | - |
| metal | MmaF32 | all (200000) | 1024 | 190.05 | 1655 | 188 | - | - | - | - |
