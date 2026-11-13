| impl | b | k | ms (median of 7) | kernel ms | GFLOP/s (pass) | first pass ms (incl. JIT) | mismatches | non-tie | max gap |
|---|---|---|---|---|---|---|---|---|---|
| triton-f32 | 4096 | 64 | 0.68 | 0.51 | 594 | 1325 | 0 | 0 | 0.00e+00 |
| triton-f32 | 65536 | 64 | 5.07 | 4.78 | 1271 | 9 | 0 | 0 | 0.00e+00 |
| triton-f32 | all (200000) | 64 | 15.38 | 14.35 | 1278 | 18 | 0 | 0 | 0.00e+00 |
| triton-bf16 | 4096 | 64 | fails: OutOfResources: out of resource: shared memory, Required: 131072, Hardware limit: 101376. Reducing block s | | | | | | |
| triton-bf16 | 65536 | 64 | fails: OutOfResources: out of resource: shared memory, Required: 131072, Hardware limit: 101376. Reducing block s | | | | | | |
| triton-bf16 | all (200000) | 64 | fails: OutOfResources: out of resource: shared memory, Required: 131072, Hardware limit: 101376. Reducing block s | | | | | | |
| triton-f32 | 4096 | 256 | 1.17 | 0.89 | 1376 | 1 | 0 | 0 | 0.00e+00 |
| triton-f32 | 65536 | 256 | 9.87 | 9.43 | 2610 | 18 | 0 | 0 | 0.00e+00 |
| triton-f32 | all (200000) | 256 | 29.73 | 28.55 | 2645 | 32 | 0 | 0 | 0.00e+00 |
| triton-bf16 | 4096 | 256 | fails: OutOfResources: out of resource: shared memory, Required: 131072, Hardware limit: 101376. Reducing block s | | | | | | |
| triton-bf16 | 65536 | 256 | fails: OutOfResources: out of resource: shared memory, Required: 131072, Hardware limit: 101376. Reducing block s | | | | | | |
| triton-bf16 | all (200000) | 256 | fails: OutOfResources: out of resource: shared memory, Required: 131072, Hardware limit: 101376. Reducing block s | | | | | | |
| triton-f32 | 4096 | 1024 | 3.66 | 3.14 | 1758 | 5 | 0 | 0 | 0.00e+00 |
| triton-f32 | 65536 | 1024 | 38.02 | 37.28 | 2711 | 40 | 0 | 0 | 0.00e+00 |
| triton-f32 | all (200000) | 1024 | 115.80 | 114.37 | 2717 | 118 | 6 | 0 | 6.27e-07 |
| triton-bf16 | 4096 | 1024 | fails: OutOfResources: out of resource: shared memory, Required: 131072, Hardware limit: 101376. Reducing block s | | | | | | |
| triton-bf16 | 65536 | 1024 | fails: OutOfResources: out of resource: shared memory, Required: 131072, Hardware limit: 101376. Reducing block s | | | | | | |
| triton-bf16 | all (200000) | 1024 | fails: OutOfResources: out of resource: shared memory, Required: 131072, Hardware limit: 101376. Reducing block s | | | | | | |
