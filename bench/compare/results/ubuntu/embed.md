| impl | device | dtype | batch | images/s | load s | peak RSS GiB | peak footprint / GPU GiB | cos vs ST f32 (min) | cos vs CPU ref (min) |
|---|---|---|---|---|---|---|---|---|---|
| candle | cuda | bf16 | 1 | 3.80 | 1.4 | 1.02 | 1.60 | 0.999627 | 0.999627 |
| candle | cuda | bf16 | 8 | 3.87 | 1.5 | 1.02 | 3.35 | 0.999615 | 0.999614 |
| candle | cuda | f32 | 1 | 3.24 | 2.5 | 1.03 | 2.03 | 1.000000 | 1.000000 |
| candle | cuda | f32 | 8 | 3.28 | 1.4 | 1.02 | 3.06 | 1.000000 | 1.000000 |
| sentence-transformers | cuda | bf16 | 1 | 5.84 | 5.9 | 2.10 | 1.08 | 0.999791 | 0.999791 |
| sentence-transformers | cuda | bf16 | 8 | 9.27 | 6.3 | 2.24 | 1.85 | 0.999847 | 0.999847 |
| sentence-transformers | cuda | f32 | 1 | 2.85 | 10.6 | 3.32 | 1.99 | 1.000000 | 1.000000 |
| sentence-transformers | cuda | f32 | 8 | 3.17 | 6.3 | 3.29 | 3.26 | 1.000000 | 1.000000 |
