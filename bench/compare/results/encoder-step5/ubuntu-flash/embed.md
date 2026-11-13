| impl | device | dtype | batch | images/s | load s | peak RSS GiB | peak footprint / GPU GiB | cos vs ST f32 (min) | cos vs CPU ref (min) |
|---|---|---|---|---|---|---|---|---|---|
| candle | cuda | bf16 | 1 | 7.22 | 2.6 | 1.02 | 0.82 | 0.999819 | 0.999819 |
| candle | cuda | bf16 | 8 | 8.27 | 2.2 | 1.02 | 0.98 | 0.999817 | 0.999816 |
| candle | cuda | f32 | 1 | 3.00 | 1.8 | 1.02 | 2.03 | 1.000000 | 1.000000 |
| candle | cuda | f32 | 8 | 3.01 | 2.2 | 1.02 | 3.06 | 1.000000 | 1.000000 |
| sentence-transformers | cuda | bf16 | 1 | 4.06 | 9.7 | 2.04 | 1.08 | 0.999791 | 0.999791 |
| sentence-transformers | cuda | bf16 | 8 | 7.01 | 7.5 | 2.19 | 1.85 | 0.999847 | 0.999847 |
| sentence-transformers | cuda | f32 | 1 | 2.42 | 10.8 | 3.28 | 1.99 | 1.000000 | 1.000000 |
| sentence-transformers | cuda | f32 | 8 | 2.92 | 8.9 | 3.27 | 3.26 | 1.000000 | 1.000000 |
