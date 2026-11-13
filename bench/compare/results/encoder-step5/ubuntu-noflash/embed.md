| impl | device | dtype | batch | images/s | load s | peak RSS GiB | peak footprint / GPU GiB | cos vs ST f32 (min) | cos vs CPU ref (min) |
|---|---|---|---|---|---|---|---|---|---|
| candle | cuda | bf16 | 1 | 3.28 | 2.8 | 1.02 | 1.60 | 0.999627 | 0.999627 |
| candle | cuda | bf16 | 8 | 3.42 | 2.8 | 1.02 | 3.04 | 0.999615 | 0.999614 |
| sentence-transformers | cuda | f32 | 1 | 2.35 | 8.9 | 3.27 | 1.99 | 1.000000 | 1.000000 |
| sentence-transformers | cuda | f32 | 8 | 2.93 | 9.6 | 3.27 | 3.26 | 1.000000 | 1.000000 |
