| impl | device | dtype | batch | images/s | load s | peak RSS GiB | peak footprint / GPU GiB | cos vs ST f32 (min) | cos vs CPU ref (min) |
|---|---|---|---|---|---|---|---|---|---|
| candle | cuda | bf16 | 1 | 9.42 | 1.4 | 1.02 | 0.82 | - | 0.999819 |
| candle | cuda | bf16 | 8 | 10.31 | 1.4 | 1.02 | 0.98 | - | 0.999816 |
| candle | cuda | f32 | 1 | 3.23 | 1.4 | 1.02 | 2.03 | - | 1.000000 |
| candle | cuda | f32 | 8 | 3.27 | 1.5 | 1.02 | 3.10 | - | 1.000000 |
