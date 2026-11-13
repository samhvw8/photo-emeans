| impl | device | dtype | batch | images/s | load s | peak RSS GiB | peak footprint / GPU GiB | cos vs ST f32 (min) | cos vs CPU ref (min) |
|---|---|---|---|---|---|---|---|---|---|
| candle | cuda | bf16 | 1 | 3.80 | 1.4 | 1.02 | 1.60 | - | 0.999627 |
| candle | cuda | bf16 | 8 | 3.86 | 1.5 | 1.02 | 3.35 | - | 0.999614 |
