| impl | device | dtype | batch | images/s | load s | peak RSS GiB | peak footprint / GPU GiB | cos vs ST f32 (min) | cos vs CPU ref (min) |
|---|---|---|---|---|---|---|---|---|---|
| sentence-transformers | cuda | bf16 | 1 | 5.41 | 7.6 | 2.05 | 1.08 | 0.999791 | 0.999791 |
| sentence-transformers | cuda | bf16 | 8 | 9.25 | 5.7 | 2.19 | 1.85 | 0.999847 | 0.999847 |
| sentence-transformers | cuda | f32 | 1 | 3.00 | 9.3 | 3.27 | 1.99 | 1.000000 | 1.000000 |
| sentence-transformers | cuda | f32 | 8 | 3.14 | 10.3 | 3.27 | 3.26 | 1.000000 | 1.000000 |
