# design

* [Pipeline](pipeline.md) - steps, data files, model, e-means additions.
* [PhotoKit access](photokit.md) - listing, 1024 px fetch incl. iCloud, video frames and renditions, resume, permissions, albums gate.
* [Bend e-means](bend-emeans.md) - the Bend 2 port: structure, parallel lanes, CPU vs Metal, parity with faiss, Bend limits.
* [e-means algorithm](emeans-algorithm.md) - moved to github.com/samhvw8/emeans/.okf (kept as history): the Rust port of faiss e-means, its Metal kernel, deviations, parity and timing.
* [e-means core](core.md) - moved to github.com/samhvw8/emeans/.okf (kept as history): the op contract every backend implements (CPU, Metal, CUDA), its f64 reference and contract check, the device-resident training loop, which family uses which block.
* [Encoder core](encoder-core.md) - moved to github.com/samhvw8/candle-embeddings/.okf (kept as history): the Encoder trait, shared blocks with the families that call them, the exactness harness.
* [EmbeddingGemma 2 family](embeddinggemma2.md) - moved to github.com/samhvw8/candle-embeddings/.okf (kept as history): the first encoder family: image input sequence, exact preprocessing, towers, dtype, parity and throughput.
* [gemma-embed](gemma-embed.md) - deprecated: the old crate, folded into the encoder core.
* [Automatic k, incremental updates, near-duplicates](autok-incremental-dedup.md) - two-level auto-k (silhouette sweep), stable-id incremental updates, kNN dedup with anti-chaining; validation tables.
* [photo-emeans CLI](cli.md) - commands, crash-safe resumable embed store, video frame store, cluster run files, gallery approval flow with videos, albums dry run.
