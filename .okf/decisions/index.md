# decisions

* [Runtime choice](runtime-choice.md) - candle on Metal, plus a Bend twin.
* [Photo source](photo-source.md) - PhotoKit resized copies and video frames, iCloud included.
* [Output](output.md) - gallery first, albums after review.
* [Data location](data-location.md) - personal data on this Mac and the user's Ubuntu box only.
* [Library split](library-split.md) - done: e-means in github.com/samhvw8/emeans (candle-free core, per-backend crates, algorithms crate, facade), the encoder in github.com/samhvw8/candle-embeddings (renamed from `encoder`); this tool depends on both as git dependencies pinned by commit.
