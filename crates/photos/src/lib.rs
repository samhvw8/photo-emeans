//! PhotoKit access for emeans: authorization, listing image assets, fetching resized JPEG copies
//! (including photos stored only in iCloud), a resumable fetch pool that writes
//! `photos/*.jpg` + `photos/manifest.jsonl`, and gated album creation and renaming.
//!
//! Read-only by default. The only functions that write to the Photos library are
//! [`albums::create_albums`] and [`albums::rename_collections`] (titles only), and both refuse to
//! run unless `confirm` is `true`.

pub mod albums;
pub mod assets;
pub mod auth;
pub mod fetch;
pub mod stop;
pub mod store;
pub mod video;
pub mod video_store;

pub use albums::{
    AlbumRequest, CollectionInfo, FolderListing, Rename, create_albums, find_folder,
    rename_collections,
};
pub use assets::{AssetInfo, count_shared_album_photos, list_photos, list_videos};
pub use auth::{Access, authorize, ensure_authorized};
pub use fetch::{FetchConfig, Fetched, fetch_resized, is_local};
pub use store::{FetchAllConfig, FetchEvent, FetchReport, ManifestLine, fetch_all, load_manifest};
pub use video::{
    Frame, Rendition, VideoConfig, VideoFrames, fetch_video_frames, frames_from_file,
    video_is_local,
};
pub use video_store::{
    FrameLine, VideoFailure, VideoFetchAllConfig, VideoFetchEvent, VideoFetchReport, VideoLine,
    fetch_all_videos, load_video_manifest, read_video_failures,
};
