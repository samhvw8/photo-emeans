//! Listing image and video assets.

use std::fmt;

use anyhow::Result;
use objc2::msg_send;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::AnyObject;
use objc2_foundation::NSNumber;
use objc2_foundation::{NSArray, NSSortDescriptor, ns_string};
use objc2_photos::{
    PHAsset, PHAssetMediaSubtype, PHAssetMediaType, PHAssetResource, PHAssetResourceType,
    PHAssetSourceType, PHFetchOptions,
};

/// A `PHAsset` handle that may cross threads.
///
/// `PHAsset` is an immutable model object; Apple documents Photos model objects as thread-safe
/// ("Photos model objects are immutable and thread-safe"), so sharing a retained handle is sound.
#[derive(Clone)]
pub(crate) struct AssetHandle(pub(crate) Retained<PHAsset>);
// SAFETY: see the type docs: PHAsset is immutable and documented as thread-safe.
unsafe impl Send for AssetHandle {}
// SAFETY: as above.
unsafe impl Sync for AssetHandle {}

impl fmt::Debug for AssetHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PHAsset")
    }
}

/// One image or video asset of the library.
#[derive(Debug, Clone)]
pub struct AssetInfo {
    /// `true` for a video asset (`PHAssetMediaTypeVideo`).
    pub is_video: bool,
    /// Playback length in seconds (`PHAsset.duration`); 0 for images.
    pub duration: f64,
    /// `PHAsset.localIdentifier`, e.g. `"8F1C...-...-.../L0/001"`.
    pub id: String,
    /// Creation date as ISO 8601 UTC (`2024-05-01T09:30:12Z`), `None` when Photos has none.
    pub created: Option<String>,
    /// Creation date as Unix seconds.
    pub created_unix: Option<f64>,
    /// Original pixel size, as Photos reports it (already in display orientation).
    pub pixel_width: u32,
    pub pixel_height: u32,
    pub favorite: bool,
    /// Set for photos taken in a burst. Only the burst's representative photo is listed.
    pub burst_id: Option<String>,
    /// `mediaSubtypes` as names: `panorama`, `hdr`, `screenshot`, `live`, `depthEffect`, `spatial`;
    /// for videos `streamed`, `highFrameRate`, `timelapse`, `screenRecording`, `cinematic`.
    pub subtypes: Vec<String>,
    pub(crate) handle: AssetHandle,
}

impl AssetInfo {
    /// File stem for this asset: the UUID part of the local identifier.
    pub fn file_stem(&self) -> String {
        file_stem(&self.id)
    }

    /// Uniform type identifier of the original photo resource (e.g. `public.heic`,
    /// `public.jpeg`, `public.png`). Reads resource metadata only; downloads nothing.
    pub fn original_uti(&self) -> Option<String> {
        // SAFETY: class method on a valid asset; getters on returned resources.
        unsafe {
            let res = PHAssetResource::assetResourcesForAsset(&self.handle.0);
            let all: Vec<_> = res.iter().collect();
            all.iter()
                .find(|r| r.r#type() == PHAssetResourceType::Photo)
                .or_else(|| all.first())
                .map(|r| r.contentType().identifier().to_string())
        }
    }

    /// Size in bytes of the original resource (the full-size video or photo), from the
    /// undocumented `PHAssetResource` key `fileSize`. Reads metadata only; downloads nothing.
    /// Used only for estimates: `None` when the key is missing.
    pub fn original_bytes(&self) -> Option<u64> {
        let want = if self.is_video {
            PHAssetResourceType::Video
        } else {
            PHAssetResourceType::Photo
        };
        // SAFETY: class method on a valid asset; `valueForKey:` on an NSObject returns an
        // autoreleased object or nil (an unknown key raises, so only a known key is read).
        unsafe {
            let res = PHAssetResource::assetResourcesForAsset(&self.handle.0);
            let all: Vec<_> = res.iter().collect();
            let r = all
                .iter()
                .find(|r| r.r#type() == want)
                .or_else(|| all.first())?;
            let v: Option<Retained<AnyObject>> =
                msg_send![&**r, valueForKey: ns_string!("fileSize")];
            v?.downcast::<NSNumber>().ok().map(|n| n.as_u64())
        }
    }
}

/// The UUID part of a local identifier (`UUID/L0/001` -> `UUID`), restricted to safe characters.
pub fn file_stem(local_id: &str) -> String {
    let head = local_id.split('/').next().unwrap_or(local_id);
    head.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Names for `PHAssetMediaSubtype` bits. Unknown bits become `bit<N>`.
pub fn subtype_names(bits: usize) -> Vec<String> {
    const NAMES: [(usize, &str); 11] = [
        (PHAssetMediaSubtype::PhotoPanorama.0, "panorama"),
        (PHAssetMediaSubtype::PhotoHDR.0, "hdr"),
        (PHAssetMediaSubtype::PhotoScreenshot.0, "screenshot"),
        (PHAssetMediaSubtype::PhotoLive.0, "live"),
        (PHAssetMediaSubtype::PhotoDepthEffect.0, "depthEffect"),
        (PHAssetMediaSubtype::SpatialMedia.0, "spatial"),
        (PHAssetMediaSubtype::VideoStreamed.0, "streamed"),
        (PHAssetMediaSubtype::VideoHighFrameRate.0, "highFrameRate"),
        (PHAssetMediaSubtype::VideoTimelapse.0, "timelapse"),
        (
            PHAssetMediaSubtype::VideoScreenRecording.0,
            "screenRecording",
        ),
        (PHAssetMediaSubtype::VideoCinematic.0, "cinematic"),
    ];
    let mut out = Vec::new();
    let mut rest = bits;
    for (bit, name) in NAMES {
        if bits & bit != 0 {
            out.push(name.to_string());
            rest &= !bit;
        }
    }
    for i in 0..usize::BITS {
        if rest & (1 << i) != 0 {
            out.push(format!("bit{i}"));
        }
    }
    out
}

/// Unix seconds to ISO 8601 UTC with second precision.
pub fn iso8601_utc(unix: f64) -> String {
    let secs = unix.floor() as i64;
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    )
}

fn fetch_options(sources: PHAssetSourceType) -> Retained<PHFetchOptions> {
    // SAFETY: plain property setters on a fresh object.
    unsafe {
        let opts = PHFetchOptions::new();
        let sort = NSSortDescriptor::sortDescriptorWithKey_ascending(
            Some(ns_string!("creationDate")),
            true,
        );
        opts.setSortDescriptors(Some(&NSArray::from_retained_slice(&[sort])));
        // Hidden assets stay out (the default, set explicitly to make it visible here).
        opts.setIncludeHiddenAssets(false);
        // One representative per burst (the default).
        opts.setIncludeAllBurstAssets(false);
        opts.setIncludeAssetSourceTypes(sources);
        opts
    }
}

/// All image assets of the user's library, oldest first.
///
/// Includes the iCloud Shared Library (its assets have source type `UserLibrary`). Excludes
/// hidden photos, non-representative burst frames, and iCloud Shared Album photos
/// (`CloudShared`): those are other people's albums, not part of the library.
pub fn list_photos() -> Result<Vec<AssetInfo>> {
    list_with_sources(PHAssetMediaType::Image, PHAssetSourceType::TypeUserLibrary)
}

/// All video assets of the user's library, oldest first. Same sources and exclusions as
/// [`list_photos`] (hidden videos and iCloud Shared Album videos stay out).
pub fn list_videos() -> Result<Vec<AssetInfo>> {
    list_with_sources(PHAssetMediaType::Video, PHAssetSourceType::TypeUserLibrary)
}

/// Number of image assets in iCloud Shared Albums (not listed by [`list_photos`]).
pub fn count_shared_album_photos() -> usize {
    let opts = fetch_options(PHAssetSourceType::TypeCloudShared);
    // SAFETY: class method with valid arguments.
    unsafe {
        PHAsset::fetchAssetsWithMediaType_options(PHAssetMediaType::Image, Some(&opts)).count()
    }
}

fn list_with_sources(
    media: PHAssetMediaType,
    sources: PHAssetSourceType,
) -> Result<Vec<AssetInfo>> {
    let opts = fetch_options(sources);
    // SAFETY: class method with valid arguments.
    let result = unsafe { PHAsset::fetchAssetsWithMediaType_options(media, Some(&opts)) };
    let n = unsafe { result.count() };
    let mut out = Vec::with_capacity(n);
    for start in (0..n).step_by(512) {
        autoreleasepool(|_| {
            for i in start..(start + 512).min(n) {
                // SAFETY: index is in range.
                let a = unsafe { result.objectAtIndex(i) };
                out.push(describe(a));
            }
        });
    }
    Ok(out)
}

fn describe(a: Retained<PHAsset>) -> AssetInfo {
    // SAFETY: property getters on a valid PHAsset.
    unsafe {
        let created_unix = a.creationDate().map(|d| d.timeIntervalSince1970());
        AssetInfo {
            is_video: a.mediaType() == PHAssetMediaType::Video,
            duration: a.duration(),
            id: a.localIdentifier().to_string(),
            created: created_unix.map(iso8601_utc),
            created_unix,
            pixel_width: a.pixelWidth() as u32,
            pixel_height: a.pixelHeight() as u32,
            favorite: a.isFavorite(),
            burst_id: a.burstIdentifier().map(|s| s.to_string()),
            subtypes: subtype_names(a.mediaSubtypes().0),
            handle: AssetHandle(a),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_dates() {
        assert_eq!(iso8601_utc(0.0), "1970-01-01T00:00:00Z");
        assert_eq!(iso8601_utc(951_782_400.0), "2000-02-29T00:00:00Z");
        assert_eq!(iso8601_utc(1_714_555_812.7), "2024-05-01T09:30:12Z");
        assert_eq!(iso8601_utc(-1.0), "1969-12-31T23:59:59Z");
    }

    #[test]
    fn subtypes() {
        assert!(subtype_names(0).is_empty());
        assert_eq!(subtype_names(1 << 2 | 1 << 3), vec!["screenshot", "live"]);
        assert_eq!(subtype_names(1 << 1 | 1 << 7), vec!["hdr", "bit7"]);
        assert_eq!(
            subtype_names(1 << 17 | 1 << 19),
            vec!["highFrameRate", "screenRecording"]
        );
    }

    #[test]
    fn stems() {
        assert_eq!(
            file_stem("8F1C2A3B-0000-4000-8000-ABCDEFABCDEF/L0/001"),
            "8F1C2A3B-0000-4000-8000-ABCDEFABCDEF"
        );
        assert_eq!(file_stem("a b.c/L0/001"), "a_b_c");
    }
}
