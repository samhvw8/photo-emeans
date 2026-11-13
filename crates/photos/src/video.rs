//! Frames of one video asset as resized JPEGs.
//!
//! 1. The poster frame (the frame Photos shows for the video) comes from
//!    [`crate::fetch::fetch_resized`]: `requestImage` works on video assets too.
//! 2. `PHImageManager requestAVAssetForVideo:options:resultHandler:` returns an `AVAsset` for a
//!    smaller rendition (`deliveryMode = MediumQualityFormat` by default, `version = Current`).
//!    Like photos, it asks with network off first and with network on only when Photos answers that
//!    the rendition is in iCloud. Photos downloads into its own cache; this crate never writes a
//!    video file.
//! 3. `AVAssetImageGenerator` (`appliesPreferredTrackTransform`, `maximumSize` = the long side, a
//!    time tolerance of a quarter of the frame spacing, SDR output) decodes the sampled frames in one
//!    batch call (`generateCGImagesAsynchronouslyForTimes`).
//! 4. Each frame is encoded with the same 8-bit sRGB JPEG path as photos.
//!
//! Results of both asynchronous calls may arrive on the main queue, so the caller runs the main
//! run loop or calls from a worker while the main thread runs it (see [`crate::fetch_all_videos`]).

use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use block2::RcBlock;
use objc2::Message;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyObject, Bool};
use objc2_av_foundation::{
    AVAsset, AVAssetImageGenerator, AVAssetImageGeneratorResult, AVAudioMix, AVURLAsset,
    NSValueAVFoundationExtensions,
};
use objc2_core_foundation::CGSize;
use objc2_core_graphics::CGImage;
use objc2_core_media::CMTime;
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSValue};
use objc2_photos::{
    PHImageCancelledKey, PHImageErrorKey, PHImageManager, PHImageResultIsInCloudKey, PHPhotosError,
    PHVideoRequestOptions, PHVideoRequestOptionsDeliveryMode, PHVideoRequestOptionsVersion,
};

use crate::assets::AssetInfo;
use crate::fetch::{FetchConfig, encode_jpeg, fetch_resized, is_main_thread, pump_main_run_loop};

const NEEDS_NETWORK: &str = "needs an iCloud download";

/// Which rendition of the video Photos should hand out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rendition {
    /// `PHVideoRequestOptionsDeliveryModeMediumQualityFormat` (the default).
    Medium,
    /// `PHVideoRequestOptionsDeliveryModeFastFormat`: the smallest Photos has.
    Fast,
}

/// Options for [`fetch_video_frames`].
#[derive(Debug, Clone)]
pub struct VideoConfig {
    /// Poster request (size, quality, timeouts, attempts). Its `max_side` and `quality` also
    /// apply to the sampled frames.
    pub image: FetchConfig,
    /// Frames per video, poster included, for clips of at least `full_secs`.
    pub frames: usize,
    /// Clips shorter than this get fewer frames: two per second, at least two.
    pub full_secs: f64,
    pub rendition: Rendition,
    /// Videos longer than this many seconds use [`Rendition::Fast`] (smaller download, 360 px
    /// short side for phone video). `None`: `rendition` for every video.
    pub fast_over: Option<f64>,
    /// Give up on the video request (and the download) after this long without progress.
    pub idle_timeout: Duration,
    /// Give up on the video request after this long, progress or not.
    pub hard_timeout: Duration,
    /// Give up on frame decoding after this long.
    pub decode_timeout: Duration,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            image: FetchConfig::default(),
            frames: 8,
            full_secs: 4.0,
            rendition: Rendition::Medium,
            fast_over: None,
            idle_timeout: Duration::from_secs(90),
            hard_timeout: Duration::from_secs(900),
            decode_timeout: Duration::from_secs(120),
        }
    }
}

/// One encoded frame.
#[derive(Debug, Clone)]
pub struct Frame {
    pub jpeg: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Time in the video in seconds; `None` for the poster frame.
    pub t: Option<f64>,
}

/// The frames of one video: the poster first, then the sampled frames in time order.
#[derive(Debug, Clone)]
pub struct VideoFrames {
    pub frames: Vec<Frame>,
    /// Photos had to download the rendition (or the poster) from iCloud.
    pub from_icloud: bool,
    /// Size of the rendition file Photos handed out, when it is a local file.
    pub rendition_bytes: Option<u64>,
    /// Time spent getting the rendition (download included) and decoding the frames.
    pub request_time: Duration,
    pub decode_time: Duration,
}

/// Frame count for a clip: `frames` from `full_secs` on, else two per second (at least two).
pub fn frame_count(duration: f64, frames: usize, full_secs: f64) -> usize {
    let frames = frames.max(2);
    if !duration.is_finite() || duration <= 0.0 {
        return 2;
    }
    if duration >= full_secs {
        return frames;
    }
    ((duration * 2.0).round() as usize).clamp(2, frames)
}

/// Sample times for `n` frames (the poster is extra): the middles of `n` equal slices.
pub fn sample_times(duration: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| duration * (i as f64 + 0.5) / n as f64)
        .collect()
}

/// An `AVAsset` that may cross threads. AVFoundation documents `AVAsset` as immutable and safe to
/// use from any thread once its properties are loaded; we only hand it to a generator.
struct AssetBox(Retained<AVAsset>);
// SAFETY: see the type docs.
unsafe impl Send for AssetBox {}

#[derive(Default)]
struct ReqState {
    result: Option<std::result::Result<AssetBox, String>>,
    last_activity: Option<Instant>,
    progress_error: Option<String>,
}

struct Shared<S> {
    state: Mutex<S>,
    cv: Condvar,
}

fn info_get(info: &NSDictionary, key: &NSString) -> Option<Retained<AnyObject>> {
    let key: &AnyObject = key.as_ref();
    info.objectForKey(key)
}

fn info_bool(info: &NSDictionary, key: &NSString) -> bool {
    info_get(info, key)
        .and_then(|v| v.downcast::<NSNumber>().ok())
        .is_some_and(|n| n.as_bool())
}

fn describe_error(e: &NSError) -> String {
    format!(
        "{} (domain {}, code {})",
        e.localizedDescription(),
        e.domain(),
        e.code()
    )
}

/// Wait for `done(state)` while running the main run loop when on the main thread.
fn wait_for<S, T>(
    shared: &Shared<S>,
    started: Instant,
    hard: Duration,
    mut take: impl FnMut(&mut S) -> Option<T>,
    idle: impl Fn(&S) -> Option<Duration>,
) -> std::result::Result<T, String> {
    let on_main = is_main_thread();
    let mut st = shared.state.lock().unwrap();
    loop {
        if let Some(r) = take(&mut st) {
            return Ok(r);
        }
        if let Some(d) = idle(&st) {
            return Err(format!("{:.0}s without progress", d.as_secs_f64()));
        }
        if started.elapsed() > hard {
            return Err(format!(
                "no result after {:.0}s",
                started.elapsed().as_secs_f64()
            ));
        }
        if on_main {
            drop(st);
            pump_main_run_loop(Duration::from_millis(50));
            st = shared.state.lock().unwrap();
        } else {
            st = shared
                .cv
                .wait_timeout(st, Duration::from_millis(500))
                .unwrap()
                .0;
        }
    }
}

/// One `requestAVAssetForVideo` call.
fn request_av_asset(
    asset: &AssetInfo,
    cfg: &VideoConfig,
    network: bool,
) -> std::result::Result<Retained<AVAsset>, String> {
    let rendition = match cfg.fast_over {
        Some(secs) if asset.duration > secs => Rendition::Fast,
        _ => cfg.rendition,
    };
    let shared = Arc::new(Shared {
        state: Mutex::new(ReqState {
            last_activity: Some(Instant::now()),
            ..ReqState::default()
        }),
        cv: Condvar::new(),
    });
    // SAFETY: property setters on a fresh object; the progress block is copied by the setter.
    let opts = unsafe {
        let o = PHVideoRequestOptions::new();
        o.setDeliveryMode(match rendition {
            Rendition::Medium => PHVideoRequestOptionsDeliveryMode::MediumQualityFormat,
            Rendition::Fast => PHVideoRequestOptionsDeliveryMode::FastFormat,
        });
        o.setVersion(PHVideoRequestOptionsVersion::Current);
        o.setNetworkAccessAllowed(network);
        let ps = shared.clone();
        let progress = RcBlock::new(
            move |_p: f64, err: *mut NSError, _stop: NonNull<Bool>, _info: *mut NSDictionary| {
                let mut st = ps.state.lock().unwrap();
                st.last_activity = Some(Instant::now());
                if let Some(e) = err.as_ref() {
                    st.progress_error = Some(describe_error(e));
                }
            },
        );
        o.setProgressHandler(RcBlock::as_ptr(&progress));
        o
    };
    let hs = shared.clone();
    let handler = RcBlock::new(
        move |av: *mut AVAsset, _mix: *mut AVAudioMix, info: *mut NSDictionary| {
            // SAFETY: Photos passes a valid AVAsset or nil, and a valid dictionary or nil.
            let (av, info) = unsafe { (av.as_ref(), info.as_ref()) };
            let outcome = autoreleasepool(|_| {
                // SAFETY: reading the documented Photos info keys.
                let (cancelled, in_cloud, err) = unsafe {
                    match info {
                        Some(i) => (
                            info_bool(i, PHImageCancelledKey),
                            info_bool(i, PHImageResultIsInCloudKey),
                            info_get(i, PHImageErrorKey).and_then(|e| e.downcast::<NSError>().ok()),
                        ),
                        None => (false, false, None),
                    }
                };
                if cancelled {
                    return Err("request cancelled".to_string());
                }
                let network_required = err.as_ref().is_some_and(|e| {
                    e.domain().to_string() == "PHPhotosErrorDomain"
                        && e.code() == PHPhotosError::NetworkAccessRequired.0
                });
                if network_required || (in_cloud && av.is_none()) {
                    return Err(NEEDS_NETWORK.to_string());
                }
                if let Some(e) = err {
                    return Err(format!("Photos error: {}", describe_error(&e)));
                }
                match av {
                    Some(a) => Ok(AssetBox(a.retain())),
                    None => Err("Photos returned no video and no error".to_string()),
                }
            });
            let mut st = hs.state.lock().unwrap();
            if st.result.is_none() {
                st.result = Some(outcome);
            }
            hs.cv.notify_all();
        },
    );
    // SAFETY: valid asset, options and block; PhotoKit copies the block.
    let manager = unsafe { PHImageManager::defaultManager() };
    let req = unsafe {
        manager.requestAVAssetForVideo_options_resultHandler(&asset.handle.0, Some(&opts), &handler)
    };
    let started = Instant::now();
    let idle_limit = cfg.idle_timeout;
    let r = wait_for(
        &shared,
        started,
        cfg.hard_timeout,
        |st| st.result.take(),
        |st| {
            st.last_activity
                .map(|t| t.elapsed())
                .filter(|d| *d > idle_limit)
        },
    );
    match r {
        Ok(Ok(a)) => Ok(a.0),
        Ok(Err(e)) => {
            let pe = shared.state.lock().unwrap().progress_error.take();
            Err(match pe {
                Some(pe) => format!("{e}; download error: {pe}"),
                None => e,
            })
        }
        Err(e) => {
            // SAFETY: cancelling our own request id.
            unsafe { manager.cancelImageRequest(req) };
            Err(format!(
                "video request timed out ({e}; network {})",
                if network { "on" } else { "off" }
            ))
        }
    }
}

/// `true` when Photos has the requested rendition of `asset` on this Mac (a network-off
/// `requestAVAssetForVideo`). Downloads nothing.
pub fn video_is_local(asset: &AssetInfo, cfg: &VideoConfig) -> Result<bool> {
    match request_av_asset(asset, cfg, false) {
        Ok(_) => Ok(true),
        Err(e) if e.starts_with(NEEDS_NETWORK) => Ok(false),
        Err(e) => bail!("local check failed for {}: {e}", asset.id),
    }
}

/// Size of the file behind an `AVURLAsset` with a `file:` URL.
fn local_file_bytes(av: &AVAsset) -> Option<u64> {
    let url = av.downcast_ref::<AVURLAsset>()?;
    // SAFETY: getter on a valid AVURLAsset.
    let url = unsafe { url.URL() };
    if !url.isFileURL() {
        return None;
    }
    let path = url.path()?.to_string();
    std::fs::metadata(path).ok().map(|m| m.len())
}

/// `true` when the asset Photos handed out is a remote URL (a stream), not a local file.
fn is_remote(av: &AVAsset) -> bool {
    av.downcast_ref::<AVURLAsset>()
        // SAFETY: getter on a valid AVURLAsset.
        .is_some_and(|u| !unsafe { u.URL() }.isFileURL())
}

struct CgBox(objc2_core_foundation::CFRetained<CGImage>);
// SAFETY: CGImage is an immutable CoreFoundation object; CF types are thread-safe to retain,
// release and read.
unsafe impl Send for CgBox {}

#[derive(Default)]
struct GenState {
    frames: Vec<Option<std::result::Result<(CgBox, f64), String>>>,
    done: usize,
}

/// Decode frames at `times` (seconds) with one batch call.
fn decode_frames(
    av: &AVAsset,
    times: &[f64],
    max_side: u32,
    timeout: Duration,
) -> std::result::Result<Vec<(objc2_core_foundation::CFRetained<CGImage>, f64)>, String> {
    // SAFETY: plain setters on a fresh generator for a valid asset.
    let generator = unsafe {
        let g = AVAssetImageGenerator::assetImageGeneratorWithAsset(av);
        g.setAppliesPreferredTrackTransform(true);
        g.setMaximumSize(CGSize::new(f64::from(max_side), f64::from(max_side)));
        let spacing = if times.len() > 1 {
            times[1] - times[0]
        } else {
            1.0
        };
        let tol = CMTime::with_seconds((spacing / 4.0).clamp(0.02, 1.0), 600);
        g.setRequestedTimeToleranceBefore(tol);
        g.setRequestedTimeToleranceAfter(tol);
        g
    };
    let shared = Arc::new(Shared {
        state: Mutex::new(GenState {
            frames: (0..times.len()).map(|_| None).collect(),
            done: 0,
        }),
        cv: Condvar::new(),
    });
    let want: Vec<f64> = times.to_vec();
    let hs = shared.clone();
    let handler = RcBlock::new(
        move |requested: CMTime,
              image: *mut CGImage,
              actual: CMTime,
              result: AVAssetImageGeneratorResult,
              err: *mut NSError| {
            // SAFETY: CMTime getters are pure; AVFoundation passes a valid image or null and a
            // valid error or nil. The image is not retained for us, so retain it here.
            let req_s = unsafe { requested.seconds() };
            let act_s = unsafe { actual.seconds() };
            let outcome = if result == AVAssetImageGeneratorResult::Succeeded {
                match NonNull::new(image) {
                    // SAFETY: a valid CGImage pointer; retain adds our own reference.
                    Some(p) => Ok((
                        CgBox(unsafe { objc2_core_foundation::CFRetained::retain(p) }),
                        act_s,
                    )),
                    None => Err("no image".to_string()),
                }
            } else {
                // SAFETY: valid error or nil.
                let e = unsafe { err.as_ref() }.map(describe_error);
                Err(format!(
                    "frame at {req_s:.2}s {}: {}",
                    if result == AVAssetImageGeneratorResult::Cancelled {
                        "cancelled"
                    } else {
                        "failed"
                    },
                    e.unwrap_or_default()
                ))
            };
            let i = want
                .iter()
                .enumerate()
                .min_by(|a, b| (a.1 - req_s).abs().total_cmp(&(b.1 - req_s).abs()))
                .map_or(0, |(i, _)| i);
            let mut st = hs.state.lock().unwrap();
            if st.frames[i].is_none() {
                st.frames[i] = Some(outcome);
                st.done += 1;
            }
            hs.cv.notify_all();
        },
    );
    let values: Vec<Retained<NSValue>> = times
        .iter()
        // SAFETY: plain constructors.
        .map(|&t| unsafe { NSValue::valueWithCMTime(CMTime::with_seconds(t, 600)) })
        .collect();
    let arr = NSArray::from_retained_slice(&values);
    // SAFETY: valid array and block; AVFoundation copies the block.
    unsafe {
        generator.generateCGImagesAsynchronouslyForTimes_completionHandler(
            &arr,
            RcBlock::as_ptr(&handler),
        )
    };
    let n = times.len();
    let r = wait_for(
        &shared,
        Instant::now(),
        timeout,
        |st| (st.done >= n).then(|| std::mem::take(&mut st.frames)),
        |_| None,
    );
    let frames = match r {
        Ok(f) => f,
        Err(e) => {
            // SAFETY: cancelling our own generator.
            unsafe { generator.cancelAllCGImageGeneration() };
            return Err(format!("frame decoding timed out ({e})"));
        }
    };
    let mut out = Vec::with_capacity(n);
    for f in frames {
        match f {
            Some(Ok((img, t))) => out.push((img.0, t)),
            Some(Err(e)) => return Err(e),
            None => return Err("a frame callback is missing".into()),
        }
    }
    Ok(out)
}

/// Decode frames of a local video file at `times` (seconds) and encode them like
/// [`fetch_video_frames`] does (same generator settings, same JPEG path). For tests on sample
/// media and for checking the frame path without the Photos library.
pub fn frames_from_file(
    path: &std::path::Path,
    times: &[f64],
    max_side: u32,
    quality: f64,
) -> Result<Vec<Frame>> {
    let path = path
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let url =
        objc2_foundation::NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
    // SAFETY: a valid file URL and no options.
    let av = unsafe { AVURLAsset::URLAssetWithURL_options(&url, None) };
    let decoded = decode_frames(&av, times, max_side, Duration::from_secs(60))
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    decoded
        .into_iter()
        .map(|(img, t)| {
            let (jpeg, width, height) = encode_jpeg(&img, max_side, quality)?;
            Ok(Frame {
                jpeg,
                width,
                height,
                t: Some(t),
            })
        })
        .collect()
}

/// Fetch the poster and the sampled frames of `asset` (a video), long side at most
/// `cfg.image.max_side`, as JPEGs.
///
/// Call from the main thread, or from a worker while the main thread runs its run loop. Retries
/// the whole video `cfg.image.attempts` times. Never downloads the original.
pub fn fetch_video_frames(asset: &AssetInfo, cfg: &VideoConfig) -> Result<VideoFrames> {
    if !asset.is_video {
        bail!("{} is not a video asset", asset.id);
    }
    let n_total = frame_count(asset.duration, cfg.frames, cfg.full_secs);
    let times = sample_times(asset.duration, n_total - 1);
    let mut errors = Vec::new();
    for attempt in 1..=cfg.image.attempts.max(1) {
        let r = autoreleasepool(|_| -> std::result::Result<VideoFrames, String> {
            let poster_cfg = FetchConfig {
                attempts: 1,
                ..cfg.image.clone()
            };
            let poster = fetch_resized(asset, &poster_cfg).map_err(|e| format!("poster: {e:#}"))?;
            let t0 = Instant::now();
            let mut from_icloud = poster.from_icloud;
            let av = match request_av_asset(asset, cfg, false) {
                Err(e) if e.starts_with(NEEDS_NETWORK) => {
                    from_icloud = true;
                    request_av_asset(asset, cfg, true)
                }
                other => other,
            }?;
            let request_time = t0.elapsed();
            if is_remote(&av) {
                return Err("Photos returned a remote stream, not a local rendition".into());
            }
            let rendition_bytes = local_file_bytes(&av);
            let t1 = Instant::now();
            let decoded = decode_frames(&av, &times, cfg.image.max_side, cfg.decode_timeout)?;
            let decode_time = t1.elapsed();
            let mut frames = vec![Frame {
                jpeg: poster.jpeg,
                width: poster.width,
                height: poster.height,
                t: None,
            }];
            for (img, t) in decoded {
                let (jpeg, width, height) =
                    encode_jpeg(&img, cfg.image.max_side, cfg.image.quality)
                        .map_err(|e| format!("encoding frame at {t:.2}s: {e:#}"))?;
                frames.push(Frame {
                    jpeg,
                    width,
                    height,
                    t: Some((t * 1000.0).round() / 1000.0),
                });
            }
            Ok(VideoFrames {
                frames,
                from_icloud,
                rendition_bytes,
                request_time,
                decode_time,
            })
        });
        match r {
            Ok(v) => return Ok(v),
            Err(e) => {
                errors.push(format!("attempt {attempt}: {e}"));
                if attempt < cfg.image.attempts {
                    std::thread::sleep(Duration::from_secs(2u64 << (attempt - 1)));
                }
            }
        }
    }
    bail!("video fetch failed for {}: {}", asset.id, errors.join("; "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts() {
        assert_eq!(frame_count(30.0, 8, 4.0), 8);
        assert_eq!(frame_count(4.0, 8, 4.0), 8);
        assert_eq!(frame_count(3.0, 8, 4.0), 6);
        assert_eq!(frame_count(1.2, 8, 4.0), 2);
        assert_eq!(frame_count(0.4, 8, 4.0), 2);
        assert_eq!(frame_count(0.0, 8, 4.0), 2);
        assert_eq!(frame_count(f64::NAN, 8, 4.0), 2);
        assert_eq!(frame_count(100.0, 1, 4.0), 2);
    }

    #[test]
    fn times_are_slice_middles() {
        assert_eq!(
            sample_times(7.0, 7),
            vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        );
        assert_eq!(sample_times(2.0, 1), vec![1.0]);
    }
}
