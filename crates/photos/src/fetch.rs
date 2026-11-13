//! Fetching one resized copy of an asset as JPEG.
//!
//! Request: `PHImageManager requestImageForAsset:targetSize:contentMode:options:resultHandler:`
//! with `deliveryMode = HighQualityFormat`, `resizeMode = Exact`, `version = Current`,
//! `synchronous = false`, target `max_side x max_side` aspect-fit. Each fetch asks first with
//! `networkAccessAllowed = false`; only on `PHPhotosErrorNetworkAccessRequired` (3164) does it ask
//! again with `networkAccessAllowed = true`, so `Fetched::from_icloud` is exact.
//!
//! - Async results arrive on the main queue (measured), so a waiting main thread runs its run
//!   loop; see [`pump_main_run_loop`].
//! - `requestImage` (not `requestImageDataAndOrientation`): the data call returns the full-size
//!   original bytes, which would download originals from iCloud. `requestImage` lets Photos pick
//!   the smallest local derivative that covers the target, or download a resized copy.
//! - `Exact`: the result fits the target box, so the output never exceeds `max_side`. `Fast` may
//!   return a larger derivative that we would have to scale again anyway.
//! - `version = Current`: the photo as the user sees it (crops and edits applied).
//! - The rendered `NSImage` is already upright (Photos applies the EXIF orientation). It is drawn
//!   into an 8-bit sRGB bitmap (which also flattens HDR/wide-gamut and alpha onto white) and
//!   encoded with ImageIO. The JPEG carries no EXIF orientation tag.

use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use block2::RcBlock;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyObject, Bool};
use objc2_app_kit::NSImage;
use objc2_core_foundation::{
    CFDictionary, CFMutableData, CFNumber, CFRunLoop, CFString, CGPoint, CGRect, CGSize,
    kCFRunLoopDefaultMode,
};
use objc2_core_graphics::{
    CGBitmapContextCreate, CGBitmapContextCreateImage, CGColorSpace, CGContext, CGImage,
    CGImageAlphaInfo, CGInterpolationQuality, kCGColorSpaceSRGB,
};
use objc2_foundation::{NSDictionary, NSError, NSNumber, NSString, NSThread};
use objc2_image_io::{CGImageDestination, kCGImageDestinationLossyCompressionQuality};
use objc2_photos::{
    PHImageCancelledKey, PHImageContentMode, PHImageErrorKey, PHImageManager,
    PHImageRequestOptions, PHImageRequestOptionsDeliveryMode, PHImageRequestOptionsResizeMode,
    PHImageRequestOptionsVersion, PHImageResultIsDegradedKey, PHImageResultIsInCloudKey,
    PHPhotosError,
};

use crate::assets::AssetInfo;

/// Error text of a network-off request for an asset that is only in iCloud at this size
/// (`PHImageResultIsInCloudKey`, or `PHPhotosErrorNetworkAccessRequired` = 3164).
const NEEDS_NETWORK: &str = "needs an iCloud download";

/// Options for [`fetch_resized`].
#[derive(Debug, Clone)]
pub struct FetchConfig {
    /// Longest side of the output, in pixels.
    pub max_side: u32,
    /// JPEG quality, 0..=1.
    pub quality: f64,
    /// Give up on an attempt when Photos reports no progress for this long.
    pub idle_timeout: Duration,
    /// Give up on an attempt after this long, progress or not.
    pub hard_timeout: Duration,
    /// Total attempts per asset (1 = no retry).
    pub attempts: u32,
}

impl Default for FetchConfig {
    fn default() -> Self {
        Self {
            max_side: 1024,
            quality: 0.88,
            idle_timeout: Duration::from_secs(60),
            hard_timeout: Duration::from_secs(300),
            attempts: 3,
        }
    }
}

/// A fetched copy.
#[derive(Debug, Clone)]
pub struct Fetched {
    pub jpeg: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// No local copy was large enough: Photos downloaded a resized copy from iCloud.
    pub from_icloud: bool,
    /// Attempts used (1 = first try worked).
    pub attempts: u32,
}

#[derive(Default)]
struct State {
    result: Option<std::result::Result<Retained<CGImage>, String>>,
    last_activity: Option<Instant>,
    progress_error: Option<String>,
}

struct Shared {
    state: Mutex<State>,
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

fn request_options(network: bool, shared: Option<Arc<Shared>>) -> Retained<PHImageRequestOptions> {
    // SAFETY: property setters on a fresh object; the progress block is copied by the setter.
    unsafe {
        let o = PHImageRequestOptions::new();
        o.setDeliveryMode(PHImageRequestOptionsDeliveryMode::HighQualityFormat);
        o.setResizeMode(PHImageRequestOptionsResizeMode::Exact);
        o.setVersion(PHImageRequestOptionsVersion::Current);
        o.setNetworkAccessAllowed(network);
        o.setSynchronous(false);
        if let Some(shared) = shared {
            // Photos calls this on an arbitrary queue while it works on the request. It also
            // fires (with 1.0) for purely local renders, so it is an activity signal for the idle
            // timeout, not proof of a download.
            let progress = RcBlock::new(
                move |_p: f64,
                      err: *mut NSError,
                      _stop: NonNull<Bool>,
                      _info: *mut NSDictionary| {
                    let mut st = shared.state.lock().unwrap();
                    st.last_activity = Some(Instant::now());
                    if let Some(e) = err.as_ref() {
                        st.progress_error = Some(describe_error(e));
                    }
                },
            );
            o.setProgressHandler(RcBlock::as_ptr(&progress));
        }
        o
    }
}

/// One request; returns the rendered CGImage and whether Photos downloaded from iCloud.
fn request_once(
    asset: &AssetInfo,
    cfg: &FetchConfig,
    network: bool,
) -> std::result::Result<Retained<CGImage>, String> {
    let shared = Arc::new(Shared {
        state: Mutex::new(State {
            last_activity: Some(Instant::now()),
            ..State::default()
        }),
        cv: Condvar::new(),
    });
    let opts = request_options(network, Some(shared.clone()));
    let handler_shared = shared.clone();
    let handler = RcBlock::new(move |image: *mut NSImage, info: *mut NSDictionary| {
        // SAFETY: Photos passes a valid NSImage or nil, and a valid dictionary or nil.
        let (image, info) = unsafe { (image.as_ref(), info.as_ref()) };
        let outcome = autoreleasepool(|_| -> Option<std::result::Result<_, String>> {
            // SAFETY: reading the documented Photos info keys.
            let (degraded, cancelled, in_cloud, err) = unsafe {
                match info {
                    Some(i) => (
                        info_bool(i, PHImageResultIsDegradedKey),
                        info_bool(i, PHImageCancelledKey),
                        info_bool(i, PHImageResultIsInCloudKey),
                        info_get(i, PHImageErrorKey).and_then(|e| e.downcast::<NSError>().ok()),
                    ),
                    None => (false, false, false, None),
                }
            };
            if degraded && err.is_none() && !cancelled {
                // A low-quality preview; the final image follows in a second call.
                return None;
            }
            if cancelled {
                return Some(Err("request cancelled".into()));
            }
            let network_required = err.as_ref().is_some_and(|e| {
                e.domain().to_string() == "PHPhotosErrorDomain"
                    && e.code() == PHPhotosError::NetworkAccessRequired.0
            });
            if network_required || (in_cloud && image.is_none()) {
                return Some(Err(NEEDS_NETWORK.into()));
            }
            if let Some(e) = err {
                return Some(Err(format!("Photos error: {}", describe_error(&e))));
            }
            let Some(image) = image else {
                return Some(Err(if in_cloud {
                    "Photos returned no image (asset in iCloud)".into()
                } else {
                    "Photos returned no image and no error".into()
                }));
            };
            // SAFETY: null rect means "the image's own size"; no context or hints.
            let cg =
                unsafe { image.CGImageForProposedRect_context_hints(ptr::null_mut(), None, None) };
            Some(cg.ok_or_else(|| "NSImage has no CGImage representation".to_string()))
        });
        if let Some(outcome) = outcome {
            let mut st = handler_shared.state.lock().unwrap();
            if st.result.is_none() {
                st.result = Some(outcome);
            }
            handler_shared.cv.notify_all();
        }
    });

    let target = CGSize::new(f64::from(cfg.max_side), f64::from(cfg.max_side));
    // SAFETY: valid asset, options and block; PhotoKit copies the block.
    let manager = unsafe { PHImageManager::defaultManager() };
    let req_id = unsafe {
        manager.requestImageForAsset_targetSize_contentMode_options_resultHandler(
            &asset.handle.0,
            target,
            PHImageContentMode::AspectFit,
            Some(&opts),
            &handler,
        )
    };

    let started = Instant::now();
    let on_main_thread = is_main_thread();
    let mut st = shared.state.lock().unwrap();
    loop {
        if let Some(r) = st.result.take() {
            return r.map_err(|e| match st.progress_error.take() {
                Some(pe) => format!("{e}; download error: {pe}"),
                None => e,
            });
        }
        let idle = st.last_activity.map_or(Duration::ZERO, |t| t.elapsed());
        if idle > cfg.idle_timeout || started.elapsed() > cfg.hard_timeout {
            drop(st);
            // SAFETY: cancelling our own request id.
            unsafe { manager.cancelImageRequest(req_id) };
            return Err(format!(
                "timed out after {:.0}s ({} without progress; network {}). \
                 If this happens on every photo, the main thread is not running its run loop: \
                 call from the main thread or through fetch_all",
                started.elapsed().as_secs_f64(),
                if idle > cfg.idle_timeout {
                    format!("{:.0}s", idle.as_secs_f64())
                } else {
                    "not idle".into()
                },
                if network { "on" } else { "off" }
            ));
        }
        if on_main_thread {
            // Photos delivers the result handler on the main queue: run it while we wait.
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

/// `true` on the process's main thread.
pub(crate) fn is_main_thread() -> bool {
    NSThread::isMainThread_class()
}

/// Run the main run loop (and so the main dispatch queue) for at most `d`.
///
/// PhotoKit delivers asynchronous image results on the main queue, even for requests made on
/// other threads. A process whose main thread blocks without running its run loop never gets
/// them. Measured on macOS 27: a background-thread request with the main thread blocked timed
/// out after 60 s; the same request with the main thread in this loop finished in 0.33 s.
pub(crate) fn pump_main_run_loop(d: Duration) {
    debug_assert!(is_main_thread());
    // SAFETY: extern static read of a constant run loop mode.
    let mode = unsafe { kCFRunLoopDefaultMode };
    CFRunLoop::run_in_mode(mode, d.as_secs_f64(), true);
}

/// Fetch a copy of `asset` with its longest side at most `cfg.max_side`, as JPEG.
///
/// Call it from the main thread (it runs the main run loop while it waits), or from another
/// thread while the main thread runs its run loop (as [`crate::fetch_all`] does).
///
/// Downloads a resized copy from iCloud when no large enough local version exists. Retries
/// `cfg.attempts` times with a growing pause. Never downloads the original.
pub fn fetch_resized(asset: &AssetInfo, cfg: &FetchConfig) -> Result<Fetched> {
    let mut errors = Vec::new();
    for attempt in 1..=cfg.attempts.max(1) {
        // Local first: with network off, Photos returns a local copy or says it needs iCloud
        // (error 3164). Only then ask again with network on. This costs nothing for local
        // photos and makes `from_icloud` exact.
        let mut from_icloud = false;
        let result = match request_once(asset, cfg, false) {
            Err(e) if e.starts_with(NEEDS_NETWORK) => {
                from_icloud = true;
                request_once(asset, cfg, true)
            }
            other => other,
        };
        match result {
            Ok(cg) => {
                let (jpeg, width, height) = encode_jpeg(&cg, cfg.max_side, cfg.quality)
                    .with_context(|| format!("encoding {}", asset.id))?;
                return Ok(Fetched {
                    jpeg,
                    width,
                    height,
                    from_icloud,
                    attempts: attempt,
                });
            }
            Err(e) => {
                errors.push(format!("attempt {attempt}: {e}"));
                if attempt < cfg.attempts {
                    std::thread::sleep(Duration::from_secs(2u64 << (attempt - 1)));
                }
            }
        }
    }
    bail!("fetch failed for {}: {}", asset.id, errors.join("; "))
}

/// `true` when Photos can render the asset at `max_side` without the network.
///
/// Makes a network-off request; Photos answers with `PHImageResultIsInCloudKey` when it would
/// have to download. Costs one local render.
pub fn is_local(asset: &AssetInfo, max_side: u32) -> Result<bool> {
    let cfg = FetchConfig {
        max_side,
        attempts: 1,
        ..FetchConfig::default()
    };
    match request_once(asset, &cfg, false) {
        Ok(_) => Ok(true),
        Err(e) if e.starts_with(NEEDS_NETWORK) => Ok(false),
        Err(e) => Err(anyhow!("local check failed for {}: {e}", asset.id)),
    }
}

/// Draw `src` into an 8-bit sRGB bitmap no larger than `max_side`, then encode as JPEG.
pub(crate) fn encode_jpeg(
    src: &CGImage,
    max_side: u32,
    quality: f64,
) -> Result<(Vec<u8>, u32, u32)> {
    let (sw, sh) = (CGImage::width(Some(src)), CGImage::height(Some(src)));
    if sw == 0 || sh == 0 {
        bail!("empty image {sw}x{sh}");
    }
    let (w, h) = fit(sw, sh, max_side as usize);
    let space = CGColorSpace::with_name(Some(unsafe { kCGColorSpaceSRGB }))
        .context("no sRGB color space")?;
    // SAFETY: null data lets CoreGraphics allocate; 8 bits RGBX is a supported format.
    let ctx = unsafe {
        CGBitmapContextCreate(
            ptr::null_mut::<c_void>(),
            w,
            h,
            8,
            0,
            Some(&space),
            CGImageAlphaInfo::NoneSkipLast.0,
        )
    }
    .context("CGBitmapContextCreate failed")?;
    let rect = CGRect::new(CGPoint::new(0.0, 0.0), CGSize::new(w as f64, h as f64));
    // White under transparent pixels (PNG screenshots), instead of black.
    CGContext::set_rgb_fill_color(Some(&ctx), 1.0, 1.0, 1.0, 1.0);
    CGContext::fill_rect(Some(&ctx), rect);
    CGContext::set_interpolation_quality(Some(&ctx), CGInterpolationQuality::High);
    CGContext::draw_image(Some(&ctx), rect, Some(src));
    let out =
        CGBitmapContextCreateImage(Some(&ctx)).context("CGBitmapContextCreateImage failed")?;

    let data = CFMutableData::new(None, 0).context("CFDataCreateMutable failed")?;
    let jpeg_type = CFString::from_static_str("public.jpeg");
    // SAFETY: valid mutable data and type; no options.
    let dest = unsafe { CGImageDestination::with_data(&data, &jpeg_type, 1, None) }
        .context("CGImageDestinationCreateWithData failed")?;
    let q = CFNumber::new_cgfloat(quality);
    // SAFETY: extern static read.
    let key: &CFString = unsafe { kCGImageDestinationLossyCompressionQuality };
    let props = CFDictionary::<CFString, CFNumber>::from_slices(&[key], &[&q]);
    // SAFETY: the dictionary holds CFString keys and CFNumber values, as ImageIO expects.
    unsafe { dest.add_image(&out, Some(props.as_ref())) };
    if !unsafe { dest.finalize() } {
        bail!("CGImageDestinationFinalize failed");
    }
    Ok((data.to_vec(), w as u32, h as u32))
}

/// Aspect-fit `w x h` into a `max x max` box; never upscales; each side at least 1.
pub fn fit(w: usize, h: usize, max: usize) -> (usize, usize) {
    let long = w.max(h);
    if long <= max {
        return (w, h);
    }
    let s = max as f64 / long as f64;
    let sw = ((w as f64 * s).round() as usize).clamp(1, max);
    let sh = ((h as f64 * s).round() as usize).clamp(1, max);
    (sw, sh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_box() {
        assert_eq!(fit(4032, 3024, 1024), (1024, 768));
        assert_eq!(fit(3024, 4032, 1024), (768, 1024));
        assert_eq!(fit(800, 600, 1024), (800, 600));
        assert_eq!(fit(20000, 10, 1024), (1024, 1));
    }

    #[test]
    fn encode_roundtrip() {
        // A 2000x1000 sRGB gradient, encoded, must come back as a 1024x512 JPEG.
        let (w, h) = (2000usize, 1000usize);
        let space = CGColorSpace::with_name(Some(unsafe { kCGColorSpaceSRGB })).unwrap();
        let ctx = unsafe {
            CGBitmapContextCreate(
                ptr::null_mut(),
                w,
                h,
                8,
                0,
                Some(&space),
                CGImageAlphaInfo::PremultipliedLast.0,
            )
        }
        .unwrap();
        CGContext::set_rgb_fill_color(Some(&ctx), 0.2, 0.5, 0.8, 1.0);
        CGContext::fill_rect(
            Some(&ctx),
            CGRect::new(CGPoint::new(0.0, 0.0), CGSize::new(w as f64, h as f64)),
        );
        let img = CGBitmapContextCreateImage(Some(&ctx)).unwrap();
        let (jpeg, ow, oh) = encode_jpeg(&img, 1024, 0.88).unwrap();
        assert_eq!((ow, oh), (1024, 512));
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]);
        assert_eq!(crate::store::jpeg_size(&jpeg), Some((1024, 512)));
    }
}
