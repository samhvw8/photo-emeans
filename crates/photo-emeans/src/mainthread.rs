//! Running a blocking PhotoKit call from the CLI without starving the main queue.
//!
//! `photos::create_albums` calls `PHPhotoLibrary performChangesAndWait:error:`. Apple's
//! documentation for `performChangesAndWait(_:)` says: "Do not call this method from the main
//! thread. [...] Photos may need to prompt the user to perform changes, so this method can block
//! execution indefinitely." So the call runs on a worker thread. The main thread does not block
//! in `join` meanwhile: it runs its run loop (and so the main dispatch queue), because PhotoKit
//! delivers some work there (image results, see `.okf/design/photokit.md`; a permission prompt
//! needs it too). This is the same arrangement as `photos::fetch_all`. A timeout turns a hang into
//! a clear error.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{self, TryRecvError};
use std::time::{Duration, Instant};

use anyhow::{Result, bail, ensure};
use objc2::MainThreadMarker;
use objc2_core_foundation::{CFRunLoop, kCFRunLoopDefaultMode};

/// `true` on the process's main thread.
pub fn is_main_thread() -> bool {
    MainThreadMarker::new().is_some()
}

/// Run the main run loop (and the main dispatch queue) for at most `d`.
pub fn pump_main_run_loop(d: Duration) {
    // SAFETY: read of a constant CoreFoundation string.
    let mode = unsafe { kCFRunLoopDefaultMode };
    CFRunLoop::run_in_mode(mode, d.as_secs_f64(), true);
}

/// Run `work` on a worker thread while the main thread runs its run loop in `slice` steps.
/// Must be called on the main thread. Fails after `timeout` (the worker cannot be cancelled; it
/// keeps running until the process exits) or when `work` panics.
pub fn run_pumping<T: Send + 'static>(
    work: impl FnOnce() -> T + Send + 'static,
    timeout: Duration,
    slice: Duration,
) -> Result<T> {
    run_with_pump(work, timeout, slice, &mut pump_main_run_loop)
}

/// [`run_pumping`] with the waiting step given by the caller (tests pass one that does not pump).
pub fn run_with_pump<T: Send + 'static>(
    work: impl FnOnce() -> T + Send + 'static,
    timeout: Duration,
    slice: Duration,
    pump: &mut dyn FnMut(Duration),
) -> Result<T> {
    ensure!(
        is_main_thread(),
        "must run on the main thread: it keeps the main queue running while the worker waits for Photos"
    );
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("photos-change".into())
        .spawn(move || {
            let r = catch_unwind(AssertUnwindSafe(work));
            let _ = tx.send(r); // the caller may have timed out and gone
        })?;
    let start = Instant::now();
    loop {
        match rx.try_recv() {
            Ok(Ok(v)) => return Ok(v),
            Ok(Err(_)) => bail!("the worker thread panicked"),
            Err(TryRecvError::Disconnected) => bail!("the worker thread ended without a result"),
            Err(TryRecvError::Empty) => {}
        }
        let elapsed = start.elapsed();
        if elapsed >= timeout {
            bail!(
                "did not finish within {:.1} s; the call is still running and cannot be cancelled",
                timeout.as_secs_f64()
            );
        }
        pump(slice.min(timeout - elapsed));
    }
}
