//! Clean stop on SIGINT / SIGTERM for the bulk fetches.
//!
//! [`install_signal_stop`] sets a handler for both signals. The first signal sets a flag:
//! [`crate::fetch_all`] and [`crate::fetch_all_videos`] start no new asset, finish the ones in
//! flight (their manifest lines are written), and return with `interrupted = true`. A second
//! signal exits at once with status 128 + signal; a kill at any point leaves no partial file in the
//! manifest (files are renamed into place before their line is written).
//!
//! The handler only touches atomics and calls `_exit`, which are async-signal-safe.

use std::sync::atomic::{AtomicI32, Ordering};

const SIGINT: i32 = 2;
const SIGTERM: i32 = 15;

/// 0 = no stop asked; otherwise the signal number (or -1 for [`request_stop`]).
static STOP: AtomicI32 = AtomicI32::new(0);

unsafe extern "C" {
    fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
    fn _exit(code: i32) -> !;
}

extern "C" fn on_signal(sig: i32) {
    if STOP.swap(sig, Ordering::SeqCst) != 0 {
        // Second signal: the user means it.
        // SAFETY: `_exit` is async-signal-safe.
        unsafe { _exit(128 + sig) };
    }
}

/// Install the SIGINT / SIGTERM handler for this process.
pub fn install_signal_stop() {
    // SAFETY: plain libc calls with a valid `extern "C"` handler.
    unsafe {
        signal(SIGINT, on_signal);
        signal(SIGTERM, on_signal);
    }
}

/// Ask running fetches to stop (as a first signal would).
pub fn request_stop() {
    let _ = STOP.compare_exchange(0, -1, Ordering::SeqCst, Ordering::SeqCst);
}

/// The signal that asked for a stop (-1 for [`request_stop`]), if any.
pub fn stop_requested() -> Option<i32> {
    match STOP.load(Ordering::SeqCst) {
        0 => None,
        s => Some(s),
    }
}

/// Clear the stop flag (tests, or a process that starts a second fetch).
pub fn clear_stop() {
    STOP.store(0, Ordering::SeqCst);
}
