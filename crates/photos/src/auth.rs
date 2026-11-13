//! Photos authorization (TCC). The first request shows a macOS prompt for the app that runs the
//! process (the terminal app for a CLI).

use std::sync::mpsc;
use std::time::Duration;

use anyhow::{Result, bail};
use block2::RcBlock;
use objc2_photos::{PHAccessLevel, PHAuthorizationStatus, PHPhotoLibrary};

/// Photos access state for read/write access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Full library access.
    Authorized,
    /// The user picked "Limited Access": only the photos they selected are visible.
    Limited,
    /// The user refused access.
    Denied,
    /// A policy (MDM, parental controls) blocks access.
    Restricted,
    /// The user has not answered yet (the prompt is open, or was never shown).
    NotDetermined,
}

impl Access {
    fn from_status(s: PHAuthorizationStatus) -> Self {
        match s {
            PHAuthorizationStatus::Authorized => Access::Authorized,
            PHAuthorizationStatus::Limited => Access::Limited,
            PHAuthorizationStatus::Denied => Access::Denied,
            PHAuthorizationStatus::Restricted => Access::Restricted,
            _ => Access::NotDetermined,
        }
    }
}

/// What the user must do to grant full access. Shown with every non-`Authorized` result.
pub const GRANT_STEPS: &str = "Open System Settings > Privacy & Security > Photos, find the app that \
runs this command (your terminal app, e.g. Terminal, iTerm or Ghostty), and set it to \"Full Access\". \
Then quit and reopen that app, and run the command again.";

/// Current read/write status without prompting.
pub fn current() -> Access {
    // SAFETY: plain class method call with a valid enum value.
    let s = unsafe { PHPhotoLibrary::authorizationStatusForAccessLevel(PHAccessLevel::ReadWrite) };
    Access::from_status(s)
}

/// Request read/write access and block until the user answers or `timeout` passes.
///
/// Returns the status as is; it does not treat `Denied` or `Limited` as an error. Use
/// [`ensure_authorized`] when the caller needs full access.
pub fn authorize(timeout: Duration) -> Access {
    let now = current();
    if now != Access::NotDetermined {
        return now;
    }
    let (tx, rx) = mpsc::channel();
    // Photos calls the handler once, on an arbitrary serial queue.
    let handler = RcBlock::new(move |s: PHAuthorizationStatus| {
        let _ = tx.send(s);
    });
    // SAFETY: the block lives until the call returns; PhotoKit copies it.
    unsafe {
        PHPhotoLibrary::requestAuthorizationForAccessLevel_handler(
            PHAccessLevel::ReadWrite,
            &handler,
        )
    };
    // The handler runs on an arbitrary queue; run the main run loop too, in case that queue is
    // the main one.
    let deadline = std::time::Instant::now() + timeout;
    let main = crate::fetch::is_main_thread();
    while std::time::Instant::now() < deadline {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(s) => return Access::from_status(s),
            Err(mpsc::RecvTimeoutError::Timeout) if main => {
                crate::fetch::pump_main_run_loop(Duration::from_millis(50))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    current()
}

/// Request access and fail with clear instructions unless the result is full access.
pub fn ensure_authorized(timeout: Duration) -> Result<()> {
    match authorize(timeout) {
        Access::Authorized => Ok(()),
        Access::Limited => bail!(
            "Photos access is Limited: only hand-picked photos are visible, so the library would be \
             incomplete. {GRANT_STEPS}"
        ),
        Access::Denied => bail!("Photos access was denied. {GRANT_STEPS}"),
        Access::Restricted => bail!(
            "Photos access is restricted by a system policy (MDM or Screen Time); the user cannot \
             change it in System Settings."
        ),
        Access::NotDetermined => bail!(
            "No answer to the Photos permission prompt within {}s. Look for the macOS dialog \
             \"... would like to access your Photos\" and click \"Allow Full Access\". If no \
             dialog appears: {GRANT_STEPS}",
            timeout.as_secs()
        ),
    }
}
