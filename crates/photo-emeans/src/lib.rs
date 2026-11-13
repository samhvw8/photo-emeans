//! `photo-emeans`: the CLI steps of the emeans pipeline (see AGENTS.md, data contract).
//!
//! - [`embed`]: resumable, crash-safe embedding of `photos/` into `embed/` ([`store`]).
//! - [`manifest`]: read-only view of `photos/manifest.jsonl`.
//! - [`cluster`]: `cluster` (plain or automatic k, optional dedup) and incremental `add`.
//! - [`gallery`]: static local HTML gallery with the album approval flow.
//! - [`albums`]: checks `approved.json` (or `cleanup.json`) and creates albums only on `--yes`.
//! - [`cleanup`]: keeper rule for near-duplicate groups and the `cleanup.json` reader.
//! - [`names`]: content names for cluster albums (`album-names.json`), used by `albums --rename-from`.
//! - [`mainthread`]: run a blocking PhotoKit call on a worker while main pumps its run loop.

pub mod albums;
pub mod cleanup;
pub mod cluster;
pub mod embed;
pub mod gallery;
pub mod mainthread;
pub mod manifest;
pub mod names;
pub mod store;
pub mod util;

/// Lower this process's CPU priority (`setpriority(PRIO_PROCESS, 0, nice)`). The Mac is shared.
pub fn renice(nice: i32) -> anyhow::Result<()> {
    unsafe extern "C" {
        fn setpriority(which: i32, who: u32, prio: i32) -> i32;
    }
    // SAFETY: plain libc call on the current process (PRIO_PROCESS = 0, who = 0).
    let r = unsafe { setpriority(0, 0, nice) };
    anyhow::ensure!(
        r == 0,
        "setpriority({nice}) failed: {}",
        std::io::Error::last_os_error()
    );
    Ok(())
}
