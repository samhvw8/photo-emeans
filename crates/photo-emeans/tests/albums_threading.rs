//! The threading wrapper behind `albums --yes`, tested without touching the Photos library.
//!
//! `harness = false`: this main() runs on the process main thread, like the CLI's main().
//! The fake "create" blocks its worker until a block it dispatched to the main queue has run.
//! That is the pattern that hangs when the main thread waits without running its run loop.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use dispatch2::DispatchQueue;
use photo_emeans::mainthread::{is_main_thread, run_pumping, run_with_pump};

/// Like a PhotoKit call whose result arrives on the main queue: blocks until main runs it.
fn fake_create() -> Result<String> {
    let (tx, rx) = mpsc::channel();
    DispatchQueue::main().exec_async(move || {
        let _ = tx.send((is_main_thread(), "folder-id".to_string()));
    });
    let (on_main, id) = rx
        .recv_timeout(Duration::from_secs(20))
        .map_err(|_| anyhow!("the main queue never ran the completion"))?;
    anyhow::ensure!(on_main, "completion ran off the main thread");
    Ok(id)
}

fn main() {
    assert!(
        is_main_thread(),
        "harness = false must give main() the main thread"
    );
    let slice = Duration::from_millis(20);

    // 1. Main pumps its run loop: the main-queue completion runs and the call returns.
    let t = Instant::now();
    let id = run_pumping(fake_create, Duration::from_secs(10), slice)
        .expect("wrapper")
        .expect("fake create");
    assert_eq!(id, "folder-id");
    println!(
        "ok 1 pumping: completed in {:.0} ms",
        t.elapsed().as_secs_f64() * 1e3
    );

    // 2. Control: main only sleeps (as a plain join would block it): the completion never runs,
    //    and the timeout turns the hang into an error.
    let t = Instant::now();
    let e = run_with_pump(
        fake_create,
        Duration::from_millis(800),
        slice,
        &mut std::thread::sleep,
    )
    .expect_err("must time out without pumping");
    assert!(format!("{e}").contains("did not finish within"), "{e}");
    assert!(t.elapsed() < Duration::from_secs(3));
    println!("ok 2 not pumping: timed out as expected ({e})");

    // 3. Called off the main thread: refused before any work starts.
    let e = std::thread::spawn(move || run_pumping(|| 1, Duration::from_secs(1), slice))
        .join()
        .unwrap()
        .expect_err("must refuse off the main thread");
    assert!(format!("{e}").contains("main thread"), "{e}");
    println!("ok 3 off main thread: refused");

    // 4. A panicking worker becomes an error, not a hang.
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let e = run_pumping(|| -> u8 { panic!("boom") }, Duration::from_secs(5), slice)
        .expect_err("panic must be an error");
    std::panic::set_hook(prev);
    assert!(format!("{e}").contains("panicked"), "{e}");
    println!("ok 4 worker panic: reported");

    // 5. A slow blocking call (like performChangesAndWait waiting on a prompt) finishes.
    let v = run_pumping(
        || {
            std::thread::sleep(Duration::from_millis(300));
            7
        },
        Duration::from_secs(5),
        slice,
    )
    .unwrap();
    assert_eq!(v, 7);
    println!("ok 5 slow worker: completed");
    println!("albums_threading: 5 checks passed");
}
