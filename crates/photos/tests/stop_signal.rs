//! SIGTERM / SIGINT stop the bulk fetches cleanly: the first signal sets the stop flag (the
//! process keeps running), and both fetch pools then start no new asset and report the signal.
//! Runs as its own `main()` on the process main thread (the fetch pools require it), with no
//! assets, so it never touches the Photos library.

use std::fs;

unsafe extern "C" {
    fn raise(sig: i32) -> i32;
}

fn main() {
    let dir = std::env::temp_dir().join(format!("photos-stop-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);

    photos::stop::install_signal_stop();
    assert_eq!(photos::stop::stop_requested(), None);
    // SAFETY: plain libc call; the handler installed above catches it.
    assert_eq!(unsafe { raise(15) }, 0);
    assert_eq!(
        photos::stop::stop_requested(),
        Some(15),
        "SIGTERM sets the flag"
    );

    let r = photos::fetch_all_videos(&dir.join("videos"), &[], &Default::default(), |_| {})
        .expect("fetch_all_videos");
    assert_eq!((r.fetched, r.interrupted), (0, Some(15)));
    let r = photos::fetch_all(&dir.join("photos"), &[], &Default::default(), |_| {})
        .expect("fetch_all");
    assert_eq!((r.fetched, r.interrupted), (0, Some(15)));
    // Nothing partial: only the (empty) manifests exist.
    for sub in ["videos", "photos"] {
        let names: Vec<String> = fs::read_dir(dir.join(sub))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            vec!["manifest.jsonl".to_string()],
            "{sub}: {names:?}"
        );
    }

    photos::stop::clear_stop();
    // SAFETY: as above.
    assert_eq!(unsafe { raise(2) }, 0);
    assert_eq!(
        photos::stop::stop_requested(),
        Some(2),
        "SIGINT sets the flag"
    );
    photos::stop::clear_stop();
    photos::stop::request_stop();
    assert_eq!(photos::stop::stop_requested(), Some(-1));
    photos::stop::clear_stop();

    fs::remove_dir_all(&dir).unwrap();
    println!("stop_signal: ok");
}
