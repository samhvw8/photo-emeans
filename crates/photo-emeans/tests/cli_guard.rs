//! `fetch-videos` refuses to run without `--limit N` or `--all`, before it touches Photos.

use std::process::Command;

fn run(args: &[&str]) -> (bool, String) {
    let dir = std::env::temp_dir().join(format!("pe-cli-guard-{}", std::process::id()));
    let out = Command::new(env!("CARGO_BIN_EXE_photo-emeans"))
        .arg("--data")
        .arg(&dir)
        .args(args)
        .output()
        .unwrap();
    assert!(
        !dir.exists(),
        "the refused command created {}",
        dir.display()
    );
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn fetch_videos_needs_limit_or_all() {
    let (ok, err) = run(&["fetch-videos"]);
    assert!(!ok);
    assert!(err.contains("--limit"), "{err}");
    let (ok, err) = run(&["fetch-videos", "--limit", "60", "--all"]);
    assert!(!ok);
    assert!(err.contains("cannot be used with"), "{err}");
    // `--sample` is an alias of `--pick`.
    let (ok, err) = run(&["fetch-videos", "--sample", "bogus", "--limit", "1"]);
    assert!(!ok);
    assert!(err.contains("bogus"), "{err}");
}
