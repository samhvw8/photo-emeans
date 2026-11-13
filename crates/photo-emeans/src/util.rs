//! Small helpers: atomic writes, time stamps, ISO 8601 parsing, logging.

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

/// Write `data` to `path` atomically: temporary file, fsync, rename, fsync of the directory.
pub fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    let dir = path
        .parent()
        .with_context(|| format!("{} has no parent", path.display()))?;
    let name = path
        .file_name()
        .with_context(|| format!("{} has no file name", path.display()))?;
    let tmp = dir.join(format!("{}.tmp", name.to_string_lossy()));
    {
        let mut f = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        f.write_all(data)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path).with_context(|| format!("renaming to {}", path.display()))?;
    sync_dir(dir)
}

/// fsync a directory, so a rename or a new file in it survives a crash.
pub fn sync_dir(dir: &Path) -> Result<()> {
    File::open(dir)
        .and_then(|d| d.sync_all())
        .with_context(|| format!("fsync {}", dir.display()))
}

pub fn now_unix() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

pub fn now_iso() -> String {
    photos::assets::iso8601_utc(now_unix())
}

/// Parse `YYYY-MM-DDTHH:MM:SS[.fff](Z|+HH:MM|-HH:MM)` to Unix seconds. `None` if malformed.
pub fn parse_iso8601(s: &str) -> Option<f64> {
    let b = s.as_bytes();
    if b.len() < 19
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
    {
        return None;
    }
    let num = |r: std::ops::Range<usize>| -> Option<i64> { s.get(r)?.parse().ok() };
    let (y, mo, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    let (h, mi, sec) = (num(11..13)?, num(14..16)?, num(17..19)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || sec > 60 {
        return None;
    }
    let mut rest = &s[19..];
    let mut frac = 0.0;
    if let Some(r) = rest.strip_prefix('.') {
        let end = r.find(|c: char| !c.is_ascii_digit()).unwrap_or(r.len());
        frac = format!("0.{}", &r[..end]).parse().ok()?;
        rest = &r[end..];
    }
    let offset = match rest {
        "Z" | "" => 0,
        o if o.len() == 6 && (o.starts_with('+') || o.starts_with('-')) && &o[3..4] == ":" => {
            let sign = if o.starts_with('-') { -1 } else { 1 };
            sign * (o[1..3].parse::<i64>().ok()? * 3600 + o[4..6].parse::<i64>().ok()? * 60)
        }
        _ => return None,
    };
    // Howard Hinnant's days_from_civil.
    let y2 = if mo <= 2 { y - 1 } else { y };
    let era = y2.div_euclid(400);
    let yoe = y2 - era * 400;
    let mp = (mo + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some((days * 86_400 + h * 3600 + mi * 60 + sec - offset) as f64 + frac)
}

/// Seconds as `1h02m`, `3m05s` or `12s`.
pub fn fmt_duration(secs: f64) -> String {
    if !secs.is_finite() || secs < 0.0 {
        return "?".into();
    }
    let s = secs.round() as u64;
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    } else if s >= 60 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_round_trip() {
        for t in [0.0, 1_546_223_978.0, 951_782_400.0, 4_102_444_799.0] {
            let s = photos::assets::iso8601_utc(t);
            assert_eq!(parse_iso8601(&s), Some(t), "{s}");
        }
        assert_eq!(parse_iso8601("2018-12-31T02:39:38Z"), Some(1_546_223_978.0));
        assert_eq!(
            parse_iso8601("2018-12-31T09:39:38+07:00"),
            Some(1_546_223_978.0)
        );
        assert_eq!(
            parse_iso8601("2018-12-31T02:39:38.5Z"),
            Some(1_546_223_978.5)
        );
        assert_eq!(parse_iso8601("2018-13-31T02:39:38Z"), None);
        assert_eq!(parse_iso8601("garbage"), None);
    }

    #[test]
    fn durations() {
        assert_eq!(fmt_duration(5.0), "5s");
        assert_eq!(fmt_duration(185.0), "3m05s");
        assert_eq!(fmt_duration(3720.0), "1h02m");
    }
}
