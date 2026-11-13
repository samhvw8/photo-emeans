//! Near-duplicate clean-up: the suggested keeper of each dedup group (shown in the gallery's
//! "Clean up" tab) and the reader of the `cleanup.json` that tab downloads.
//!
//! Nothing here deletes or changes a photo. `albums --cleanup` only collects the marked photos
//! into one album, and the user deletes them in the Photos app.
//!
//! Keeper rule, per group of 2 or more members:
//!
//! 1. a favorite before a non-favorite;
//! 2. then the larger pixel area (`width x height` of the stored copy in the manifest; copies have
//!    a long side of at most 1024 px, so this ranks originals smaller than that, and crops);
//! 3. then the group's anchor (`DedupResult::anchor`: the representative when the run's
//!    `dedup.json` predates stored anchors);
//! 4. then the lower row.
//!
//! The keeper and every other favorite of the group start as "keep"; the rest start as "remove".
//! A favorite is never pre-marked for removal.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, bail, ensure};
use emeans::DedupResult;
use serde::{Deserialize, Serialize};

use crate::manifest::Item;
use crate::store::Kind;

/// What the keeper rule knows about one member of a group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Member {
    pub row: u32,
    pub favorite: bool,
    pub width: u32,
    pub height: u32,
}

impl Member {
    fn pixels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// Order of the keeper rule: the better member first.
fn better(a: &Member, b: &Member, anchor: u32) -> Ordering {
    b.favorite
        .cmp(&a.favorite)
        .then(b.pixels().cmp(&a.pixels()))
        .then((b.row == anchor).cmp(&(a.row == anchor)))
        .then(a.row.cmp(&b.row))
}

/// The members ranked by the keeper rule, the keeper first.
pub fn rank(members: &[Member], anchor: u32) -> Vec<Member> {
    let mut v = members.to_vec();
    v.sort_by(|a, b| better(a, b, anchor));
    v
}

/// The suggested keeper's row. Panics on an empty group.
pub fn keeper(members: &[Member], anchor: u32) -> u32 {
    members
        .iter()
        .min_by(|a, b| better(a, b, anchor))
        .expect("a group has members")
        .row
}

/// Rows that start as "keep": the keeper and every other favorite.
pub fn pre_keep(members: &[Member], anchor: u32) -> Vec<u32> {
    let k = keeper(members, anchor);
    rank(members, anchor)
        .into_iter()
        .filter(|m| m.row == k || m.favorite)
        .map(|m| m.row)
        .collect()
}

/// One near-duplicate group as the gallery page gets it.
#[derive(Debug, Clone, Serialize)]
pub struct CleanGroup {
    /// Group index in `dedup.json`.
    pub g: u32,
    pub anchor: u32,
    /// True when every member is a screenshot (`subtypes` has `screenshot`).
    pub shots: bool,
    /// Rows ranked by the keeper rule, the keeper first.
    pub rows: Vec<u32>,
    /// Rows that start as "keep".
    pub keep: Vec<u32>,
    /// `[width, height]` of the stored copy, parallel to `rows`.
    pub px: Vec<(u32, u32)>,
    /// 1 when the row is a screenshot, parallel to `rows`.
    pub shot: Vec<u8>,
}

/// The clean-up data of a run.
#[derive(Debug, Clone, Serialize)]
pub struct CleanupData {
    /// False when `dedup.json` has no anchors, so the representative stands in for the anchor.
    pub anchors_stored: bool,
    /// Groups of 2 or more members, largest first, then by group index.
    pub groups: Vec<CleanGroup>,
}

/// Build the groups of 2 or more members from a run's dedup result. `items[row]` is the manifest
/// entry of each row of the run.
pub fn groups(dd: &DedupResult, items: &[&Item]) -> Result<CleanupData> {
    let n = items.len();
    ensure!(
        dd.group_of.len() == n,
        "dedup.json covers {} rows, the run has {n}",
        dd.group_of.len()
    );
    let ng = dd.n_groups();
    let mut by_group: Vec<Vec<u32>> = vec![Vec::new(); ng];
    for (row, &g) in dd.group_of.iter().enumerate() {
        ensure!(
            (g as usize) < ng,
            "dedup.json: row {row} is in group {g} of {ng}"
        );
        by_group[g as usize].push(row as u32);
    }
    let mut out = Vec::new();
    for (g, rows) in by_group.into_iter().enumerate() {
        ensure!(
            dd.sizes.get(g).copied() == Some(rows.len() as u32),
            "dedup.json: group {g} has {} rows but size {:?}",
            rows.len(),
            dd.sizes.get(g)
        );
        if rows.len() < 2 {
            continue;
        }
        let anchor = dd.anchor(g);
        let members: Vec<Member> = rows
            .iter()
            .map(|&r| {
                let it = items[r as usize];
                Member {
                    row: r,
                    favorite: it.favorite,
                    width: it.width,
                    height: it.height,
                }
            })
            .collect();
        let ranked = rank(&members, anchor);
        let is_shot = |r: u32| items[r as usize].subtypes.iter().any(|s| s == "screenshot");
        out.push(CleanGroup {
            g: g as u32,
            anchor,
            shots: rows.iter().all(|&r| is_shot(r)),
            keep: pre_keep(&members, anchor),
            px: ranked.iter().map(|m| (m.width, m.height)).collect(),
            shot: ranked.iter().map(|m| u8::from(is_shot(m.row))).collect(),
            rows: ranked.iter().map(|m| m.row).collect(),
        });
    }
    out.sort_by(|a, b| b.rows.len().cmp(&a.rows.len()).then(a.g.cmp(&b.g)));
    Ok(CleanupData {
        anchors_stored: dd.anchors.len() == dd.representatives.len(),
        groups: out,
    })
}

/// One entry of `cleanup.json`: a photo or video the user marked for removal.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    /// PHAsset local identifier.
    pub asset_id: String,
    /// Its image relative to `$EMEANS_DATA` (`photos/<uuid>.jpg`, or `videos/<uuid>/f00.jpg`).
    pub file: String,
    /// Group index in the run's `dedup.json`.
    pub group: u32,
    pub reason: String,
}

/// Parse `cleanup.json` text. Refuses an empty list, an empty id and an id listed twice.
pub fn parse(text: &str) -> Result<Vec<Entry>> {
    let list: Vec<Entry> = serde_json::from_str(text)
        .context("cleanup.json must be a JSON array of {asset_id, file, group, reason}")?;
    ensure!(!list.is_empty(), "cleanup.json lists no photos");
    let mut seen = HashSet::new();
    for e in &list {
        ensure!(
            !e.asset_id.trim().is_empty(),
            "cleanup.json has an empty asset_id"
        );
        ensure!(
            seen.insert(e.asset_id.as_str()),
            "cleanup.json lists {} twice",
            e.asset_id
        );
    }
    Ok(list)
}

/// What `albums --cleanup` would put into its one album.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checked {
    pub asset_ids: Vec<String>,
    pub videos: usize,
    pub groups: usize,
    /// First and last creation date (`YYYY-MM-DD`), when known.
    pub dates: Option<(String, String)>,
}

/// Check the entries against the manifests: every id must be a fetched photo or video, and its
/// `file` must be that asset's image (a mismatch means the file is from another library or was
/// edited).
pub fn check(list: &[Entry], media: &HashMap<String, Item>) -> Result<Checked> {
    let mut dates = Vec::new();
    let mut videos = 0;
    for e in list {
        let Some(it) = media.get(&e.asset_id) else {
            bail!(
                "{} in cleanup.json is not in photos/ or videos/manifest.jsonl; was the file \
                 made from another library or edited?",
                e.asset_id
            );
        };
        ensure!(
            it.image == e.file,
            "{} in cleanup.json has file {:?}, the manifest says {:?}",
            e.asset_id,
            e.file,
            it.image
        );
        videos += usize::from(it.kind == Kind::Video);
        if let Some(d) = it.created.as_deref().and_then(|c| c.get(..10)) {
            dates.push(d);
        }
    }
    dates.sort_unstable();
    Ok(Checked {
        asset_ids: list.iter().map(|e| e.asset_id.clone()).collect(),
        videos,
        groups: list.iter().map(|e| e.group).collect::<HashSet<_>>().len(),
        dates: dates
            .first()
            .zip(dates.last())
            .map(|(a, b)| (a.to_string(), b.to_string())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(row: u32, favorite: bool, w: u32, h: u32) -> Member {
        Member {
            row,
            favorite,
            width: w,
            height: h,
        }
    }

    #[test]
    fn keeper_rule() {
        // A favorite wins over a larger copy and over the anchor.
        let g = [
            m(0, false, 1024, 768),
            m(1, true, 640, 480),
            m(2, false, 1024, 768),
        ];
        assert_eq!(keeper(&g, 2), 1);
        // No favorite: the larger pixel area wins over the anchor.
        let g = [
            m(0, false, 1024, 768),
            m(1, false, 1024, 576),
            m(2, false, 600, 600),
        ];
        assert_eq!(keeper(&g, 1), 0);
        // Equal area: the anchor wins.
        let g = [
            m(4, false, 1024, 768),
            m(7, false, 768, 1024),
            m(9, false, 1024, 768),
        ];
        assert_eq!(keeper(&g, 9), 9);
        // Equal area, anchor not among the best: the lower row.
        let g = [
            m(7, false, 1024, 768),
            m(4, false, 1024, 768),
            m(9, false, 500, 500),
        ];
        assert_eq!(keeper(&g, 9), 4);
        // Two favorites: the larger one keeps first; both start as "keep".
        let g = [
            m(0, true, 800, 600),
            m(1, true, 1024, 768),
            m(2, false, 1024, 768),
        ];
        assert_eq!(keeper(&g, 2), 1);
        assert_eq!(pre_keep(&g, 2), vec![1, 0]);
        assert_eq!(
            rank(&g, 2).iter().map(|m| m.row).collect::<Vec<_>>(),
            [1, 0, 2]
        );
        // Order of the input does not change the answer.
        let mut r = g;
        r.reverse();
        assert_eq!(keeper(&r, 2), 1);
    }

    fn item(kind: Kind, image: &str, fav: bool, w: u32, h: u32, shot: bool) -> Item {
        Item {
            kind,
            image: image.into(),
            created: Some("2020-05-06T07:08:09Z".into()),
            favorite: fav,
            burst_id: None,
            width: w,
            height: h,
            subtypes: if shot {
                vec!["screenshot".into()]
            } else {
                vec![]
            },
            duration: 0.0,
            frames: vec![],
        }
    }

    #[test]
    fn groups_from_dedup() {
        let its = [
            item(Kind::Photo, "photos/a.jpg", false, 1024, 768, false),
            item(Kind::Photo, "photos/b.jpg", false, 1024, 768, false),
            item(Kind::Photo, "photos/c.jpg", false, 1024, 768, true),
            item(Kind::Photo, "photos/d.jpg", true, 400, 300, true),
            item(Kind::Photo, "photos/e.jpg", false, 1024, 768, true),
            item(Kind::Photo, "photos/f.jpg", false, 1024, 768, false),
        ];
        let refs: Vec<&Item> = its.iter().collect();
        let dd = DedupResult {
            group_of: vec![0, 0, 2, 2, 2, 5],
            representatives: vec![1, 3, 5],
            anchors: vec![],
            sizes: vec![2, 3, 1],
            nn_sim: vec![1.0; 6],
            linked_pairs: 3,
            split_components: 0,
        };
        // group_of must index 0..n_groups.
        assert!(groups(&dd, &refs).is_err());
        let dd = DedupResult {
            group_of: vec![0, 0, 1, 1, 1, 2],
            ..dd
        };
        let c = groups(&dd, &refs).unwrap();
        assert!(!c.anchors_stored);
        assert_eq!(c.groups.len(), 2);
        let (big, small) = (&c.groups[0], &c.groups[1]);
        assert_eq!(
            (big.g, big.rows.as_slice(), big.shots),
            (1, &[3, 2, 4][..], true)
        );
        assert_eq!(big.keep, [3]);
        assert_eq!(big.anchor, 3, "representative stands in for the anchor");
        assert_eq!(big.px[0], (400, 300));
        assert_eq!((small.g, small.rows.as_slice()), (0, &[1, 0][..]));
        assert!(!small.shots);
        assert_eq!(small.keep, [1], "equal area: the anchor (representative 1)");
        // With stored anchors the anchor decides the tie.
        let dd = DedupResult {
            anchors: vec![0, 4, 5],
            ..dd
        };
        let c = groups(&dd, &refs).unwrap();
        assert!(c.anchors_stored);
        assert_eq!(c.groups[1].keep, [0]);
        assert_eq!(
            c.groups[0].rows,
            [3, 4, 2],
            "the anchor ranks before the lower row"
        );
        // sizes must agree with group_of.
        let dd = DedupResult {
            sizes: vec![2, 2, 2],
            ..dd
        };
        assert!(groups(&dd, &refs).is_err());
    }

    #[test]
    fn cleanup_json_reader() {
        let media: HashMap<String, Item> = [
            (
                "A/L0/001",
                item(Kind::Photo, "photos/A.jpg", false, 4, 3, false),
            ),
            (
                "B/L0/001",
                item(Kind::Photo, "photos/B.jpg", false, 4, 3, false),
            ),
            (
                "V/L0/001",
                item(Kind::Video, "videos/V/f00.jpg", false, 4, 3, false),
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let ok = r#"[{"asset_id":"A/L0/001","file":"photos/A.jpg","group":3,"reason":"near-duplicate of B/L0/001"},
                    {"asset_id":"V/L0/001","file":"videos/V/f00.jpg","group":3,"reason":"x"},
                    {"asset_id":"B/L0/001","file":"photos/B.jpg","group":8,"reason":"x"}]"#;
        let c = check(&parse(ok).unwrap(), &media).unwrap();
        assert_eq!(c.asset_ids, ["A/L0/001", "V/L0/001", "B/L0/001"]);
        assert_eq!((c.videos, c.groups), (1, 2));
        assert_eq!(c.dates, Some(("2020-05-06".into(), "2020-05-06".into())));
        for bad in [
            "[]",
            "{}",
            r#"[{"asset_id":"A/L0/001","file":"photos/A.jpg","group":1}]"#,
            r#"[{"asset_id":"A/L0/001","file":"photos/A.jpg","group":-1,"reason":""}]"#,
            r#"[{"asset_id":"A/L0/001","file":"photos/A.jpg","group":1,"reason":"","extra":1}]"#,
            r#"[{"asset_id":" ","file":"photos/A.jpg","group":1,"reason":""}]"#,
            r#"[{"asset_id":"A/L0/001","file":"photos/A.jpg","group":1,"reason":""},
                {"asset_id":"A/L0/001","file":"photos/A.jpg","group":2,"reason":""}]"#,
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
        for bad in [
            r#"[{"asset_id":"Z/L0/001","file":"photos/Z.jpg","group":1,"reason":""}]"#,
            r#"[{"asset_id":"A/L0/001","file":"photos/B.jpg","group":1,"reason":""}]"#,
        ] {
            assert!(check(&parse(bad).unwrap(), &media).is_err(), "{bad}");
        }
    }
}
