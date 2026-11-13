//! Content names for cluster albums: score each album's photos against a vocabulary of everyday
//! photo concepts (`labels.txt`, embedded with the EmbeddingGemma 2 text side) and title the album
//! "Beach & Sunset · 2019–2024 · 65". Writes `album-names.json`, the input (and undo record) of
//! `albums --rename-from`. Reads only embeddings and metadata, never a photo.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::albums::{self, Approved};
use crate::store::D;

/// The vocabulary shipped with the tool (`crates/photo-emeans/labels.txt`).
pub const LABELS: &str = include_str!("../labels.txt");

/// Folder title the albums get unless `--folder-to` says otherwise.
pub const FOLDER_TO: &str = "Photo groups";

/// One concept: the name used in titles and the texts whose vectors are averaged into it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Label {
    pub name: String,
    pub texts: Vec<String>,
}

/// Parse the vocabulary: `Name | text` per line, `#` comments, blank lines ignored. A line without
/// `| text` embeds "a photo of <name in lower case>". Repeated names are merged in order.
pub fn parse_labels(text: &str) -> Result<Vec<Label>> {
    let mut out: Vec<Label> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, t) = match line.split_once('|') {
            Some((n, t)) => (n.trim(), t.trim().to_string()),
            None => (line, format!("a photo of {}", line.to_lowercase())),
        };
        ensure!(
            !name.is_empty() && !t.is_empty(),
            "labels line {}: empty name or text",
            i + 1
        );
        ensure!(
            !name.contains('\u{b7}') && !name.contains('&'),
            "labels line {}: a name may not contain \u{b7} or & (title separators)",
            i + 1
        );
        match index.get(name) {
            Some(&j) => out[j].texts.push(t),
            None => {
                index.insert(name.to_string(), out.len());
                out.push(Label {
                    name: name.to_string(),
                    texts: vec![t],
                });
            }
        }
    }
    ensure!(out.len() >= 2, "the vocabulary needs at least 2 labels");
    Ok(out)
}

/// Turns texts into unit vectors of dimension `D`. The model side of [`label_vectors`].
pub trait TextEmbedder {
    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}

/// One cached text vector (`names/label-vectors.jsonl`).
#[derive(Serialize, Deserialize)]
struct CacheLine {
    /// Model, revision, compute dtype and prompt: a vector is reused only under the same key.
    key: String,
    text: String,
    v: Vec<f32>,
}

/// Unit vectors (`labels.len() x D`) for `labels`: the normalized mean of each label's text
/// vectors. Texts already in `cache` under `key` are reused; the rest go to `embedder` (created
/// only when needed) and are appended to `cache`.
pub fn label_vectors(
    labels: &[Label],
    cache: &Path,
    key: &str,
    embedder: &mut dyn FnMut() -> Result<Box<dyn TextEmbedder>>,
    log: &mut dyn FnMut(&str),
) -> Result<Vec<f32>> {
    let mut known: HashMap<String, Vec<f32>> = HashMap::new();
    if cache.exists() {
        for (i, line) in fs::read_to_string(cache)?.lines().enumerate() {
            let l: CacheLine = serde_json::from_str(line)
                .with_context(|| format!("{} line {}", cache.display(), i + 1))?;
            if l.key == key {
                ensure!(
                    l.v.len() == D,
                    "{} line {}: not {D}-d",
                    cache.display(),
                    i + 1
                );
                known.insert(l.text, l.v);
            }
        }
    }
    let mut missing: Vec<&str> = labels
        .iter()
        .flat_map(|l| l.texts.iter().map(String::as_str))
        .filter(|t| !known.contains_key(*t))
        .collect();
    missing.sort_unstable();
    missing.dedup();
    if !missing.is_empty() {
        log(&format!("embedding {} label texts", missing.len()));
        let mut e = embedder()?;
        let vs = e.embed(&missing)?;
        ensure!(
            vs.len() == missing.len(),
            "embedder returned {} vectors",
            vs.len()
        );
        if let Some(dir) = cache.parent() {
            fs::create_dir_all(dir)?;
        }
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(cache)?;
        for (t, v) in missing.iter().zip(vs) {
            ensure!(
                v.len() == D && v.iter().all(|x| x.is_finite()),
                "bad vector for {t:?}"
            );
            let line = CacheLine {
                key: key.to_string(),
                text: t.to_string(),
                v,
            };
            writeln!(f, "{}", serde_json::to_string(&line)?)?;
            known.insert(line.text, line.v);
        }
        f.sync_data()?;
    }
    let mut out = Vec::with_capacity(labels.len() * D);
    for l in labels {
        let mut m = [0f64; D];
        for t in &l.texts {
            for (a, b) in m.iter_mut().zip(&known[t.as_str()]) {
                *a += f64::from(*b);
            }
        }
        let norm = m.iter().map(|x| x * x).sum::<f64>().sqrt();
        ensure!(norm > 0.0, "label {:?} has a zero vector", l.name);
        out.extend(m.iter().map(|x| (x / norm) as f32));
    }
    Ok(out)
}

/// Cosine of every row of `x` (`n x D`, unit rows) with every label (`l x D`): `n x l`.
pub fn scores(x: &[f32], t: &[f32]) -> Vec<f32> {
    let l = t.len() / D;
    let mut s = vec![0f32; x.len() / D * l];
    s.par_chunks_mut(l)
        .zip(x.par_chunks(D))
        .for_each(|(out, row)| {
            for (j, o) in out.iter_mut().enumerate() {
                *o = row
                    .iter()
                    .zip(&t[j * D..(j + 1) * D])
                    .map(|(a, b)| a * b)
                    .sum();
            }
        });
    s
}

/// How clusters are scored and when a name counts as weak. Stored in `album-names.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoreOpts {
    /// Subtract each label's mean score over the whole library before a photo picks its top
    /// label, so labels that score high on everything ("hubs") do not win every vote.
    pub center: bool,
    /// Each photo votes for its `top_k` best labels, so near-synonyms ("Office", "Coworkers") do
    /// not split one theme's votes. A share is then the fraction of members that have the label
    /// among their `top_k`.
    pub top_k: usize,
    /// The top label needs at least this share of the votes, or the cluster is "Mixed".
    pub min_share: f32,
    /// The weakest label in the name needs this many more votes (as a share) than the best label
    /// left out; a name that fails with two labels is tried with one, then it is "Mixed".
    pub min_margin: f32,
    /// A second label joins the name when it has at least this share of the votes.
    pub second_share: f32,
}

impl Default for ScoreOpts {
    fn default() -> Self {
        Self {
            center: true,
            top_k: 3,
            min_share: 0.4,
            // Off: with top-3 votes, near-synonyms tie by design (two labels at 100% and
            // 96% of one album), so a share margin marks coherent albums as Mixed.
            min_margin: 0.0,
            second_share: 0.4,
        }
    }
}

/// One label's result in a cluster.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LabelScore {
    pub label: String,
    /// Share of the members that have this label among their `top_k` labels.
    pub share: f32,
    /// Mean raw cosine of the members with the label.
    pub mean: f32,
    /// Mean cosine minus the label's library mean (what the vote uses when `center`).
    pub lift: f32,
}

/// The naming result of one cluster.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Naming {
    /// "Beach & Sunset", or "Mixed (Food)" when weak.
    pub name: String,
    pub mixed: bool,
    /// Share of the weakest named label minus the share of the best label left out.
    pub margin: f32,
    /// The best labels by votes (then by lift), at most 3.
    pub top: Vec<LabelScore>,
}

/// Name one cluster from its members' rows of `s` (`n x l`, raw cosines). `bias[j]` is label j's
/// library mean.
pub fn name_cluster(
    rows: &[usize],
    s: &[f32],
    bias: &[f32],
    labels: &[Label],
    o: &ScoreOpts,
) -> Naming {
    let l = labels.len();
    let mut votes = vec![0usize; l];
    let mut sum = vec![0f64; l];
    let k = o.top_k.clamp(1, l);
    let mut order: Vec<usize> = (0..l).collect();
    for &r in rows {
        let row = &s[r * l..(r + 1) * l];
        let v = |j: usize| if o.center { row[j] - bias[j] } else { row[j] };
        for (acc, x) in sum.iter_mut().zip(row) {
            *acc += f64::from(*x);
        }
        // The photo's k best labels each get its vote (ties: lower index first).
        order.sort_unstable_by(|&a, &b| v(b).total_cmp(&v(a)).then(a.cmp(&b)));
        for &j in &order[..k] {
            votes[j] += 1;
        }
    }
    let n = rows.len().max(1) as f32;
    let mut all: Vec<LabelScore> = (0..l)
        .map(|j| {
            let mean = (sum[j] / n as f64) as f32;
            LabelScore {
                label: labels[j].name.clone(),
                share: votes[j] as f32 / n,
                mean,
                lift: mean - bias[j],
            }
        })
        .collect();
    let key = |a: &LabelScore| if o.center { a.lift } else { a.mean };
    all.sort_by(|a, b| {
        b.share
            .total_cmp(&a.share)
            .then(key(b).total_cmp(&key(a)))
            .then(a.label.cmp(&b.label))
    });
    // The second label: the best one with enough votes whose name does not repeat the first's
    // ("Sign" and "Signboard", "City" and "Night City" say the same thing twice).
    let first = all[0].label.to_lowercase();
    let second = all[1..]
        .iter()
        .take_while(|x| x.share >= o.second_share)
        .position(|x| {
            let n = x.label.to_lowercase();
            !n.contains(&first) && !first.contains(&n)
        })
        .map(|i| i + 1);
    let share = |i: usize| all.get(i).map_or(0.0, |x| x.share);
    let s1 = share(0);
    // Best label left out of the name.
    let left_out = |named: &[usize]| {
        (1..all.len())
            .find(|i| !named.contains(i))
            .map_or(0.0, share)
    };
    // The named labels must stand out from the best label left out of the name.
    let (name, mixed, margin) = if s1 < o.min_share {
        (format!("Mixed ({})", all[0].label), true, s1 - share(1))
    } else if let Some(j) = second.filter(|&j| share(j) - left_out(&[0, j]) >= o.min_margin) {
        let m = share(j) - left_out(&[0, j]);
        (format!("{} & {}", all[0].label, all[j].label), false, m)
    } else if s1 - share(1) >= o.min_margin {
        (all[0].label.clone(), false, s1 - share(1))
    } else {
        (format!("Mixed ({})", all[0].label), true, s1 - share(1))
    };
    all.truncate(3);
    Naming {
        name,
        mixed,
        margin,
        top: all,
    }
}

/// `2019–2024`, `2019` for one year, `None` without dates. Dates are ISO strings.
pub fn year_span(dates: &[&str]) -> Option<String> {
    let mut years: Vec<&str> = dates
        .iter()
        .filter_map(|d| d.get(..4))
        .filter(|y| y.bytes().all(|b| b.is_ascii_digit()))
        .collect();
    years.sort_unstable();
    match (years.first(), years.last()) {
        (Some(a), Some(b)) if a == b => Some(a.to_string()),
        (Some(a), Some(b)) => Some(format!("{a}\u{2013}{b}")),
        _ => None,
    }
}

/// `Beach & Sunset · 2019–2024 · 65`, or `Beach · 65` without dates.
pub fn title(name: &str, years: Option<&str>, count: usize) -> String {
    match years {
        Some(y) => format!("{name} \u{b7} {y} \u{b7} {count}"),
        None => format!("{name} \u{b7} {count}"),
    }
}

/// The folder rename recorded in `album-names.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FolderNames {
    /// Photos local identifier, filled in by the first `albums --rename-from` that finds it.
    pub local_id: Option<String>,
    pub old_title: String,
    pub new_title: String,
}

/// One album in `album-names.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AlbumName {
    /// Gallery key of the cluster (`"19.1"`).
    pub cluster: String,
    /// Photos local identifier, filled in by the first `albums --rename-from` that finds it.
    pub local_id: Option<String>,
    pub old_title: String,
    pub new_title: String,
    /// Assets in the album when it was created.
    pub count: usize,
    pub mixed: bool,
    pub margin: f32,
    pub top: Vec<LabelScore>,
}

/// `album-names.json`: the rename plan and its undo record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NamesFile {
    pub version: u32,
    pub created: String,
    /// The approved list the albums were created from.
    pub approved: String,
    pub model: String,
    pub prompt: String,
    /// Labels file (`builtin` for the shipped one), its FNV-1a 64 hash and label count.
    pub labels_source: String,
    pub labels_fnv: String,
    pub labels: usize,
    pub method: ScoreOpts,
    pub folder: FolderNames,
    /// Album titles in the folder that stay as they are (the duplicates album).
    pub unchanged: Vec<String>,
    pub albums: Vec<AlbumName>,
}

impl NamesFile {
    pub fn read(path: &Path) -> Result<Self> {
        let f: Self = serde_json::from_slice(
            &fs::read(path).with_context(|| format!("reading {}", path.display()))?,
        )
        .with_context(|| format!("parsing {}", path.display()))?;
        ensure!(
            f.version == 1,
            "{}: unknown version {}",
            path.display(),
            f.version
        );
        Ok(f)
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        crate::util::write_atomic(path, &serde_json::to_vec_pretty(self)?)
    }
}

/// Inputs of [`name_albums`] besides the data.
pub struct NameOpts<'a> {
    pub labels: &'a [Label],
    /// Label vectors (`labels.len() x D`).
    pub label_vectors: &'a [f32],
    pub score: ScoreOpts,
    /// Entries of the approved list to leave alone (by cluster key).
    pub skip_clusters: &'a [&'a str],
}

/// Name every album of `approved` whose cluster is not skipped. `rows_of` maps an asset id to its
/// embedding row, `created` to its capture date. Titles are unique (` #2` on a collision) and do
/// not repeat a skipped album's title.
pub fn name_albums(
    approved: &[Approved],
    x: &[f32],
    rows_of: &HashMap<&str, usize>,
    created: &dyn Fn(&str) -> Option<String>,
    o: &NameOpts,
) -> Result<(Vec<AlbumName>, Vec<String>)> {
    let s = scores(x, o.label_vectors);
    let l = o.labels.len();
    let n = x.len() / D;
    let mut bias = vec![0f64; l];
    for row in s.chunks(l) {
        for (b, v) in bias.iter_mut().zip(row) {
            *b += f64::from(*v);
        }
    }
    let bias: Vec<f32> = bias.iter().map(|b| (b / n.max(1) as f64) as f32).collect();
    let key = |a: &Approved| match &a.cluster {
        serde_json::Value::String(s) => s.clone(),
        v => v.to_string(),
    };
    let mut unchanged = Vec::new();
    let mut named = Vec::new();
    for a in approved {
        let k = key(a);
        if o.skip_clusters.contains(&k.as_str()) {
            unchanged.push(a.title.clone());
            continue;
        }
        let rows: Vec<usize> =
            a.asset_ids
                .iter()
                .map(|id| {
                    rows_of.get(id.as_str()).copied().with_context(|| {
                        format!("album {:?}: asset {id} has no embedding", a.title)
                    })
                })
                .collect::<Result<_>>()?;
        ensure!(!rows.is_empty(), "album {:?} is empty", a.title);
        let naming = name_cluster(&rows, &s, &bias, o.labels, &o.score);
        let dates: Vec<String> = a.asset_ids.iter().filter_map(|id| created(id)).collect();
        let d: Vec<&str> = dates.iter().map(String::as_str).collect();
        named.push((
            k,
            a,
            title(&naming.name, year_span(&d).as_deref(), a.asset_ids.len()),
            naming,
        ));
    }
    // Unique among themselves and against the titles that stay.
    let mut titles: Vec<String> = unchanged.clone();
    titles.extend(named.iter().map(|(_, _, t, _)| t.clone()));
    let titles = albums::unique_titles(titles);
    let out = named
        .into_iter()
        .zip(titles.into_iter().skip(unchanged.len()))
        .map(|((k, a, _, naming), new_title)| AlbumName {
            cluster: k,
            local_id: None,
            old_title: a.title.clone(),
            new_title,
            count: a.asset_ids.len(),
            mixed: naming.mixed,
            margin: naming.margin,
            top: naming.top,
        })
        .collect();
    Ok((out, unchanged))
}

/// FNV-1a 64 of `text`, hex.
pub fn fnv(text: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// The lines `names` prints: one per album plus a summary of the shares.
pub fn format_names(albums: &[AlbumName]) -> Vec<String> {
    let w = albums
        .iter()
        .map(|a| a.old_title.chars().count())
        .max()
        .unwrap_or(0);
    let mut out: Vec<String> = albums
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let top: Vec<String> = a
                .top
                .iter()
                .map(|t| format!("{} {:.0}%", t.label, t.share * 100.0))
                .collect();
            format!(
                "  {:>3}. {:<w$}  ->  {}   [{}]",
                i + 1,
                a.old_title,
                a.new_title,
                top.join(", ")
            )
        })
        .collect();
    let mixed = albums.iter().filter(|a| a.mixed).count();
    let mut shares: Vec<f32> = albums.iter().map(|a| a.top[0].share).collect();
    shares.sort_by(f32::total_cmp);
    let q = |p: f32| shares[((shares.len() - 1) as f32 * p).round() as usize];
    if !shares.is_empty() {
        out.push(format!(
            "{} albums, {mixed} Mixed. Top-label share: min {:.2}, p10 {:.2}, p25 {:.2}, median {:.2}, p75 {:.2}, max {:.2}.",
            albums.len(),
            q(0.0),
            q(0.1),
            q(0.25),
            q(0.5),
            q(0.75),
            q(1.0)
        ));
        let distinct: HashSet<&str> = albums.iter().map(|a| a.top[0].label.as_str()).collect();
        out.push(format!("{} distinct top labels.", distinct.len()));
    }
    out
}

/// Check that `labels` and the cluster file agree before naming. Fails on an empty vocabulary or
/// a label vector count that does not match.
pub fn check_vectors(labels: &[Label], t: &[f32]) -> Result<()> {
    if t.len() != labels.len() * D {
        bail!("{} label vectors for {} labels", t.len() / D, labels.len());
    }
    Ok(())
}

/// The EmbeddingGemma 2 text side with the search-query prompt (the prompt the encoder's
/// image-to-caption check uses; images are embedded without a prompt).
pub struct GemmaText(pub encoder::families::embeddinggemma2::EmbeddingGemma2);

/// The prompt label texts are embedded with.
pub const PROMPT: &str = encoder::families::embeddinggemma2::prompts::SEARCH_QUERY;

impl TextEmbedder for GemmaText {
    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        use encoder::Encoder;
        self.0.embed_texts(texts, Some(PROMPT))
    }
}

/// Options of the `names` command.
pub struct RunOpts<'a> {
    pub approved: &'a Path,
    /// Labels file; `None` = the shipped [`LABELS`].
    pub labels: Option<&'a Path>,
    pub out: &'a Path,
    pub folder: &'a str,
    pub folder_to: &'a str,
    pub score: ScoreOpts,
    /// Model id, revision and dtype, for the vector cache key and the file.
    pub model: &'a str,
    /// Overwrite an existing `out`.
    pub force: bool,
}

/// `names`: write `album-names.json` for the albums of `approved` and print the plan. Never
/// touches Photos.
pub fn run(
    data: &Path,
    o: &RunOpts,
    embedder: &mut dyn FnMut() -> Result<Box<dyn TextEmbedder>>,
    out: &mut dyn FnMut(&str),
) -> Result<()> {
    ensure!(
        o.force || !o.out.exists(),
        "{} exists; it may be the undo record of a rename that was applied. Move it away or pass \
         --force",
        o.out.display()
    );
    let approved_text = fs::read_to_string(o.approved)
        .with_context(|| format!("reading {}", o.approved.display()))?;
    let approved = albums::parse(&approved_text)?;
    let (labels_text, labels_source) = match o.labels {
        Some(p) => (
            fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?,
            p.display().to_string(),
        ),
        None => (LABELS.to_string(), "builtin".to_string()),
    };
    let labels = parse_labels(&labels_text)?;
    let key = format!("{}|{PROMPT}", o.model);
    let t = label_vectors(
        &labels,
        &data.join("names").join("label-vectors.jsonl"),
        &key,
        embedder,
        out,
    )?;
    check_vectors(&labels, &t)?;
    let emb = crate::store::load(&data.join("embed"))?;
    let rows_of: HashMap<&str, usize> = emb
        .ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();
    let media = crate::cluster::manifest_by_id(data)?;
    let created = |id: &str| media.get(id).and_then(|m| m.created.clone());
    let (named, unchanged) = name_albums(
        &approved,
        &emb.x[..emb.ids.len() * D],
        &rows_of,
        &created,
        &NameOpts {
            labels: &labels,
            label_vectors: &t,
            score: o.score.clone(),
            skip_clusters: &["duplicates"],
        },
    )?;
    ensure!(
        !named.is_empty(),
        "no cluster albums to name in {}",
        o.approved.display()
    );
    let file = NamesFile {
        version: 1,
        created: crate::util::now_iso(),
        approved: o.approved.display().to_string(),
        model: o.model.to_string(),
        prompt: PROMPT.to_string(),
        labels_source,
        labels_fnv: fnv(&labels_text),
        labels: labels.len(),
        method: o.score.clone(),
        folder: FolderNames {
            local_id: None,
            old_title: o.folder.to_string(),
            new_title: o.folder_to.to_string(),
        },
        unchanged,
        albums: named,
    };
    file.write(o.out)?;
    out(&format!(
        "Plan: rename folder {:?} to {:?} and {} of its albums ({} labels, scored over {} embedded rows):",
        o.folder,
        o.folder_to,
        file.albums.len(),
        labels.len(),
        emb.ids.len()
    ));
    for l in format_names(&file.albums) {
        out(&l);
    }
    for t in &file.unchanged {
        out(&format!("Unchanged: {t:?}"));
    }
    out(&format!("Wrote {}.", o.out.display()));
    out(&format!(
        "Nothing was renamed. Check it in Photos with: albums --rename-from {}",
        o.out.display()
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(i: usize) -> Vec<f32> {
        let mut v = vec![0f32; D];
        v[i] = 1.0;
        v
    }

    #[test]
    fn parses_the_vocabulary() {
        let l = parse_labels("# c\n\nBeach | sand and sea\nCat\nBeach | waves\n").unwrap();
        assert_eq!(l.len(), 2);
        assert_eq!(l[0].texts, ["sand and sea", "waves"]);
        assert_eq!(l[1].texts, ["a photo of cat"]);
        assert!(parse_labels("A | \nB").is_err());
        assert!(parse_labels("A \u{b7} B | x\nC").is_err());
        assert!(parse_labels("A & B | x\nC").is_err());
        assert!(parse_labels("Only").is_err());
        let shipped = parse_labels(LABELS).unwrap();
        assert!((300..=500).contains(&shipped.len()), "{}", shipped.len());
    }

    #[test]
    fn years_and_titles() {
        assert_eq!(
            year_span(&["2024-01-02T00:00:00Z", "2019-05-01", ""]).as_deref(),
            Some("2019\u{2013}2024")
        );
        assert_eq!(
            year_span(&["2021-01-01", "2021-12-31"]).as_deref(),
            Some("2021")
        );
        assert_eq!(year_span(&["", "x"]), None);
        assert_eq!(
            title("Beach & Sunset", Some("2019\u{2013}2024"), 65),
            "Beach & Sunset \u{b7} 2019\u{2013}2024 \u{b7} 65"
        );
        assert_eq!(title("Cat", None, 3), "Cat \u{b7} 3");
    }

    fn labels(names: &[&str]) -> Vec<Label> {
        names
            .iter()
            .map(|n| Label {
                name: n.to_string(),
                texts: vec![n.to_string()],
            })
            .collect()
    }

    /// Top-1 votes with a share margin, the rules these tests were written for.
    const K1: ScoreOpts = ScoreOpts {
        center: false,
        top_k: 1,
        min_share: 0.3,
        min_margin: 0.1,
        second_share: 0.2,
    };

    #[test]
    fn top_k_votes_join_synonyms() {
        let ls = labels(&["Office", "Coworkers", "Meeting", "Beach"]);
        let bias = [0.0; 4];
        // Each photo's best label differs, but all three office labels beat Beach for everyone.
        let s = [
            0.5, 0.4, 0.3, 0.0, //
            0.3, 0.5, 0.4, 0.0, //
            0.4, 0.3, 0.5, 0.0, //
            0.5, 0.3, 0.4, 0.1, //
        ];
        let rows = [0, 1, 2, 3];
        let k1 = name_cluster(&rows, &s, &bias, &ls, &K1);
        assert!((k1.top[0].share - 0.5).abs() < 1e-6, "{k1:?}");
        let o = ScoreOpts {
            center: false,
            ..Default::default()
        };
        let k3 = name_cluster(&rows, &s, &bias, &ls, &o);
        assert!(!k3.mixed);
        assert_eq!(k3.top[0].share, 1.0);
        // All three have 100%; the tie goes to the mean (Office 0.425, Meeting 0.4, Coworkers 0.375).
        assert_eq!(k3.name, "Office & Meeting");
        // A second label that repeats the first's name is skipped for the next one.
        let ls = labels(&["City", "Night City", "Sky", "Food"]);
        let s = [
            0.9, 0.8, 0.7, 0.0, //
            0.9, 0.8, 0.7, 0.0, //
        ];
        let n = name_cluster(&[0, 1], &s, &[0.0; 4], &ls, &o);
        assert_eq!(n.name, "City & Sky", "{n:?}");
        let ls = labels(&["Signboard", "Sign", "Food"]);
        let s = [0.9, 0.8, 0.0, 0.9, 0.8, 0.0];
        let o2 = ScoreOpts {
            top_k: 2,
            ..o.clone()
        };
        assert_eq!(
            name_cluster(&[0, 1], &s, &[0.0; 3], &ls, &o2).name,
            "Signboard"
        );
        // A cluster where no label reaches 40% of the members' top 3 is Mixed: 10 photos whose
        // top 3 are labels r, r+1, r+2 (mod 10), so every label gets 30%.
        let names: Vec<String> = (0..10).map(|i| format!("L{i}")).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let ls = labels(&names);
        let mut s = vec![0f32; 10 * 10];
        for r in 0..10 {
            for k in 0..3 {
                s[r * 10 + (r + k) % 10] = 1.0 - 0.1 * k as f32;
            }
        }
        let rows: Vec<usize> = (0..10).collect();
        let n = name_cluster(&rows, &s, &[0.0; 10], &ls, &o);
        assert!(n.mixed, "{n:?}");
        assert!((n.top[0].share - 0.3).abs() < 1e-6);
        assert!(n.name.starts_with("Mixed ("));
    }

    #[test]
    fn names_by_vote() {
        let ls = labels(&["Beach", "Sunset", "Food"]);
        let o = K1;
        let bias = [0.0; 3];
        // Rows: scores against [Beach, Sunset, Food].
        let s = [
            0.9, 0.1, 0.0, //
            0.8, 0.2, 0.0, //
            0.7, 0.1, 0.0, //
            0.2, 0.6, 0.0, //
            0.0, 0.0, 0.5, //
        ];
        // 3 Beach, 1 Sunset, 1 Food: Sunset does not stand out from Food, so one label.
        let n = name_cluster(&[0, 1, 2, 3, 4], &s, &bias, &ls, &o);
        assert_eq!(n.name, "Beach", "{n:?}");
        assert!(!n.mixed);
        assert!((n.top[0].share - 0.6).abs() < 1e-6);
        assert!((n.margin - 0.4).abs() < 1e-6);
        // 3 Beach and 2 Sunset (row 3 twice): two labels.
        let n = name_cluster(&[0, 1, 2, 3, 3], &s, &bias, &ls, &o);
        assert_eq!(n.name, "Beach & Sunset", "{n:?}");
        // 3 Beach out of 3: one label.
        assert_eq!(name_cluster(&[0, 1, 2], &s, &bias, &ls, &o).name, "Beach");
        // One vote each: weak top label, so Mixed with the strongest hint (by mean on a tie).
        let n = name_cluster(&[0, 3, 4], &s, &bias, &ls, &o);
        assert!(n.mixed);
        assert_eq!(n.name, "Mixed (Beach)");
    }

    #[test]
    fn centering_removes_a_hub_label() {
        let ls = labels(&["Hub", "Cat"]);
        // Every photo scores higher on Hub, but Hub is high for the whole library.
        let s = [0.5, 0.4, 0.5, 0.45, 0.5, 0.42];
        let bias = [0.5, 0.2];
        let o = ScoreOpts { center: true, ..K1 };
        assert_eq!(name_cluster(&[0, 1, 2], &s, &bias, &ls, &o).name, "Cat");
        let raw = ScoreOpts { center: false, ..o };
        assert_eq!(name_cluster(&[0, 1, 2], &s, &bias, &ls, &raw).name, "Hub");
    }

    #[test]
    fn albums_get_unique_titles_and_skip_duplicates() {
        let ls = labels(&["A", "B"]);
        let t = [unit(0), unit(1)].concat();
        let x = [unit(0), unit(0), unit(1)].concat();
        let ids = ["p0", "p1", "p2"];
        let rows_of: HashMap<&str, usize> = ids.iter().enumerate().map(|(i, s)| (*s, i)).collect();
        let ap = |c: &str, title: &str, a: &[&str]| Approved {
            cluster: serde_json::Value::String(c.into()),
            title: title.into(),
            asset_ids: a.iter().map(|s| s.to_string()).collect(),
        };
        let list = [
            ap("1.1", "old 1", &["p0"]),
            ap("1.2", "old 2", &["p1"]),
            ap("2.1", "old 3", &["p2"]),
            ap("duplicates", "A \u{b7} 2020 \u{b7} 1", &["p0"]),
        ];
        let created = |id: &str| (id != "p2").then(|| "2020-01-01T00:00:00Z".to_string());
        let o = NameOpts {
            labels: &ls,
            label_vectors: &t,
            score: K1,
            skip_clusters: &["duplicates"],
        };
        let (named, unchanged) = name_albums(&list, &x, &rows_of, &created, &o).unwrap();
        assert_eq!(unchanged, ["A \u{b7} 2020 \u{b7} 1"]);
        let titles: Vec<&str> = named.iter().map(|a| a.new_title.as_str()).collect();
        assert_eq!(
            titles,
            [
                "A \u{b7} 2020 \u{b7} 1 #2",
                "A \u{b7} 2020 \u{b7} 1 #3",
                "B \u{b7} 1"
            ]
        );
        assert_eq!(named[0].old_title, "old 1");
        assert!(named.iter().all(|a| a.local_id.is_none()));
        // An asset without an embedding fails.
        let bad = [ap("9", "x", &["nope"])];
        assert!(name_albums(&bad, &x, &rows_of, &created, &o).is_err());
    }

    #[test]
    fn cache_reuses_vectors() {
        let dir = std::env::temp_dir().join(format!("names-cache-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let cache = dir.join("label-vectors.jsonl");
        struct Fake(usize);
        impl TextEmbedder for Fake {
            fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
                self.0 += texts.len();
                Ok(texts.iter().map(|t| unit(t.len() % D)).collect())
            }
        }
        let ls = parse_labels("A | a\nB | bb\nA | ccc").unwrap();
        let calls = std::cell::Cell::new(0);
        let mut mk = || -> Result<Box<dyn TextEmbedder>> {
            calls.set(calls.get() + 1);
            Ok(Box::new(Fake(0)))
        };
        let v1 = label_vectors(&ls, &cache, "k", &mut mk, &mut |_| {}).unwrap();
        assert_eq!(v1.len(), 2 * D);
        // A = mean of unit(1) and unit(3), normalized.
        assert!((v1[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        let v2 = label_vectors(&ls, &cache, "k", &mut mk, &mut |_| {}).unwrap();
        assert_eq!(v1, v2);
        assert_eq!(calls.get(), 1, "second call must not load the model");
        // Another key embeds again.
        label_vectors(&ls, &cache, "other", &mut mk, &mut |_| {}).unwrap();
        assert_eq!(calls.get(), 2);
        let _ = fs::remove_dir_all(&dir);
    }
}
