//! Creating Photos albums for approved clusters, and renaming albums and their folder. This is the
//! only code that writes to the library. Albums sync to iCloud and show on every device of the user.

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use anyhow::{Result, bail};
use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSFastEnumeration, NSString};
use objc2_photos::{
    PHAsset, PHAssetCollection, PHAssetCollectionChangeRequest, PHCollection,
    PHCollectionEditOperation, PHCollectionList, PHCollectionListChangeRequest, PHFetchResult,
    PHPhotoLibrary,
};

/// One album to create: a title and the local identifiers of its photos.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlbumRequest {
    pub title: String,
    pub asset_ids: Vec<String>,
}

/// Check the arguments of [`create_albums`] without touching the library.
pub fn check_request(folder_name: &str, albums: &[AlbumRequest], confirm: bool) -> Result<()> {
    if !confirm {
        bail!(
            "refusing to create albums without confirm=true: albums sync to iCloud on all devices; \
             the user must approve the exact list first"
        );
    }
    if folder_name.trim().is_empty() {
        bail!("folder name is empty");
    }
    if albums.is_empty() {
        bail!("no albums to create");
    }
    let mut titles = HashSet::new();
    for a in albums {
        if a.title.trim().is_empty() {
            bail!("an album has an empty title");
        }
        if !titles.insert(a.title.as_str()) {
            bail!("album title {:?} appears twice", a.title);
        }
        if a.asset_ids.is_empty() {
            bail!("album {:?} has no photos", a.title);
        }
        let mut ids = HashSet::new();
        for id in &a.asset_ids {
            if id.trim().is_empty() {
                bail!("album {:?} has an empty asset id", a.title);
            }
            if !ids.insert(id.as_str()) {
                bail!("album {:?} lists {id} twice", a.title);
            }
        }
    }
    Ok(())
}

fn fetch_ids(ids: &[String]) -> Retained<PHFetchResult<PHAsset>> {
    let ns: Vec<Retained<NSString>> = ids.iter().map(|s| NSString::from_str(s)).collect();
    let arr = NSArray::from_retained_slice(&ns);
    // SAFETY: class method with a valid array of identifiers.
    unsafe { PHAsset::fetchAssetsWithLocalIdentifiers_options(&arr, None) }
}

/// Create one folder named `folder_name` holding one album per entry of `albums`.
///
/// Refuses unless `confirm` is `true`. Checks every asset id before any change; a missing id
/// fails the whole call. All albums are created in one `performChangesAndWait` transaction, so
/// either all exist afterwards or none. Never adds the same folder twice by itself: calling it
/// again creates a second folder. Never deletes or edits assets. Do not call from the main thread.
///
/// Returns the local identifier of the new folder.
pub fn create_albums(folder_name: &str, albums: &[AlbumRequest], confirm: bool) -> Result<String> {
    check_request(folder_name, albums, confirm)?;

    let mut results = Vec::with_capacity(albums.len());
    for a in albums {
        let r = fetch_ids(&a.asset_ids);
        // SAFETY: count on a valid fetch result.
        let found = unsafe { r.count() };
        if found != a.asset_ids.len() {
            bail!(
                "album {:?}: {} of {} photos not found in the library; nothing was created",
                a.title,
                a.asset_ids.len() - found,
                a.asset_ids.len()
            );
        }
        results.push(r);
    }

    let folder_title = NSString::from_str(folder_name);
    let titles: Vec<Retained<NSString>> = albums
        .iter()
        .map(|a| NSString::from_str(&a.title))
        .collect();
    let folder_id: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
    let block_folder_id = folder_id.clone();
    let change = RcBlock::new(move || {
        // SAFETY: change requests are only valid inside a performChanges block, which this is.
        unsafe {
            let folder = PHCollectionListChangeRequest::creationRequestForCollectionListWithTitle(
                &folder_title,
            );
            let mut placeholders = Vec::with_capacity(titles.len());
            for (title, assets) in titles.iter().zip(&results) {
                let req =
                    PHAssetCollectionChangeRequest::creationRequestForAssetCollectionWithTitle(
                        title,
                    );
                let assets: &ProtocolObject<dyn NSFastEnumeration> =
                    ProtocolObject::from_ref(&**assets);
                req.addAssets(assets);
                placeholders.push(req.placeholderForCreatedAssetCollection());
            }
            let children = NSArray::from_retained_slice(&placeholders);
            folder.addChildCollections(ProtocolObject::from_ref(&*children));
            *block_folder_id.borrow_mut() = Some(
                folder
                    .placeholderForCreatedCollectionList()
                    .localIdentifier()
                    .to_string(),
            );
        }
    });
    // SAFETY: the block outlives the call (performChangesAndWait is synchronous).
    let library = unsafe { PHPhotoLibrary::sharedPhotoLibrary() };
    unsafe { library.performChangesAndWait_error(RcBlock::as_ptr(&change)) }
        .map_err(|e| anyhow::anyhow!("Photos refused the change: {}", e.localizedDescription()))?;
    let id = folder_id.borrow_mut().take();
    id.ok_or_else(|| anyhow::anyhow!("change block did not run"))
}

/// One collection directly inside a folder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionInfo {
    /// `PHCollection` local identifier.
    pub id: String,
    pub title: String,
    /// Assets in it when it is an album; `None` for a sub-folder.
    pub assets: Option<usize>,
}

/// A top-level folder of the library and its direct children, in Photos' order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FolderListing {
    pub id: String,
    pub title: String,
    pub children: Vec<CollectionInfo>,
}

fn ns_array(ids: &[&str]) -> Retained<NSArray<NSString>> {
    let ns: Vec<Retained<NSString>> = ids.iter().map(|s| NSString::from_str(s)).collect();
    NSArray::from_retained_slice(&ns)
}

fn title_of(c: &PHCollection) -> String {
    // SAFETY: property read on a valid collection.
    unsafe { c.localizedTitle() }
        .map(|t| t.to_string())
        .unwrap_or_default()
}

fn list_children(folder: &PHCollectionList) -> Vec<CollectionInfo> {
    // SAFETY: fetches on a valid folder; indices are within count.
    unsafe {
        let r = PHCollection::fetchCollectionsInCollectionList_options(folder, None);
        let n = r.count();
        let children: Vec<Retained<PHCollection>> = (0..n).map(|i| r.objectAtIndex(i)).collect();
        let ids: Vec<String> = children
            .iter()
            .map(|c| c.localIdentifier().to_string())
            .collect();
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let albums = PHAssetCollection::fetchAssetCollectionsWithLocalIdentifiers_options(
            &ns_array(&refs),
            None,
        );
        let mut counts = std::collections::HashMap::new();
        for i in 0..albums.count() {
            let a = albums.objectAtIndex(i);
            let assets = PHAsset::fetchAssetsInAssetCollection_options(&a, None).count();
            counts.insert(a.localIdentifier().to_string(), assets);
        }
        children
            .iter()
            .zip(ids)
            .map(|(c, id)| CollectionInfo {
                title: title_of(c),
                assets: counts.get(&id).copied(),
                id,
            })
            .collect()
    }
}

/// Find the one top-level folder titled `title` and list what is directly inside it. Read-only.
/// Fails when no top-level folder or more than one has that title.
pub fn find_folder(title: &str) -> Result<FolderListing> {
    // SAFETY: read-only fetches; indices are within count.
    let found: Vec<Retained<PHCollectionList>> = unsafe {
        let r = PHCollection::fetchTopLevelUserCollectionsWithOptions(None);
        let ids: Vec<String> = (0..r.count())
            .map(|i| r.objectAtIndex(i))
            .filter(|c| c.canContainCollections() && title_of(c) == title)
            .map(|c| c.localIdentifier().to_string())
            .collect();
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let lists = PHCollectionList::fetchCollectionListsWithLocalIdentifiers_options(
            &ns_array(&refs),
            None,
        );
        (0..lists.count()).map(|i| lists.objectAtIndex(i)).collect()
    };
    match found.as_slice() {
        [] => bail!("no top-level Photos folder is titled {title:?}"),
        [f] => Ok(FolderListing {
            // SAFETY: property read on a valid folder.
            id: unsafe { f.localIdentifier() }.to_string(),
            title: title.to_string(),
            children: list_children(f),
        }),
        _ => bail!(
            "{} top-level Photos folders are titled {title:?}; rename one by hand first",
            found.len()
        ),
    }
}

/// One title change: the collection's local identifier, its title now, and its new title.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rename {
    pub id: String,
    pub from: String,
    pub to: String,
}

/// Check the arguments of [`rename_collections`] without touching the library.
pub fn check_renames(folder: &Rename, albums: &[Rename], confirm: bool) -> Result<()> {
    if !confirm {
        bail!(
            "refusing to rename albums without confirm=true: titles sync to iCloud on all \
             devices; the user must approve the exact list first"
        );
    }
    if albums.is_empty() {
        bail!("no albums to rename");
    }
    let mut ids = HashSet::new();
    let mut to = HashSet::new();
    for r in std::iter::once(folder).chain(albums) {
        if r.id.trim().is_empty() {
            bail!("a rename has an empty local identifier");
        }
        if r.from.trim().is_empty() || r.to.trim().is_empty() {
            bail!("rename of {} has an empty title", r.id);
        }
        if r.to != r.to.trim() {
            bail!("new title {:?} has leading or trailing spaces", r.to);
        }
        if !ids.insert(r.id.as_str()) {
            bail!("{} is renamed twice", r.id);
        }
    }
    for r in albums {
        if !to.insert(r.to.as_str()) {
            bail!("new album title {:?} appears twice", r.to);
        }
    }
    Ok(())
}

/// Rename the folder `folder.id` (when `from != to`) and the albums in it, in one transaction.
///
/// Refuses unless `confirm` is `true`. Before any change it checks that the folder and every album
/// still carry their `from` title, that every album is directly inside the folder, that Photos
/// allows renaming each one, and that no other album in the folder already has one of the new
/// titles. It changes titles only: no album gains or loses an asset, nothing is deleted. After the
/// change it reads every title back and fails if one differs. Do not call from the main thread.
pub fn rename_collections(folder: &Rename, albums: &[Rename], confirm: bool) -> Result<()> {
    check_renames(folder, albums, confirm)?;
    // SAFETY: read-only fetches of valid identifiers.
    let list = unsafe {
        PHCollectionList::fetchCollectionListsWithLocalIdentifiers_options(
            &ns_array(&[folder.id.as_str()]),
            None,
        )
    };
    if unsafe { list.count() } != 1 {
        bail!(
            "folder {} is not in the library; nothing was renamed",
            folder.id
        );
    }
    let list = unsafe { list.objectAtIndex(0) };
    let now = title_of(&list);
    if now != folder.from {
        bail!(
            "folder {} is titled {now:?}, expected {:?}; nothing was renamed",
            folder.id,
            folder.from
        );
    }
    let children = list_children(&list);
    let by_id: std::collections::HashMap<&str, &CollectionInfo> =
        children.iter().map(|c| (c.id.as_str(), c)).collect();
    let renamed: HashSet<&str> = albums.iter().map(|r| r.id.as_str()).collect();
    for r in albums {
        let Some(c) = by_id.get(r.id.as_str()) else {
            bail!(
                "album {} ({:?}) is not in the folder; nothing was renamed",
                r.id,
                r.from
            );
        };
        if c.assets.is_none() {
            bail!(
                "{} ({:?}) is a folder, not an album; nothing was renamed",
                r.id,
                r.from
            );
        }
        if c.title != r.from {
            bail!(
                "album {} is titled {:?}, expected {:?}; nothing was renamed",
                r.id,
                c.title,
                r.from
            );
        }
    }
    let new_titles: HashSet<&str> = albums.iter().map(|r| r.to.as_str()).collect();
    if let Some(c) = children
        .iter()
        .find(|c| !renamed.contains(c.id.as_str()) && new_titles.contains(c.title.as_str()))
    {
        bail!(
            "another collection in the folder is already titled {:?}; nothing was renamed",
            c.title
        );
    }
    let ids: Vec<&str> = albums.iter().map(|r| r.id.as_str()).collect();
    // SAFETY: read-only fetch of valid identifiers; indices are within count.
    let collections: std::collections::HashMap<String, Retained<PHAssetCollection>> = unsafe {
        let r = PHAssetCollection::fetchAssetCollectionsWithLocalIdentifiers_options(
            &ns_array(&ids),
            None,
        );
        (0..r.count())
            .map(|i| r.objectAtIndex(i))
            .map(|a| (a.localIdentifier().to_string(), a))
            .collect()
    };
    let mut targets = Vec::with_capacity(albums.len());
    for r in albums {
        let a = collections
            .get(&r.id)
            .ok_or_else(|| anyhow::anyhow!("album {} vanished; nothing was renamed", r.id))?;
        // SAFETY: property read on a valid collection.
        if !unsafe { a.canPerformEditOperation(PHCollectionEditOperation::Rename) } {
            bail!(
                "Photos does not allow renaming album {:?}; nothing was renamed",
                r.from
            );
        }
        targets.push((a.clone(), NSString::from_str(&r.to)));
    }
    let rename_folder = folder.from != folder.to;
    if rename_folder && !unsafe { list.canPerformEditOperation(PHCollectionEditOperation::Rename) }
    {
        bail!(
            "Photos does not allow renaming the folder {:?}; nothing was renamed",
            folder.from
        );
    }
    let folder_title = NSString::from_str(&folder.to);
    let block_list = list.clone();
    let failed: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
    let block_failed = failed.clone();
    let change = RcBlock::new(move || {
        // SAFETY: change requests are only valid inside a performChanges block, which this is.
        unsafe {
            for (a, title) in &targets {
                match PHAssetCollectionChangeRequest::changeRequestForAssetCollection(a) {
                    Some(req) => req.setTitle(title),
                    None => {
                        *block_failed.borrow_mut() = Some(format!(
                            "no change request for album {}",
                            a.localIdentifier()
                        ));
                    }
                }
            }
            if rename_folder {
                match PHCollectionListChangeRequest::changeRequestForCollectionList(&block_list) {
                    Some(req) => req.setTitle(&folder_title),
                    None => {
                        *block_failed.borrow_mut() =
                            Some("no change request for the folder".into());
                    }
                }
            }
        }
    });
    let library = unsafe { PHPhotoLibrary::sharedPhotoLibrary() };
    // SAFETY: the block outlives the call (performChangesAndWait is synchronous).
    unsafe { library.performChangesAndWait_error(RcBlock::as_ptr(&change)) }
        .map_err(|e| anyhow::anyhow!("Photos refused the change: {}", e.localizedDescription()))?;
    if let Some(e) = failed.borrow_mut().take() {
        bail!("{e}: Photos applied the other renames; run the same command to see the state");
    }
    verify_renamed(folder, albums)
}

/// Read the folder and albums back and check every title is the new one.
pub fn verify_renamed(folder: &Rename, albums: &[Rename]) -> Result<()> {
    let listing = find_folder(&folder.to)?;
    if listing.id != folder.id {
        bail!(
            "the folder titled {:?} is {}, expected {}",
            folder.to,
            listing.id,
            folder.id
        );
    }
    let by_id: std::collections::HashMap<&str, &str> = listing
        .children
        .iter()
        .map(|c| (c.id.as_str(), c.title.as_str()))
        .collect();
    let wrong: Vec<String> = albums
        .iter()
        .filter(|r| by_id.get(r.id.as_str()) != Some(&r.to.as_str()))
        .map(|r| format!("{} ({:?})", r.id, r.to))
        .collect();
    if !wrong.is_empty() {
        bail!(
            "{} albums do not have their new title after the change (first: {})",
            wrong.len(),
            wrong[0]
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // Only argument checks. Nothing here may reach `create_albums`'s PhotoKit calls.
    use super::*;

    fn album(title: &str, ids: &[&str]) -> AlbumRequest {
        AlbumRequest {
            title: title.into(),
            asset_ids: ids.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn refuses_without_confirm() {
        let e = check_request("emeans", &[album("a", &["x"])], false).unwrap_err();
        assert!(e.to_string().contains("confirm=true"));
    }

    #[test]
    fn accepts_a_valid_request() {
        check_request(
            "emeans",
            &[album("a", &["x", "y"]), album("b", &["x"])],
            true,
        )
        .unwrap();
    }

    fn rn(id: &str, from: &str, to: &str) -> Rename {
        Rename {
            id: id.into(),
            from: from.into(),
            to: to.into(),
        }
    }

    #[test]
    fn rename_checks() {
        let f = rn("F", "emeans", "Photo groups");
        let ok = [rn("a", "x", "Beach"), rn("b", "y", "Cats")];
        check_renames(&f, &ok, true).unwrap();
        // The folder may keep its title.
        check_renames(&rn("F", "emeans", "emeans"), &ok, true).unwrap();
        let e = check_renames(&f, &ok, false).unwrap_err();
        assert!(e.to_string().contains("confirm=true"));
        assert!(check_renames(&f, &[], true).is_err());
        assert!(check_renames(&f, &[rn("a", "x", "")], true).is_err());
        assert!(check_renames(&f, &[rn("a", "", "B")], true).is_err());
        assert!(check_renames(&f, &[rn("a", "x", " B")], true).is_err());
        assert!(check_renames(&f, &[rn("", "x", "B")], true).is_err());
        assert!(check_renames(&f, &[rn("a", "x", "B"), rn("a", "y", "C")], true).is_err());
        assert!(check_renames(&f, &[rn("a", "x", "B"), rn("b", "y", "B")], true).is_err());
        assert!(check_renames(&f, &[rn("F", "x", "B")], true).is_err());
    }

    #[test]
    fn rejects_bad_arguments() {
        assert!(check_request(" ", &[album("a", &["x"])], true).is_err());
        assert!(check_request("f", &[], true).is_err());
        assert!(check_request("f", &[album("", &["x"])], true).is_err());
        assert!(check_request("f", &[album("a", &["x"]), album("a", &["y"])], true).is_err());
        assert!(check_request("f", &[album("a", &[])], true).is_err());
        assert!(check_request("f", &[album("a", &["x", "x"])], true).is_err());
        assert!(check_request("f", &[album("a", &[""])], true).is_err());
    }
}
