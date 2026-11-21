use std::collections::HashMap as StdHashMap;
use std::env;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use arrow_array::builder::{Date32Builder, Int64Builder, StringBuilder};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use hashbrown::HashMap;
use memchr::memchr;
use memmap2::Mmap;
use mimalloc::MiMalloc;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use serde::Deserialize;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const MS_PER_DAY: i64 = 86_400_000;
const MS_24H: i64 = 86_400_000;
const MS_30M: i64 = 1_800_000;
const MS_7D: i64 = 604_800_000;
const INVALID_TS: i64 = i64::MIN;

const EV_BUCKETS: usize = 64;
const PERSON_BUCKETS: usize = 128;
const CHUNK_SIZE: usize = 4 * 1024 * 1024;
const MAX_MMAP_CHUNK: usize = 1 << 31;
const PARQUET_BATCH_ROWS: usize = 131_072;

#[derive(Deserialize)]
struct Manifest {
    window_start: String,
    window_end: String,
}

#[inline(always)]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

#[inline(always)]
fn hash_bytes(bytes: &[u8]) -> u64 {
    const K: u64 = 0x517cc1b727220a95;
    let mut h = (bytes.len() as u64).wrapping_mul(K) ^ 0x9e3779b97f4a7c15;
    let mut i = 0;
    while i + 8 <= bytes.len() {
        let w = u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
        h = (h.rotate_left(5) ^ w).wrapping_mul(K);
        i += 8;
    }
    if i + 4 <= bytes.len() {
        let w = u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as u64;
        h = (h.rotate_left(5) ^ w).wrapping_mul(K);
        i += 4;
    }
    while i < bytes.len() {
        h = (h.rotate_left(5) ^ (bytes[i] as u64)).wrapping_mul(K);
        i += 1;
    }
    mix64(h)
}

#[inline(always)]
fn hash_linked_id(pid: u32) -> u64 {
    mix64((pid as u64).wrapping_add(0x9e3779b97f4a7c15))
}

// ---------------------------------------------------------------------------
// Zero-Copy Mmap Store & Packed SliceRef / PersonKey
// ---------------------------------------------------------------------------

struct MmapChunk {
    mmap_idx: usize,
    base: usize,
    len: usize,
}

struct MmapStore {
    mmaps: Vec<Mmap>,
    chunks: Vec<MmapChunk>,
}

impl MmapStore {
    fn new() -> Self {
        Self {
            mmaps: Vec::new(),
            chunks: Vec::new(),
        }
    }

    fn add_csv_file(&mut self, path: &Path) -> Vec<(u16, usize, usize)> {
        let file = File::open(path).expect("failed to open csv");
        let meta = file.metadata().expect("failed to stat csv");
        if meta.len() == 0 {
            return Vec::new();
        }
        let mmap = unsafe { Mmap::map(&file).expect("mmap failed") };
        let data: &[u8] = &mmap;
        let header_end = match memchr(b'\n', data) {
            Some(pos) => pos + 1,
            None => return Vec::new(),
        };
        if header_end >= data.len() {
            return Vec::new();
        }

        let mmap_idx = self.mmaps.len();
        let file_len = data.len();
        let mut work = Vec::new();
        let mut v_start = header_end;

        while v_start < file_len {
            let mut v_end = (v_start + MAX_MMAP_CHUNK).min(file_len);
            if v_end < file_len {
                match memchr(b'\n', &data[v_end..]) {
                    Some(nl) => v_end += nl + 1,
                    None => v_end = file_len,
                }
            }
            let chunk_id = self.chunks.len() as u16;
            assert!((chunk_id as usize) < 32768, "too many virtual file chunks");
            let v_len = v_end - v_start;
            let v_slice = &data[v_start..v_end];
            self.chunks.push(MmapChunk {
                mmap_idx,
                base: v_start,
                len: v_len,
            });

            let mut pos = 0usize;
            while pos < v_len {
                let mut end = (pos + CHUNK_SIZE).min(v_len);
                if end < v_len {
                    match memchr(b'\n', &v_slice[end..]) {
                        Some(nl) => end += nl + 1,
                        None => end = v_len,
                    }
                }
                work.push((chunk_id, pos, end));
                pos = end;
            }
            v_start = v_end;
        }

        self.mmaps.push(mmap);
        work
    }

    #[inline(always)]
    fn chunk_bytes(&self, chunk_id: u16) -> &[u8] {
        let c = &self.chunks[chunk_id as usize];
        &self.mmaps[c.mmap_idx][c.base..c.base + c.len]
    }

    #[inline(always)]
    fn get_slice(&self, sref: u64) -> &[u8] {
        let offset = (sref as u32) as usize;
        let len = ((sref >> 32) as u16) as usize;
        let chunk_id = ((sref >> 48) as u16) & 0x7FFF;
        let c = &self.chunks[chunk_id as usize];
        let start = c.base + offset;
        &self.mmaps[c.mmap_idx][start..start + len]
    }
}

#[inline(always)]
fn make_slice_ref(chunk_id: u16, offset: usize, len: usize) -> u64 {
    (offset as u64) | ((len as u64) << 32) | ((chunk_id as u64) << 48)
}

const LINKED_BIT: u64 = 1u64 << 63;

#[inline(always)]
fn make_linked_person(pid: u32) -> u64 {
    LINKED_BIT | (pid as u64)
}

#[inline(always)]
fn is_linked_person(pkey: u64) -> bool {
    (pkey & LINKED_BIT) != 0
}

#[inline(always)]
fn linked_person_id(pkey: u64) -> u32 {
    pkey as u32
}

// ---------------------------------------------------------------------------
// Strict CSV Dialect, Timestamp & Price Parsers (per rules.md)
// ---------------------------------------------------------------------------

#[inline(always)]
fn unquote(f: &[u8]) -> Option<&[u8]> {
    if f.is_empty() {
        return Some(f);
    }
    if f[0] == b'"' {
        if f.len() >= 2 && f[f.len() - 1] == b'"' {
            let inner = &f[1..f.len() - 1];
            if memchr(b'"', inner).is_none() {
                return Some(inner);
            }
        }
        None
    } else if memchr(b'"', f).is_none() {
        Some(f)
    } else {
        None
    }
}

#[inline(always)]
fn split_2(line: &[u8]) -> Option<[&[u8]; 2]> {
    let c1 = memchr(b',', line)?;
    let r1 = &line[c1 + 1..];
    if memchr(b',', r1).is_some() {
        return None;
    }
    Some([unquote(&line[..c1])?, unquote(r1)?])
}

#[inline(always)]
fn split_4(line: &[u8]) -> Option<[&[u8]; 4]> {
    let c1 = memchr(b',', line)?;
    let s1 = &line[c1 + 1..];
    let c2 = memchr(b',', s1)?;
    let s2 = &s1[c2 + 1..];
    let c3 = memchr(b',', s2)?;
    let s3 = &s2[c3 + 1..];
    if memchr(b',', s3).is_some() {
        return None;
    }
    Some([
        unquote(&line[..c1])?,
        unquote(&s1[..c2])?,
        unquote(&s2[..c3])?,
        unquote(s3)?,
    ])
}

#[inline(always)]
fn split_7(line: &[u8]) -> Option<[&[u8]; 7]> {
    let c1 = memchr(b',', line)?;
    let s1 = &line[c1 + 1..];
    let c2 = memchr(b',', s1)?;
    let s2 = &s1[c2 + 1..];
    let c3 = memchr(b',', s2)?;
    let s3 = &s2[c3 + 1..];
    let c4 = memchr(b',', s3)?;
    let s4 = &s3[c4 + 1..];
    let c5 = memchr(b',', s4)?;
    let s5 = &s4[c5 + 1..];
    let c6 = memchr(b',', s5)?;
    let s6 = &s5[c6 + 1..];
    if memchr(b',', s6).is_some() {
        return None;
    }
    Some([
        unquote(&line[..c1])?,
        unquote(&s1[..c2])?,
        unquote(&s2[..c3])?,
        unquote(&s3[..c4])?,
        unquote(&s4[..c5])?,
        unquote(&s5[..c6])?,
        unquote(s6)?,
    ])
}

#[inline(always)]
fn parse_2digits(b0: u8, b1: u8) -> Option<u32> {
    let d0 = b0.wrapping_sub(b'0');
    let d1 = b1.wrapping_sub(b'0');
    if d0 <= 9 && d1 <= 9 {
        Some((d0 as u32) * 10 + (d1 as u32))
    } else {
        None
    }
}

#[inline(always)]
fn parse_4digits(b: &[u8]) -> Option<i64> {
    let d0 = b[0].wrapping_sub(b'0');
    let d1 = b[1].wrapping_sub(b'0');
    let d2 = b[2].wrapping_sub(b'0');
    let d3 = b[3].wrapping_sub(b'0');
    if d0 <= 9 && d1 <= 9 && d2 <= 9 && d3 <= 9 {
        Some((d0 as i64) * 1000 + (d1 as i64) * 100 + (d2 as i64) * 10 + (d3 as i64))
    } else {
        None
    }
}

#[inline(always)]
fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

#[inline(always)]
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = y - if m <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let m = m as u64;
    let d = d as u64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + (doe as i64) - 719_468
}

#[inline(always)]
fn parse_iso_prefix_19(s: &[u8]) -> Option<i64> {
    if s[4] != b'-' || s[7] != b'-' || s[10] != b'T' || s[13] != b':' || s[16] != b':' {
        return None;
    }
    let y = parse_4digits(s)?;
    let m = parse_2digits(s[5], s[6])?;
    if m < 1 || m > 12 {
        return None;
    }
    let d = parse_2digits(s[8], s[9])?;
    if d < 1 || d > days_in_month(y, m) {
        return None;
    }
    let hh = parse_2digits(s[11], s[12])?;
    let mm = parse_2digits(s[14], s[15])?;
    let ss = parse_2digits(s[17], s[18])?;
    if hh > 23 || mm > 59 || ss > 59 {
        return None;
    }
    let days = days_from_civil(y, m, d);
    Some((days * 86_400 + (hh as i64) * 3600 + (mm as i64) * 60 + (ss as i64)) * 1000)
}

#[inline(always)]
fn parse_ts_f1(s: &[u8]) -> Option<i64> {
    if s.len() != 20 || s[19] != b'Z' {
        return None;
    }
    parse_iso_prefix_19(s)
}

#[inline(always)]
fn parse_ts(s: &[u8]) -> Option<i64> {
    let len = s.len();
    if len == 0 {
        return None;
    }
    if len == 20 && s[19] == b'Z' {
        return parse_iso_prefix_19(s);
    }
    if len == 25 && (s[19] == b'+' || s[19] == b'-') && s[22] == b':' {
        let base_ms = parse_iso_prefix_19(s)?;
        let oh = parse_2digits(s[20], s[21])?;
        let om = parse_2digits(s[23], s[24])?;
        if oh > 23 || om > 59 {
            return None;
        }
        let off_ms = ((oh as i64) * 3600 + (om as i64) * 60) * 1000;
        return Some(if s[19] == b'+' {
            base_ms - off_ms
        } else {
            base_ms + off_ms
        });
    }
    let mut val: u64 = 0;
    for &b in s {
        let d = b.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        val = val.checked_mul(10)?.checked_add(d as u64)?;
        if val > i64::MAX as u64 {
            return None;
        }
    }
    Some(val as i64)
}

#[inline(always)]
fn parse_price_cents(s: &[u8]) -> Option<u32> {
    if s.is_empty() {
        return None;
    }
    let mut val: u32 = 0;
    for &b in s {
        let d = b.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        val = val.saturating_mul(10).saturating_add(d as u32);
        if val > 10_000_000 {
            return None;
        }
    }
    if val == 0 {
        None
    } else {
        Some(val)
    }
}

// ---------------------------------------------------------------------------
// Linked IDs (DSU + Flat Open-Addressing Lookup Table)
// ---------------------------------------------------------------------------

#[derive(Copy, Clone)]
struct LinkSlot {
    tag: u32,
    idx: u32,
}

struct LinkEntry {
    off: u32,
    len: u32,
    person_id: u32,
}

struct LinkedMap {
    mask: usize,
    slots: Vec<LinkSlot>,
    entries: Vec<LinkEntry>,
    arena: Vec<u8>,
    canon_names: Vec<(u32, u32)>,
    canon_arena: Vec<u8>,
}

impl LinkedMap {
    fn empty() -> Self {
        Self {
            mask: 0,
            slots: vec![LinkSlot { tag: 0, idx: 0 }],
            entries: Vec::new(),
            arena: Vec::new(),
            canon_names: Vec::new(),
            canon_arena: Vec::new(),
        }
    }

    #[inline(always)]
    fn lookup(&self, user_id: &[u8]) -> Option<u32> {
        if self.entries.is_empty() {
            return None;
        }
        let h = hash_bytes(user_id);
        let tag = ((h >> 32) as u32) | 1;
        let mut pos = (h as usize) & self.mask;
        loop {
            let slot = unsafe { *self.slots.get_unchecked(pos) };
            if slot.tag == 0 {
                return None;
            }
            if slot.tag == tag {
                let e = unsafe { self.entries.get_unchecked(slot.idx as usize) };
                if &self.arena[e.off as usize..(e.off + e.len) as usize] == user_id {
                    return Some(e.person_id);
                }
            }
            pos = (pos + 1) & self.mask;
        }
    }

    #[inline(always)]
    fn canon_name(&self, pid: u32) -> &str {
        let (off, len) = self.canon_names[pid as usize];
        std::str::from_utf8(&self.canon_arena[off as usize..(off + len) as usize])
            .expect("invalid utf8 user_id")
    }
}

struct Dsu {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl Dsu {
    fn new() -> Self {
        Self {
            parent: Vec::new(),
            size: Vec::new(),
        }
    }

    #[inline(always)]
    fn push(&mut self) -> u32 {
        let id = self.parent.len() as u32;
        self.parent.push(id);
        self.size.push(1);
        id
    }

    #[inline(always)]
    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            self.parent[x as usize] = self.parent[p as usize];
            x = self.parent[x as usize];
        }
        x
    }

    #[inline(always)]
    fn union(&mut self, a: u32, b: u32) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.size[ra as usize] < self.size[rb as usize] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb as usize] = ra;
        self.size[ra as usize] += self.size[rb as usize];
    }
}

fn load_links(links_dir: &Path, ncpu: usize) -> LinkedMap {
    let files = list_csv_files(links_dir);
    if files.is_empty() {
        return LinkedMap::empty();
    }

    let mut store = MmapStore::new();
    let mut work = Vec::new();
    for f in &files {
        work.extend(store.add_csv_file(f));
    }
    if work.is_empty() {
        return LinkedMap::empty();
    }

    let work_idx = AtomicUsize::new(0);
    let mut thread_pairs: Vec<Vec<(u64, u64)>> = (0..ncpu).map(|_| Vec::new()).collect();
    std::thread::scope(|s| {
        for out in &mut thread_pairs {
            let store_ref = &store;
            let work_ref = &work;
            let idx_ref = &work_idx;
            s.spawn(move || loop {
                let wi = idx_ref.fetch_add(1, Ordering::Relaxed);
                if wi >= work_ref.len() {
                    break;
                }
                let (chunk_id, start, end) = work_ref[wi];
                let chunk = store_ref.chunk_bytes(chunk_id);
                let base_ptr = chunk.as_ptr() as usize;
                let mut pos = start;
                while pos < end {
                    let nl = memchr(b'\n', &chunk[pos..end])
                        .map(|k| pos + k)
                        .unwrap_or(end);
                    let line = &chunk[pos..nl];
                    pos = nl + 1;
                    if let Some([a, b]) = split_2(line) {
                        if !a.is_empty() && !b.is_empty() && a != b {
                            out.push((
                                make_slice_ref(
                                    chunk_id,
                                    (a.as_ptr() as usize) - base_ptr,
                                    a.len(),
                                ),
                                make_slice_ref(
                                    chunk_id,
                                    (b.as_ptr() as usize) - base_ptr,
                                    b.len(),
                                ),
                            ));
                        }
                    }
                }
            });
        }
    });

    let total_pairs: usize = thread_pairs.iter().map(|v| v.len()).sum();
    if total_pairs == 0 {
        return LinkedMap::empty();
    }

    let mut id_map: HashMap<&[u8], u32> = HashMap::with_capacity(total_pairs);
    let mut id_slices: Vec<&[u8]> = Vec::with_capacity(total_pairs);
    let mut dsu = Dsu::new();

    for pairs in &thread_pairs {
        for &(ra, rb) in pairs {
            let sa = store.get_slice(ra);
            let sb = store.get_slice(rb);
            let ia = match id_map.get(sa) {
                Some(&id) => id,
                None => {
                    let id = dsu.push();
                    id_slices.push(sa);
                    id_map.insert(sa, id);
                    id
                }
            };
            let ib = match id_map.get(sb) {
                Some(&id) => id,
                None => {
                    let id = dsu.push();
                    id_slices.push(sb);
                    id_map.insert(sb, id);
                    id
                }
            };
            dsu.union(ia, ib);
        }
    }

    let n_nodes = id_slices.len();
    let mut root_to_pid = vec![u32::MAX; n_nodes];
    let mut comp_min: Vec<&[u8]> = Vec::new();
    let mut node_pid = vec![0u32; n_nodes];

    for i in 0..n_nodes {
        let r = dsu.find(i as u32) as usize;
        let pid = if root_to_pid[r] == u32::MAX {
            let p = comp_min.len() as u32;
            root_to_pid[r] = p;
            comp_min.push(id_slices[i]);
            p
        } else {
            let p = root_to_pid[r];
            if id_slices[i] < comp_min[p as usize] {
                comp_min[p as usize] = id_slices[i];
            }
            p
        };
        node_pid[i] = pid;
    }

    let mut canon_arena = Vec::with_capacity(comp_min.iter().map(|s| s.len()).sum());
    let mut canon_names = Vec::with_capacity(comp_min.len());
    for s in comp_min {
        let off = canon_arena.len() as u32;
        canon_arena.extend_from_slice(s);
        canon_names.push((off, s.len() as u32));
    }

    let mut arena = Vec::with_capacity(id_slices.iter().map(|s| s.len()).sum());
    let mut entries = Vec::with_capacity(n_nodes);
    let cap = (n_nodes * 2).next_power_of_two().max(16);
    let mask = cap - 1;
    let mut slots = vec![LinkSlot { tag: 0, idx: 0 }; cap];

    for i in 0..n_nodes {
        let s = id_slices[i];
        let off = arena.len() as u32;
        arena.extend_from_slice(s);
        let entry_idx = entries.len() as u32;
        entries.push(LinkEntry {
            off,
            len: s.len() as u32,
            person_id: node_pid[i],
        });
        let h = hash_bytes(s);
        let tag = ((h >> 32) as u32) | 1;
        let mut pos = (h as usize) & mask;
        while slots[pos].tag != 0 {
            pos = (pos + 1) & mask;
        }
        slots[pos] = LinkSlot {
            tag,
            idx: entry_idx,
        };
    }

    LinkedMap {
        mask,
        slots,
        entries,
        arena,
        canon_names,
        canon_arena,
    }
}

// ---------------------------------------------------------------------------
// Reference Tables: Products & Campaigns
// ---------------------------------------------------------------------------

struct Products {
    cat_names: Vec<String>,
    unknown_cat_id: u16,
    map: HashMap<Vec<u8>, u16>,
}

fn load_products(path: &Path) -> Products {
    let data = fs::read(path).expect("failed to read products.csv");
    let mut raw_pairs: Vec<(&[u8], &[u8])> = Vec::new();
    let mut cat_set: HashMap<&[u8], ()> = HashMap::new();
    cat_set.insert(b"unknown", ());

    if let Some(first_nl) = memchr(b'\n', &data) {
        let mut pos = first_nl + 1;
        while pos < data.len() {
            let nl = memchr(b'\n', &data[pos..])
                .map(|k| pos + k)
                .unwrap_or(data.len());
            let line = &data[pos..nl];
            pos = nl + 1;
            if let Some([pid, cat]) = split_2(line) {
                if !pid.is_empty() {
                    raw_pairs.push((pid, cat));
                    cat_set.insert(cat, ());
                }
            }
        }
    }

    let mut sorted_cats: Vec<&[u8]> = cat_set.keys().copied().collect();
    sorted_cats.sort_unstable();

    let mut cat_to_id: HashMap<&[u8], u16> = HashMap::with_capacity(sorted_cats.len());
    let mut cat_names = Vec::with_capacity(sorted_cats.len());
    let mut unknown_cat_id = 0u16;
    for (idx, c) in sorted_cats.into_iter().enumerate() {
        let cid = idx as u16;
        if c == b"unknown" {
            unknown_cat_id = cid;
        }
        cat_to_id.insert(c, cid);
        cat_names.push(std::str::from_utf8(c).expect("utf8 cat").to_owned());
    }

    let mut map = HashMap::with_capacity(raw_pairs.len());
    for (pid, cat) in raw_pairs {
        map.insert(pid.to_vec(), cat_to_id[cat]);
    }

    Products {
        cat_names,
        unknown_cat_id,
        map,
    }
}

struct Campaigns {
    ids: Vec<String>,
    channels: Vec<String>,
    organic_id: u32,
    map: HashMap<Vec<u8>, u32>,
}

fn load_campaigns(path: &Path) -> Campaigns {
    let data = fs::read(path).expect("failed to read campaigns.csv");
    let mut ids = Vec::new();
    let mut channels = Vec::new();
    let mut map = HashMap::new();

    if let Some(first_nl) = memchr(b'\n', &data) {
        let mut pos = first_nl + 1;
        while pos < data.len() {
            let nl = memchr(b'\n', &data[pos..])
                .map(|k| pos + k)
                .unwrap_or(data.len());
            let line = &data[pos..nl];
            pos = nl + 1;
            if let Some([cid, ch]) = split_2(line) {
                if !cid.is_empty() {
                    let idx = ids.len() as u32;
                    ids.push(std::str::from_utf8(cid).expect("utf8 cid").to_owned());
                    channels.push(std::str::from_utf8(ch).expect("utf8 ch").to_owned());
                    map.insert(cid.to_vec(), idx);
                }
            }
        }
    }

    let organic_id = ids.len() as u32;
    ids.push("organic".to_owned());
    channels.push("organic".to_owned());

    Campaigns {
        ids,
        channels,
        organic_id,
        map,
    }
}

// ---------------------------------------------------------------------------
// Pipeline Structs & Accumulators
// ---------------------------------------------------------------------------

#[derive(Copy, Clone)]
struct RawEvent {
    ev_hash: u64,
    ev_id: u64,
    ingest_ts: i64,
    ts: i64,
    user_ref: u64,
    price_cents: u32,
    cat_id: u16,
}

#[derive(Copy, Clone)]
struct CleanEvent {
    ts: i64,
    user_ref: u64,
    price_cents: u32,
    cat_id: u16,
}

#[derive(Copy, Clone)]
struct CleanClick {
    ts: i64,
    user_ref: u64,
    click_id: u64,
    camp_id: u32,
}

#[derive(Copy, Clone)]
struct PerPersonEvent {
    ts: i64,
    price_cents: u32,
    cat_id: u16,
}

#[derive(Copy, Clone)]
struct PerPersonClick {
    ts: i64,
    click_id: u64,
    camp_id: u32,
}

#[derive(Copy, Clone, Default)]
struct CampDayStats {
    sessions: i64,
    purchasing_sessions: i64,
    revenue_cents: i64,
    assisted_sessions: i64,
}

impl CampDayStats {
    #[inline(always)]
    fn is_non_zero(&self) -> bool {
        self.sessions != 0 || self.assisted_sessions != 0
    }

    #[inline(always)]
    fn add_assign(&mut self, other: &CampDayStats) {
        self.sessions += other.sessions;
        self.purchasing_sessions += other.purchasing_sessions;
        self.revenue_cents += other.revenue_cents;
        self.assisted_sessions += other.assisted_sessions;
    }
}

struct CampDailyAcc {
    num_camps: usize,
    min_day: i32,
    num_days: usize,
    dense: Vec<CampDayStats>,
    sparse: StdHashMap<(u32, i32), CampDayStats>,
}

impl CampDailyAcc {
    fn new(num_camps: usize, min_day: i32, max_day: i32) -> Self {
        let span = if max_day >= min_day {
            (max_day as i64 - min_day as i64 + 1) as usize
        } else {
            0
        };
        if span > 0 && span <= 4096 && num_camps.saturating_mul(span) <= 20_000_000 {
            Self {
                num_camps,
                min_day,
                num_days: span,
                dense: vec![CampDayStats::default(); num_camps * span],
                sparse: StdHashMap::new(),
            }
        } else {
            Self {
                num_camps,
                min_day,
                num_days: 0,
                dense: Vec::new(),
                sparse: StdHashMap::new(),
            }
        }
    }

    #[inline(always)]
    fn credit(&mut self, camp_id: u32, day: i32, rev: i64, has_purchase: bool) {
        if self.num_days > 0 {
            let d = (day - self.min_day) as usize;
            if d < self.num_days {
                let s = unsafe {
                    self.dense
                        .get_unchecked_mut(camp_id as usize * self.num_days + d)
                };
                s.sessions += 1;
                s.purchasing_sessions += has_purchase as i64;
                s.revenue_cents += rev;
                return;
            }
        }
        let s = self.sparse.entry((camp_id, day)).or_default();
        s.sessions += 1;
        s.purchasing_sessions += has_purchase as i64;
        s.revenue_cents += rev;
    }

    #[inline(always)]
    fn add_assist(&mut self, camp_id: u32, day: i32, count: i64) {
        if count == 0 {
            return;
        }
        if self.num_days > 0 {
            let d = (day - self.min_day) as usize;
            if d < self.num_days {
                let s = unsafe {
                    self.dense
                        .get_unchecked_mut(camp_id as usize * self.num_days + d)
                };
                s.assisted_sessions += count;
                return;
            }
        }
        self.sparse.entry((camp_id, day)).or_default().assisted_sessions += count;
    }

    fn merge_from(&mut self, other: &CampDailyAcc) {
        for (a, b) in self.dense.iter_mut().zip(other.dense.iter()) {
            a.add_assign(b);
        }
        for (&k, v) in &other.sparse {
            self.sparse.entry(k).or_default().add_assign(v);
        }
    }
}

#[derive(Copy, Clone)]
struct PersonSlot {
    tag: u32,
    idx: u32,
}

struct LocalPersonTable {
    mask: usize,
    slots: Vec<PersonSlot>,
    occupied: Vec<u32>,
    keys: Vec<u64>,
}

impl LocalPersonTable {
    fn new() -> Self {
        Self {
            mask: 0,
            slots: Vec::new(),
            occupied: Vec::new(),
            keys: Vec::new(),
        }
    }

    fn prepare(&mut self, max_items: usize) {
        let cap = (max_items * 2).next_power_of_two().max(256);
        if self.slots.len() < cap {
            self.slots.resize(cap, PersonSlot { tag: 0, idx: 0 });
        }
        self.mask = cap - 1;
        self.occupied.clear();
        self.keys.clear();
    }

    fn reset(&mut self) {
        for &pos in &self.occupied {
            self.slots[pos as usize].tag = 0;
        }
        self.occupied.clear();
        self.keys.clear();
    }

    #[inline(always)]
    fn key_hash(pkey: u64, store: &MmapStore) -> u64 {
        if is_linked_person(pkey) {
            hash_linked_id(linked_person_id(pkey))
        } else {
            hash_bytes(store.get_slice(pkey))
        }
    }

    #[inline(always)]
    fn keys_equal(a: u64, b: u64, store: &MmapStore) -> bool {
        if a == b {
            return true;
        }
        if is_linked_person(a) || is_linked_person(b) {
            return false;
        }
        store.get_slice(a) == store.get_slice(b)
    }

    #[inline(always)]
    fn get_or_insert(&mut self, pkey: u64, store: &MmapStore) -> u32 {
        let h = Self::key_hash(pkey, store);
        let tag = ((h >> 25) as u32) | 1;
        let mut pos = (h as usize) & self.mask;
        loop {
            let slot = unsafe { *self.slots.get_unchecked(pos) };
            if slot.tag == 0 {
                let idx = self.keys.len() as u32;
                self.keys.push(pkey);
                unsafe {
                    *self.slots.get_unchecked_mut(pos) = PersonSlot { tag, idx };
                }
                self.occupied.push(pos as u32);
                return idx;
            }
            if slot.tag == tag {
                let existing = unsafe { *self.keys.get_unchecked(slot.idx as usize) };
                if Self::keys_equal(existing, pkey, store) {
                    return slot.idx;
                }
            }
            pos = (pos + 1) & self.mask;
        }
    }

    #[inline(always)]
    fn get(&self, pkey: u64, store: &MmapStore) -> Option<u32> {
        let h = Self::key_hash(pkey, store);
        let tag = ((h >> 25) as u32) | 1;
        let mut pos = (h as usize) & self.mask;
        loop {
            let slot = unsafe { *self.slots.get_unchecked(pos) };
            if slot.tag == 0 {
                return None;
            }
            if slot.tag == tag {
                let existing = unsafe { *self.keys.get_unchecked(slot.idx as usize) };
                if Self::keys_equal(existing, pkey, store) {
                    return Some(slot.idx);
                }
            }
            pos = (pos + 1) & self.mask;
        }
    }
}

// ---------------------------------------------------------------------------
// Fast Uncompressed Parquet Writers
// ---------------------------------------------------------------------------

fn fast_writer_props() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_dictionary_enabled(false)
        .set_statistics_enabled(EnabledStatistics::None)
        .set_write_batch_size(65_536)
        .set_max_row_group_size(262_144)
        .build()
}

fn user_daily_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("user_id", DataType::Utf8, false),
        Field::new("day", DataType::Date32, false),
        Field::new("sessions", DataType::Int64, false),
        Field::new("events", DataType::Int64, false),
        Field::new("revenue_cents", DataType::Int64, false),
        Field::new("top_category", DataType::Utf8, true),
    ]))
}

fn campaign_daily_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("campaign_id", DataType::Utf8, false),
        Field::new("channel", DataType::Utf8, false),
        Field::new("day", DataType::Date32, false),
        Field::new("sessions", DataType::Int64, false),
        Field::new("purchasing_sessions", DataType::Int64, false),
        Field::new("revenue_cents", DataType::Int64, false),
        Field::new("assisted_sessions", DataType::Int64, false),
    ]))
}

struct UserDailyWriter {
    path: PathBuf,
    schema: Arc<Schema>,
    writer: Option<ArrowWriter<BufWriter<File>>>,
    user_id: StringBuilder,
    day: Date32Builder,
    sessions: Int64Builder,
    events: Int64Builder,
    revenue_cents: Int64Builder,
    top_category: StringBuilder,
    pending: usize,
    total_rows: usize,
}

impl UserDailyWriter {
    fn new(path: PathBuf, schema: Arc<Schema>) -> Self {
        Self {
            path,
            schema,
            writer: None,
            user_id: StringBuilder::with_capacity(PARQUET_BATCH_ROWS, PARQUET_BATCH_ROWS * 16),
            day: Date32Builder::with_capacity(PARQUET_BATCH_ROWS),
            sessions: Int64Builder::with_capacity(PARQUET_BATCH_ROWS),
            events: Int64Builder::with_capacity(PARQUET_BATCH_ROWS),
            revenue_cents: Int64Builder::with_capacity(PARQUET_BATCH_ROWS),
            top_category: StringBuilder::with_capacity(PARQUET_BATCH_ROWS, PARQUET_BATCH_ROWS * 8),
            pending: 0,
            total_rows: 0,
        }
    }

    #[inline(always)]
    fn push(
        &mut self,
        uid: &str,
        day: i32,
        sessions: i64,
        events: i64,
        revenue_cents: i64,
        top_cat: Option<&str>,
    ) {
        self.user_id.append_value(uid);
        self.day.append_value(day);
        self.sessions.append_value(sessions);
        self.events.append_value(events);
        self.revenue_cents.append_value(revenue_cents);
        match top_cat {
            Some(c) => self.top_category.append_value(c),
            None => self.top_category.append_null(),
        }
        self.pending += 1;
        if self.pending >= PARQUET_BATCH_ROWS {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.pending == 0 {
            return;
        }
        let writer = self.writer.get_or_insert_with(|| {
            let f = File::create(&self.path).expect("create user_daily parquet");
            ArrowWriter::try_new(
                BufWriter::with_capacity(1 << 20, f),
                self.schema.clone(),
                Some(fast_writer_props()),
            )
            .expect("ArrowWriter init")
        });

        let batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(self.user_id.finish()) as ArrayRef,
                Arc::new(self.day.finish()) as ArrayRef,
                Arc::new(self.sessions.finish()) as ArrayRef,
                Arc::new(self.events.finish()) as ArrayRef,
                Arc::new(self.revenue_cents.finish()) as ArrayRef,
                Arc::new(self.top_category.finish()) as ArrayRef,
            ],
        )
        .expect("RecordBatch build");
        writer.write(&batch).expect("parquet write");
        self.total_rows += self.pending;
        self.pending = 0;
    }

    fn finish(mut self) -> usize {
        self.flush();
        if let Some(w) = self.writer {
            w.close().expect("parquet close");
        }
        self.total_rows
    }
}

fn write_empty_user_daily(path: &Path, schema: &Arc<Schema>) {
    let f = File::create(path).expect("create empty user_daily");
    let w = ArrowWriter::try_new(
        BufWriter::new(f),
        schema.clone(),
        Some(fast_writer_props()),
    )
    .expect("empty ArrowWriter");
    w.close().expect("close empty user_daily");
}

fn write_campaign_daily(path: &Path, acc: &CampDailyAcc, campaigns: &Campaigns) {
    let schema = campaign_daily_schema();
    let f = File::create(path).expect("create campaign_daily parquet");
    let mut writer = ArrowWriter::try_new(
        BufWriter::with_capacity(1 << 20, f),
        schema.clone(),
        Some(fast_writer_props()),
    )
    .expect("campaign ArrowWriter");

    let mut cid_b = StringBuilder::new();
    let mut ch_b = StringBuilder::new();
    let mut day_b = Date32Builder::new();
    let mut sess_b = Int64Builder::new();
    let mut psess_b = Int64Builder::new();
    let mut rev_b = Int64Builder::new();
    let mut assist_b = Int64Builder::new();
    let mut rows = 0usize;

    if acc.num_days > 0 {
        for c in 0..acc.num_camps {
            let base = c * acc.num_days;
            let cid = &campaigns.ids[c];
            let ch = &campaigns.channels[c];
            for d in 0..acc.num_days {
                let s = &acc.dense[base + d];
                if s.is_non_zero() {
                    cid_b.append_value(cid);
                    ch_b.append_value(ch);
                    day_b.append_value(acc.min_day + d as i32);
                    sess_b.append_value(s.sessions);
                    psess_b.append_value(s.purchasing_sessions);
                    rev_b.append_value(s.revenue_cents);
                    assist_b.append_value(s.assisted_sessions);
                    rows += 1;
                }
            }
        }
    }
    for (&(c, day), s) in &acc.sparse {
        if s.is_non_zero() {
            cid_b.append_value(&campaigns.ids[c as usize]);
            ch_b.append_value(&campaigns.channels[c as usize]);
            day_b.append_value(day);
            sess_b.append_value(s.sessions);
            psess_b.append_value(s.purchasing_sessions);
            rev_b.append_value(s.revenue_cents);
            assist_b.append_value(s.assisted_sessions);
            rows += 1;
        }
    }

    if rows > 0 {
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(cid_b.finish()) as ArrayRef,
                Arc::new(ch_b.finish()) as ArrayRef,
                Arc::new(day_b.finish()) as ArrayRef,
                Arc::new(sess_b.finish()) as ArrayRef,
                Arc::new(psess_b.finish()) as ArrayRef,
                Arc::new(rev_b.finish()) as ArrayRef,
                Arc::new(assist_b.finish()) as ArrayRef,
            ],
        )
        .expect("campaign batch");
        writer.write(&batch).expect("campaign write");
    }
    writer.close().expect("campaign close");
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn list_csv_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = fs::read_dir(dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("csv") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

fn clean_output_dir(dir: &Path) {
    fs::create_dir_all(dir).expect("create output dir");
    if let Ok(rd) = fs::read_dir(dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("parquet") {
                let _ = fs::remove_file(p);
            }
        }
    }
}

#[inline(always)]
fn sort_events_by_ts(evs: &mut [PerPersonEvent]) {
    let n = evs.len();
    if n <= 1 {
        return;
    }
    if n <= 24 {
        for i in 1..n {
            let key = evs[i];
            let mut j = i;
            while j > 0 && evs[j - 1].ts > key.ts {
                evs[j] = evs[j - 1];
                j -= 1;
            }
            evs[j] = key;
        }
    } else {
        evs.sort_unstable_by_key(|e| e.ts);
    }
}

// ---------------------------------------------------------------------------
// Main Pipeline
// ---------------------------------------------------------------------------

fn main() {
    let data_dir = PathBuf::from(env::var("DATA_DIR").unwrap_or_else(|_| "/data".into()));
    let output_dir = PathBuf::from(env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into()));
    let ncpu: usize = env::var("NCPU")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8)
        })
        .min(64);

    let user_out_dir = output_dir.join("user_daily");
    let camp_out_dir = output_dir.join("campaign_daily");
    clean_output_dir(&user_out_dir);
    clean_output_dir(&camp_out_dir);

    // Stage 1: Manifest, products, campaigns, linked IDs
    let manifest_bytes = fs::read(data_dir.join("manifest.json")).expect("read manifest.json");
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes).expect("parse manifest.json");
    let window_start = parse_ts(manifest.window_start.as_bytes()).expect("bad window_start");
    let window_end = parse_ts(manifest.window_end.as_bytes()).expect("bad window_end");
    let click_window_start = window_start - MS_7D;
    let min_day = window_start.div_euclid(MS_PER_DAY) as i32;
    let max_day = (window_end - 1).div_euclid(MS_PER_DAY) as i32;

    let (products, campaigns, linked_map) = std::thread::scope(|s| {
        let p_handle = s.spawn(|| load_products(&data_dir.join("products.csv")));
        let c_handle = s.spawn(|| load_campaigns(&data_dir.join("campaigns.csv")));
        let l_map = load_links(&data_dir.join("links"), ncpu);
        (p_handle.join().unwrap(), c_handle.join().unwrap(), l_map)
    });

    // Stage 2: Mmap & parallel parse events and clicks
    let event_files = list_csv_files(&data_dir.join("events"));
    let click_files = list_csv_files(&data_dir.join("clicks"));

    let mut store = MmapStore::new();
    let mut ev_work = Vec::new();
    for f in &event_files {
        ev_work.extend(store.add_csv_file(f));
    }
    let mut clk_work = Vec::new();
    for f in &click_files {
        clk_work.extend(store.add_csv_file(f));
    }

    let mut thread_raw_ev: Vec<Vec<Vec<RawEvent>>> = (0..ncpu)
        .map(|_| (0..EV_BUCKETS).map(|_| Vec::new()).collect())
        .collect();
    let mut thread_clean_clk: Vec<Vec<Vec<CleanClick>>> = (0..ncpu)
        .map(|_| (0..PERSON_BUCKETS).map(|_| Vec::new()).collect())
        .collect();

    let ev_w_idx = AtomicUsize::new(0);
    let clk_w_idx = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for (ev_out, clk_out) in thread_raw_ev
            .iter_mut()
            .zip(thread_clean_clk.iter_mut())
        {
            let store_ref = &store;
            let ev_work_ref = &ev_work;
            let clk_work_ref = &clk_work;
            let ev_idx_ref = &ev_w_idx;
            let clk_idx_ref = &clk_w_idx;
            let prod_ref = &products;
            let camp_ref = &campaigns;
            let link_ref = &linked_map;

            s.spawn(move || {
                loop {
                    let wi = ev_idx_ref.fetch_add(1, Ordering::Relaxed);
                    if wi >= ev_work_ref.len() {
                        break;
                    }
                    let (chunk_id, start, end) = ev_work_ref[wi];
                    let chunk = store_ref.chunk_bytes(chunk_id);
                    let base_ptr = chunk.as_ptr() as usize;
                    let mut pos = start;
                    while pos < end {
                        let nl = memchr(b'\n', &chunk[pos..end])
                            .map(|k| pos + k)
                            .unwrap_or(end);
                        let line = &chunk[pos..nl];
                        pos = nl + 1;

                        let Some([event_id, user_id, ts_b, ingest_b, ev_type, prod_id, price_b]) =
                            split_7(line)
                        else {
                            continue;
                        };
                        if event_id.is_empty() {
                            continue;
                        }
                        let Some(ingest_ts) = parse_ts_f1(ingest_b) else {
                            continue;
                        };

                        let ev_hash = hash_bytes(event_id);
                        let bucket = (ev_hash >> 58) as usize;
                        let ev_id_ref = make_slice_ref(
                            chunk_id,
                            (event_id.as_ptr() as usize) - base_ptr,
                            event_id.len(),
                        );

                        let mut ts_val = INVALID_TS;
                        let mut user_ref = 0u64;
                        let mut price_cents = 0u32;
                        let mut cat_id = 0u16;

                        if !user_id.is_empty() && !prod_id.is_empty() {
                            if let Some(ts) = parse_ts(ts_b) {
                                if ts >= window_start
                                    && ts < window_end
                                    && ingest_ts <= ts.saturating_add(MS_24H)
                                {
                                    let valid_type = match ev_type {
                                        b"view" | b"cart" => true,
                                        b"purchase" => {
                                            if let Some(p) = parse_price_cents(price_b) {
                                                price_cents = p;
                                                cat_id = prod_ref
                                                    .map
                                                    .get(prod_id)
                                                    .copied()
                                                    .unwrap_or(prod_ref.unknown_cat_id);
                                                true
                                            } else {
                                                false
                                            }
                                        }
                                        _ => false,
                                    };
                                    if valid_type {
                                        ts_val = ts;
                                        user_ref = match link_ref.lookup(user_id) {
                                            Some(pid) => make_linked_person(pid),
                                            None => make_slice_ref(
                                                chunk_id,
                                                (user_id.as_ptr() as usize) - base_ptr,
                                                user_id.len(),
                                            ),
                                        };
                                    }
                                }
                            }
                        }

                        unsafe {
                            ev_out.get_unchecked_mut(bucket).push(RawEvent {
                                ev_hash,
                                ev_id: ev_id_ref,
                                ingest_ts,
                                ts: ts_val,
                                user_ref,
                                price_cents,
                                cat_id,
                            });
                        }
                    }
                }

                loop {
                    let wi = clk_idx_ref.fetch_add(1, Ordering::Relaxed);
                    if wi >= clk_work_ref.len() {
                        break;
                    }
                    let (chunk_id, start, end) = clk_work_ref[wi];
                    let chunk = store_ref.chunk_bytes(chunk_id);
                    let base_ptr = chunk.as_ptr() as usize;
                    let mut pos = start;
                    while pos < end {
                        let nl = memchr(b'\n', &chunk[pos..end])
                            .map(|k| pos + k)
                            .unwrap_or(end);
                        let line = &chunk[pos..nl];
                        pos = nl + 1;

                        let Some([click_id, user_id, camp_id_b, ts_b]) = split_4(line) else {
                            continue;
                        };
                        if click_id.is_empty() || user_id.is_empty() {
                            continue;
                        }
                        let Some(ts) = parse_ts(ts_b) else {
                            continue;
                        };
                        if ts < click_window_start || ts >= window_end {
                            continue;
                        }
                        let Some(&camp_id) = camp_ref.map.get(camp_id_b) else {
                            continue;
                        };

                        let (user_ref, p_hash) = match link_ref.lookup(user_id) {
                            Some(pid) => (make_linked_person(pid), hash_linked_id(pid)),
                            None => (
                                make_slice_ref(
                                    chunk_id,
                                    (user_id.as_ptr() as usize) - base_ptr,
                                    user_id.len(),
                                ),
                                hash_bytes(user_id),
                            ),
                        };
                        let click_id_ref = make_slice_ref(
                            chunk_id,
                            (click_id.as_ptr() as usize) - base_ptr,
                            click_id.len(),
                        );
                        let pb = (p_hash >> 57) as usize;
                        unsafe {
                            clk_out.get_unchecked_mut(pb).push(CleanClick {
                                ts,
                                user_ref,
                                click_id: click_id_ref,
                                camp_id,
                            });
                        }
                    }
                }
            });
        }
    });

    let mut ev_buckets_by_b: Vec<Vec<Vec<RawEvent>>> = (0..EV_BUCKETS)
        .map(|_| Vec::with_capacity(ncpu))
        .collect();
    for mut t_vecs in thread_raw_ev {
        for (b, v) in t_vecs.drain(..).enumerate() {
            ev_buckets_by_b[b].push(v);
        }
    }

    // Stage 3: Parallel Event Dedup (64 buckets) -> 128 Person Partitions
    let mut thread_clean_ev: Vec<Vec<Vec<CleanEvent>>> = (0..ncpu)
        .map(|_| (0..PERSON_BUCKETS).map(|_| Vec::new()).collect())
        .collect();

    #[derive(Copy, Clone)]
    struct DedupSlot {
        tag: u32,
        packed_ref: u32,
    }

    let b_atomic = AtomicUsize::new(0);
    let mut ev_buckets_opt: Vec<Mutex<Vec<Vec<RawEvent>>>> =
        ev_buckets_by_b.into_iter().map(Mutex::new).collect();

    std::thread::scope(|s| {
        for clean_out in &mut thread_clean_ev {
            let store_ref = &store;
            let b_idx_ref = &b_atomic;
            let buckets_ref = &ev_buckets_opt;

            s.spawn(move || {
                let mut table: Vec<DedupSlot> = Vec::new();
                let mut occupied: Vec<u32> = Vec::new();

                loop {
                    let b = b_idx_ref.fetch_add(1, Ordering::Relaxed);
                    if b >= EV_BUCKETS {
                        break;
                    }
                    let sub_vecs = std::mem::take(&mut *buckets_ref[b].lock().unwrap());
                    let total: usize = sub_vecs.iter().map(|v| v.len()).sum();
                    if total == 0 {
                        continue;
                    }

                    let cap = (total * 2).next_power_of_two().max(1024);
                    if table.len() < cap {
                        table.resize(
                            cap,
                            DedupSlot {
                                tag: 0,
                                packed_ref: 0,
                            },
                        );
                    }
                    let mask = cap - 1;
                    occupied.clear();

                    for (ti, vec) in sub_vecs.iter().enumerate() {
                        let t_bits = (ti as u32) << 26;
                        for (idx, ev) in vec.iter().enumerate() {
                            let tag = ((ev.ev_hash >> 26) as u32) | 1;
                            let packed_ref = t_bits | (idx as u32);
                            let mut pos = (ev.ev_hash as usize) & mask;
                            loop {
                                let slot = unsafe { *table.get_unchecked(pos) };
                                if slot.tag == 0 {
                                    unsafe {
                                        *table.get_unchecked_mut(pos) =
                                            DedupSlot { tag, packed_ref };
                                    }
                                    occupied.push(pos as u32);
                                    break;
                                }
                                if slot.tag == tag {
                                    let old_t = (slot.packed_ref >> 26) as usize;
                                    let old_i = (slot.packed_ref & 0x03FF_FFFF) as usize;
                                    let old_ev = unsafe {
                                        sub_vecs.get_unchecked(old_t).get_unchecked(old_i)
                                    };
                                    if store_ref.get_slice(old_ev.ev_id)
                                        == store_ref.get_slice(ev.ev_id)
                                    {
                                        if ev.ingest_ts < old_ev.ingest_ts {
                                            unsafe {
                                                table.get_unchecked_mut(pos).packed_ref =
                                                    packed_ref;
                                            }
                                        }
                                        break;
                                    }
                                }
                                pos = (pos + 1) & mask;
                            }
                        }
                    }

                    for &pos in &occupied {
                        let slot = unsafe { *table.get_unchecked(pos as usize) };
                        unsafe {
                            table.get_unchecked_mut(pos as usize).tag = 0;
                        }
                        let t = (slot.packed_ref >> 26) as usize;
                        let i = (slot.packed_ref & 0x03FF_FFFF) as usize;
                        let ev = unsafe { sub_vecs.get_unchecked(t).get_unchecked(i) };
                        if ev.ts == INVALID_TS {
                            continue;
                        }
                        let p_hash = if is_linked_person(ev.user_ref) {
                            hash_linked_id(linked_person_id(ev.user_ref))
                        } else {
                            hash_bytes(store_ref.get_slice(ev.user_ref))
                        };
                        let pb = (p_hash >> 57) as usize;
                        unsafe {
                            clean_out.get_unchecked_mut(pb).push(CleanEvent {
                                ts: ev.ts,
                                user_ref: ev.user_ref,
                                price_cents: ev.price_cents,
                                cat_id: ev.cat_id,
                            });
                        }
                    }
                }
            });
        }
    });
    ev_buckets_opt.clear();

    let mut part_evs: Vec<Mutex<Vec<Vec<CleanEvent>>>> = (0..PERSON_BUCKETS)
        .map(|_| Mutex::new(Vec::with_capacity(ncpu)))
        .collect();
    for mut t_vecs in thread_clean_ev {
        for (p, v) in t_vecs.drain(..).enumerate() {
            part_evs[p].get_mut().unwrap().push(v);
        }
    }

    let mut part_clks: Vec<Mutex<Vec<Vec<CleanClick>>>> = (0..PERSON_BUCKETS)
        .map(|_| Mutex::new(Vec::with_capacity(ncpu)))
        .collect();
    for mut t_vecs in thread_clean_clk {
        for (p, v) in t_vecs.drain(..).enumerate() {
            part_clks[p].get_mut().unwrap().push(v);
        }
    }

    let mut part_order: Vec<(usize, usize)> = (0..PERSON_BUCKETS)
        .map(|p| {
            let cnt: usize = part_evs[p]
                .get_mut()
                .unwrap()
                .iter()
                .map(|v| v.len())
                .sum();
            (p, cnt)
        })
        .collect();
    part_order.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    // Stage 4: Parallel Sessionize, Attribute, Assists & Write user_daily
    let ud_schema = user_daily_schema();
    let num_camps = campaigns.ids.len();
    let organic_id = campaigns.organic_id;
    let num_cats = products.cat_names.len();

    let p_atomic = AtomicUsize::new(0);
    let mut thread_camp_accs: Vec<CampDailyAcc> = (0..ncpu)
        .map(|_| CampDailyAcc::new(num_camps, min_day, max_day))
        .collect();
    let mut thread_ud_rows = vec![0usize; ncpu];

    std::thread::scope(|s| {
        for (tid, (camp_acc, ud_count)) in thread_camp_accs
            .iter_mut()
            .zip(thread_ud_rows.iter_mut())
            .enumerate()
        {
            let store_ref = &store;
            let link_ref = &linked_map;
            let prod_ref = &products;
            let part_evs_ref = &part_evs;
            let part_clks_ref = &part_clks;
            let order_ref = &part_order;
            let p_idx_ref = &p_atomic;
            let ud_path = user_out_dir.join(format!("part-{}.parquet", tid));
            let schema_clone = ud_schema.clone();

            s.spawn(move || {
                let mut ud_writer = UserDailyWriter::new(ud_path, schema_clone);
                let mut person_table = LocalPersonTable::new();

                let mut ev_pids: Vec<u32> = Vec::new();
                let mut ev_counts: Vec<u32> = Vec::new();
                let mut ev_offsets: Vec<u32> = Vec::new();
                let mut ev_cursor: Vec<u32> = Vec::new();
                let mut grouped_evs: Vec<PerPersonEvent> = Vec::new();

                let mut clk_matched: Vec<(u32, PerPersonClick)> = Vec::new();
                let mut clk_counts: Vec<u32> = Vec::new();
                let mut clk_offsets: Vec<u32> = Vec::new();
                let mut clk_cursor: Vec<u32> = Vec::new();
                let mut grouped_clks: Vec<PerPersonClick> = Vec::new();

                let mut camp_counts = vec![0u32; num_camps];
                let mut camp_start_idx = vec![0i64; num_camps];
                let mut camp_active_pos = vec![0u32; num_camps];
                let mut active_camps: Vec<u32> = Vec::with_capacity(64);

                let mut cat_revs = vec![0i64; num_cats];
                let mut touched_cats: Vec<u16> = Vec::with_capacity(16);

                loop {
                    let ord_i = p_idx_ref.fetch_add(1, Ordering::Relaxed);
                    if ord_i >= PERSON_BUCKETS {
                        break;
                    }
                    let (p, n_ev) = order_ref[ord_i];
                    let ev_vecs = std::mem::take(&mut *part_evs_ref[p].lock().unwrap());
                    let clk_vecs = std::mem::take(&mut *part_clks_ref[p].lock().unwrap());
                    if n_ev == 0 {
                        continue;
                    }

                    person_table.prepare(n_ev);
                    ev_pids.clear();
                    ev_pids.reserve(n_ev);
                    ev_counts.clear();

                    for v in &ev_vecs {
                        for ev in v {
                            let lp = person_table.get_or_insert(ev.user_ref, store_ref);
                            ev_pids.push(lp);
                            if (lp as usize) >= ev_counts.len() {
                                ev_counts.push(1);
                            } else {
                                ev_counts[lp as usize] += 1;
                            }
                        }
                    }

                    let num_persons = ev_counts.len();
                    ev_offsets.clear();
                    ev_offsets.reserve(num_persons + 1);
                    let mut sum = 0u32;
                    ev_offsets.push(0);
                    for &c in &ev_counts {
                        sum += c;
                        ev_offsets.push(sum);
                    }
                    ev_cursor.clear();
                    ev_cursor.extend_from_slice(&ev_offsets[..num_persons]);
                    grouped_evs.resize(
                        n_ev,
                        PerPersonEvent {
                            ts: 0,
                            price_cents: 0,
                            cat_id: 0,
                        },
                    );

                    let mut flat_i = 0usize;
                    for v in &ev_vecs {
                        for ev in v {
                            let lp = unsafe { *ev_pids.get_unchecked(flat_i) } as usize;
                            flat_i += 1;
                            let dst = unsafe {
                                let c = ev_cursor.get_unchecked_mut(lp);
                                let d = *c as usize;
                                *c += 1;
                                d
                            };
                            unsafe {
                                *grouped_evs.get_unchecked_mut(dst) = PerPersonEvent {
                                    ts: ev.ts,
                                    price_cents: ev.price_cents,
                                    cat_id: ev.cat_id,
                                };
                            }
                        }
                    }
                    drop(ev_vecs);

                    clk_matched.clear();
                    clk_counts.clear();
                    clk_counts.resize(num_persons, 0);
                    for v in &clk_vecs {
                        for ck in v {
                            if let Some(lp) = person_table.get(ck.user_ref, store_ref) {
                                clk_counts[lp as usize] += 1;
                                clk_matched.push((
                                    lp,
                                    PerPersonClick {
                                        ts: ck.ts,
                                        click_id: ck.click_id,
                                        camp_id: ck.camp_id,
                                    },
                                ));
                            }
                        }
                    }
                    drop(clk_vecs);

                    let n_clk = clk_matched.len();
                    clk_offsets.clear();
                    clk_offsets.reserve(num_persons + 1);
                    let mut csum = 0u32;
                    clk_offsets.push(0);
                    for &c in &clk_counts {
                        csum += c;
                        clk_offsets.push(csum);
                    }
                    clk_cursor.clear();
                    clk_cursor.extend_from_slice(&clk_offsets[..num_persons]);
                    grouped_clks.resize(
                        n_clk,
                        PerPersonClick {
                            ts: 0,
                            click_id: 0,
                            camp_id: 0,
                        },
                    );
                    for &(lp, ck) in &clk_matched {
                        let dst = unsafe {
                            let c = clk_cursor.get_unchecked_mut(lp as usize);
                            let d = *c as usize;
                            *c += 1;
                            d
                        };
                        unsafe {
                            *grouped_clks.get_unchecked_mut(dst) = ck;
                        }
                    }

                    for u in 0..num_persons {
                        let pkey = person_table.keys[u];
                        let uid_str = if is_linked_person(pkey) {
                            link_ref.canon_name(linked_person_id(pkey))
                        } else {
                            std::str::from_utf8(store_ref.get_slice(pkey))
                                .expect("utf8 unlinked user_id")
                        };

                        let e0 = ev_offsets[u] as usize;
                        let e1 = ev_offsets[u + 1] as usize;
                        let evs = &mut grouped_evs[e0..e1];
                        sort_events_by_ts(evs);

                        let c0 = clk_offsets[u] as usize;
                        let c1 = clk_offsets[u + 1] as usize;
                        let clks = &mut grouped_clks[c0..c1];
                        let has_clicks = !clks.is_empty();
                        if clks.len() > 1 {
                            clks.sort_unstable_by(|a, b| {
                                a.ts.cmp(&b.ts).then_with(|| {
                                    store_ref
                                        .get_slice(b.click_id)
                                        .cmp(store_ref.get_slice(a.click_id))
                                })
                            });
                        }

                        let mut l = 0usize;
                        let mut r = 0usize;
                        let mut cur_day = i32::MIN;
                        let mut day_sessions = 0i64;
                        let mut day_events = 0i64;
                        let mut day_rev = 0i64;

                        let mut i = 0usize;
                        while i < evs.len() {
                            let sess_start = evs[i].ts;
                            let sess_day = sess_start.div_euclid(MS_PER_DAY) as i32;

                            if sess_day != cur_day {
                                if cur_day != i32::MIN {
                                    let top_cat = if touched_cats.is_empty() {
                                        None
                                    } else {
                                        let mut best_rev = -1i64;
                                        let mut best_cat = u16::MAX;
                                        for &c in &touched_cats {
                                            let rv = cat_revs[c as usize];
                                            cat_revs[c as usize] = 0;
                                            if rv > best_rev || (rv == best_rev && c < best_cat) {
                                                best_rev = rv;
                                                best_cat = c;
                                            }
                                        }
                                        touched_cats.clear();
                                        Some(prod_ref.cat_names[best_cat as usize].as_str())
                                    };
                                    ud_writer.push(
                                        uid_str,
                                        cur_day,
                                        day_sessions,
                                        day_events,
                                        day_rev,
                                        top_cat,
                                    );

                                    for &ac in &active_camps {
                                        let diff = day_sessions - camp_start_idx[ac as usize];
                                        if diff > 0 {
                                            camp_acc.add_assist(ac, cur_day, diff);
                                        }
                                        camp_start_idx[ac as usize] = 0;
                                    }
                                }
                                cur_day = sess_day;
                                day_sessions = 0;
                                day_events = 0;
                                day_rev = 0;
                            }

                            let i0 = i;
                            let mut prev_ts = sess_start;
                            let mut sess_rev = 0i64;
                            let mut sess_has_purchase = false;
                            while i < evs.len() {
                                let e = unsafe { *evs.get_unchecked(i) };
                                if i > i0 && e.ts > prev_ts.saturating_add(MS_30M) {
                                    break;
                                }
                                prev_ts = e.ts;
                                if e.price_cents > 0 {
                                    let p = e.price_cents as i64;
                                    sess_rev += p;
                                    sess_has_purchase = true;
                                    let c = e.cat_id as usize;
                                    if cat_revs[c] == 0 {
                                        touched_cats.push(e.cat_id);
                                    }
                                    cat_revs[c] += p;
                                }
                                i += 1;
                            }
                            let sess_ev_cnt = (i - i0) as i64;

                            if !has_clicks {
                                camp_acc.credit(organic_id, sess_day, sess_rev, sess_has_purchase);
                            } else {
                                while r < clks.len() && clks[r].ts <= sess_start {
                                    let c = clks[r].camp_id as usize;
                                    if camp_counts[c] == 0 {
                                        camp_start_idx[c] = day_sessions;
                                        camp_active_pos[c] = active_camps.len() as u32;
                                        active_camps.push(c as u32);
                                    }
                                    camp_counts[c] += 1;
                                    r += 1;
                                }
                                let min_click_ts = sess_start.saturating_sub(MS_7D);
                                while l < r && clks[l].ts < min_click_ts {
                                    let c = clks[l].camp_id as usize;
                                    camp_counts[c] -= 1;
                                    if camp_counts[c] == 0 {
                                        let diff = day_sessions - camp_start_idx[c];
                                        if diff > 0 {
                                            camp_acc.add_assist(c as u32, sess_day, diff);
                                        }
                                        let pos = camp_active_pos[c] as usize;
                                        let last = active_camps.pop().unwrap();
                                        if pos < active_camps.len() {
                                            active_camps[pos] = last;
                                            camp_active_pos[last as usize] = pos as u32;
                                        }
                                    }
                                    l += 1;
                                }
                                if l == r {
                                    camp_acc.credit(
                                        organic_id,
                                        sess_day,
                                        sess_rev,
                                        sess_has_purchase,
                                    );
                                } else {
                                    camp_acc.credit(
                                        clks[r - 1].camp_id,
                                        sess_day,
                                        sess_rev,
                                        sess_has_purchase,
                                    );
                                }
                            }

                            day_sessions += 1;
                            day_events += sess_ev_cnt;
                            day_rev += sess_rev;
                        }

                        if cur_day != i32::MIN {
                            let top_cat = if touched_cats.is_empty() {
                                None
                            } else {
                                let mut best_rev = -1i64;
                                let mut best_cat = u16::MAX;
                                for &c in &touched_cats {
                                    let rv = cat_revs[c as usize];
                                    cat_revs[c as usize] = 0;
                                    if rv > best_rev || (rv == best_rev && c < best_cat) {
                                        best_rev = rv;
                                        best_cat = c;
                                    }
                                }
                                touched_cats.clear();
                                Some(prod_ref.cat_names[best_cat as usize].as_str())
                            };
                            ud_writer.push(
                                uid_str,
                                cur_day,
                                day_sessions,
                                day_events,
                                day_rev,
                                top_cat,
                            );

                            for &ac in &active_camps {
                                let diff = day_sessions - camp_start_idx[ac as usize];
                                if diff > 0 {
                                    camp_acc.add_assist(ac, cur_day, diff);
                                }
                                camp_counts[ac as usize] = 0;
                            }
                            active_camps.clear();
                        }
                    }

                    person_table.reset();
                }

                *ud_count = ud_writer.finish();
            });
        }
    });

    if thread_ud_rows.iter().sum::<usize>() == 0 {
        write_empty_user_daily(&user_out_dir.join("part-0.parquet"), &ud_schema);
    }

    let mut final_camp = thread_camp_accs.remove(0);
    for acc in &thread_camp_accs {
        final_camp.merge_from(acc);
    }
    write_campaign_daily(
        &camp_out_dir.join("part-0.parquet"),
        &final_camp,
        &campaigns,
    );
}