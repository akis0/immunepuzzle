use anyhow::Result;
use rand::{Rng, SeedableRng, TryRngCore, rngs::OsRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::{HashMap, HashSet};

/// Axial coordinates for hex grid: (q, r)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Axial {
    q: i32,
    r: i32,
}

impl Axial {
    fn new(q: i32, r: i32) -> Self {
        Self { q, r }
    }
    fn add(self, d: (i32, i32)) -> Self {
        Self::new(self.q + d.0, self.r + d.1)
    }
}

/// 6 neighbor directions in axial coords (pointy-top convention)
/// Edge index i corresponds to DIRS[i]
const DIRS: [(i32, i32); 6] = [
    (1, 0),  // 0
    (1, -1), // 1
    (0, -1), // 2
    (-1, 0), // 3
    (-1, 1), // 4
    (0, 1),  // 5
];

fn opposite_dir(i: usize) -> usize {
    (i + 3) % 6
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum PieceKind {
    True,
    False,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Piece {
    id: usize,
    kind: PieceKind,
    /// For false pieces, which cluster they belong to (optional).
    cluster_id: Option<usize>,
    /// 6 edges, index 0..5 match DIRS order.
    edges: [i32; 6],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Puzzle {
    side_len: i32,
    edge_alphabet: i32,
    true_piece_count: usize,
    false_piece_count: usize,
    pieces: Vec<Piece>,
    /// Some basic stats to sanity-check "hardness knobs"
    stats: Stats,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Stats {
    /// how many times each absolute edge id appears in true set (excluding 0)
    true_edge_multiplicity_hist: Vec<(i32, usize)>,
    /// number of distinct canonical (rotation-normalized) shapes in true set
    true_unique_shapes: usize,
    /// number of distinct canonical shapes in all pieces
    all_unique_shapes: usize,
}

#[derive(Debug, Clone)]
struct Config {
    side_len: i32,      // e.g. 8
    edge_alphabet: i32, // small => harder (more collisions)
    seed: u64,

    /// If true, regenerate true set until no two true pieces share same canonical shape (rotation-normalized).
    enforce_unique_true_shapes: bool,
    max_true_regen: usize,

    /// NEW: anchor edges ratio (0.0..1.0)
    /// Some internal edges get globally-unique ids to suppress accidental symmetries / duplicate shapes.
    anchor_ratio: f64,

    /// False cluster spec: list of (cluster_size, count)
    false_clusters: Vec<(usize, usize)>,
    /// For each boundary edge of a false cluster (an edge not connected to another false piece in same cluster):
    /// - with prob flat_ratio => set to 0 (looks like boundary)
    /// - else with prob attach_ratio => reuse some existing true edge id (to create "warped match")
    /// - else => unmatched unique id (no counterpart)
    false_flat_ratio: f64,
    false_attach_ratio: f64,
}

fn random_seed() -> u64 {
    let mut rng = OsRng;
    rng.try_next_u64().expect("os rng for seed")
}

fn parse_opt_u64(flag: &str) -> Option<u64> {
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == flag {
            if let Some(v) = it.next() {
                if let Ok(x) = v.parse::<u64>() {
                    return Some(x);
                }
            }
        }
    }
    None
}

fn parse_opt_usize(flag: &str) -> Option<usize> {
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == flag {
            if let Some(v) = it.next() {
                if let Ok(x) = v.parse::<usize>() {
                    return Some(x);
                }
            }
        }
    }
    None
}

fn parse_opt_i32(flag: &str) -> Option<i32> {
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == flag {
            if let Some(v) = it.next() {
                if let Ok(x) = v.parse::<i32>() {
                    return Some(x);
                }
            }
        }
    }
    None
}

fn parse_opt_f64(flag: &str) -> Option<f64> {
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == flag {
            if let Some(v) = it.next() {
                if let Ok(x) = v.parse::<f64>() {
                    return Some(x);
                }
            }
        }
    }
    None
}

fn main() -> Result<()> {
    // ---- CLIで調整できるようにしておく（無ければデフォルト） ----
    let trials: usize = parse_opt_usize("--trials").unwrap_or(10000);
    let fixed_seed: Option<u64> = parse_opt_u64("--seed"); // 指定があれば最初のseedだけ固定
    let side_len: i32 = parse_opt_i32("--side").unwrap_or(8);
    let edge_alphabet: i32 = parse_opt_i32("--alphabet").unwrap_or(20);
    let anchor_ratio: f64 = parse_opt_f64("--anchor-ratio").unwrap_or(0.10);

    // 偽の境界挙動
    let false_flat_ratio: f64 = parse_opt_f64("--false-flat").unwrap_or(0.45);
    let false_attach_ratio: f64 = parse_opt_f64("--false-attach").unwrap_or(0.80);

    // 偽塊スペック（例：強め。必要ならここもCLI化してOK）
    // 合計:60
    let false_clusters: Vec<(usize, usize)> = vec![(1, 10), (2, 10), (4, 4), (7, 2)];

    // 真ピース生成時の「回転同値で形状が被らない」方向の選抜（保険）
    let enforce_unique_true_shapes = true;
    let max_true_regen = 500;

    // ---- ここから選抜ループ（免疫選抜） ----
    // best = (puzzle, seed, hardness_score, uniq_nodes, false_nodes)
    let mut best: Option<(Puzzle, u64, u64, u64, u64)> = None;

    for t in 0..trials {
        // seed：--seedが指定されていれば最初だけそれを使い、以降はランダム
        let seed = match fixed_seed {
            Some(s) if t == 0 => s,
            _ => random_seed(),
        };

        let cfg = Config {
            side_len,
            edge_alphabet,
            seed,
            enforce_unique_true_shapes,
            max_true_regen,
            anchor_ratio,
            false_clusters: false_clusters.clone(),
            false_flat_ratio,
            false_attach_ratio,
        };

        let puzzle = generate_puzzle(&cfg)?;

        // (A) 真ピースだけで「回転同値を除いて一意」
        let true_n = puzzle.true_piece_count;
        let rep_u = check_unique_true(puzzle.side_len, &puzzle.pieces[..true_n]);
        if !rep_u.unique_mod_rotation {
            let extra = if rep_u.nonunique_only_by_reflection {
                " [mirror-pair only]"
            } else {
                ""
            };
            eprintln!(
                "[{}/{}] seed={} REJECT: not unique{} (solutions_found={}, nodes={})",
                t + 1,
                trials,
                seed,
                extra,
                rep_u.solutions_found,
                rep_u.nodes
            );
            continue;
        }

        // (B) 偽ピースを1枚でも使う敷き詰め解が存在しない
        let rep_f = exists_solution_using_false(puzzle.side_len, &puzzle.pieces);
        if rep_f.exists {
            eprintln!(
                "[{}/{}] seed={} REJECT: false-containing tiling exists (nodes={})",
                t + 1,
                trials,
                seed,
                rep_f.nodes
            );
            continue;
        }

        // ---- hardness（硬さ）スコア ----
        // ここは趣味でいじれる。例：
        // - 偽混入存在判定の探索ノード数を主採用（大きいほど“偽が刺さってそう”）
        // - 真の一意性判定の探索ノード数も加点
        let hardness_score = rep_f.nodes.saturating_add(rep_u.nodes / 4);

        eprintln!(
            "[{}/{}] seed={} ACCEPT: unique OK (u_nodes={}), no-false-tiling (f_nodes={}), score={}",
            t + 1,
            trials,
            seed,
            rep_u.nodes,
            rep_f.nodes,
            hardness_score
        );

        match &best {
            None => best = Some((puzzle, seed, hardness_score, rep_u.nodes, rep_f.nodes)),
            Some((_, _, best_score, _, _)) if hardness_score > *best_score => {
                best = Some((puzzle, seed, hardness_score, rep_u.nodes, rep_f.nodes))
            }
            _ => {}
        }
    }

    // ---- 結果出力 ----
    if let Some((p, seed, score, u_nodes, f_nodes)) = best {
        eprintln!(
            "SELECTED seed={} score={} (u_nodes={}, f_nodes={})",
            seed, score, u_nodes, f_nodes
        );
        println!("{}", serde_json::to_string_pretty(&p)?);
    } else {
        eprintln!(
            "No acceptable instance found in {} trials. \
Try increasing --trials, or relax constraints (e.g., raise --alphabet, lower --false-attach).",
            trials
        );
    }

    Ok(())
}

/// Build all coords inside a hex of side_len s.
/// Using radius r = s-1, include all axial (q,r) with:
/// |q| <= r, |r| <= r, |q+r| <= r
fn hex_board_coords(side_len: i32) -> Vec<Axial> {
    let radius = side_len - 1;
    let mut coords = Vec::new();
    for q in -radius..=radius {
        for r in -radius..=radius {
            let s = -q - r;
            if q.abs() <= radius && r.abs() <= radius && s.abs() <= radius {
                coords.push(Axial::new(q, r));
            }
        }
    }
    coords
}

/// Rotation of edge array by k * 60deg (cyclic shift)
fn rotate_edges(edges: &[i32; 6], k: usize) -> [i32; 6] {
    let mut out = [0i32; 6];
    for i in 0..6 {
        out[(i + k) % 6] = edges[i];
    }
    out
}

/// Canonicalize a piece shape up to rotation: choose lexicographically smallest rotation.
fn canonical_shape(edges: &[i32; 6]) -> [i32; 6] {
    let mut best = *edges;
    for k in 1..6 {
        let rot = rotate_edges(edges, k);
        if rot < best {
            best = rot;
        }
    }
    best
}

/// Generate "true" puzzle pieces for a hex board by assigning random edge ids to each adjacency.
/// Signs are consistent: neighbors have opposite values. Boundary edges are 0.
fn generate_true_pieces(
    rng: &mut StdRng,
    side_len: i32,
    edge_alphabet: i32,
    anchor_ratio: f64,
    enforce_unique_shapes: bool,
    max_regen: usize,
) -> Result<(Vec<Piece>, Vec<i32>, Stats)> {
    let coords = hex_board_coords(side_len);
    let coord_set: HashSet<Axial> = coords.iter().copied().collect();

    // Precompute all undirected adjacencies (a < b)
    // Store edge id per undirected pair.
    // Key: (min, max)
    let mut attempt = 0usize;
    let anchor_start = edge_alphabet + 1_000;

    loop {
        attempt += 1;
        let mut edge_id_map: HashMap<(Axial, Axial), i32> = HashMap::new();
        let mut next_anchor_id: i32 = anchor_start;

        // Assign random id in 1..=edge_alphabet to each adjacency
        for &a in &coords {
            for (dir_idx, d) in DIRS.iter().enumerate() {
                let b = a.add(*d);
                if !coord_set.contains(&b) {
                    continue;
                }
                let (minc, maxc) = if a < b { (a, b) } else { (b, a) };
                edge_id_map.entry((minc, maxc)).or_insert_with(|| {
                    let roll: f64 = rng.random();
                    if roll < anchor_ratio {
                        let v = next_anchor_id;
                        next_anchor_id += 1;
                        v
                    } else {
                        rng.random_range(1..=edge_alphabet)
                    }
                });
                // dir_idx used later when building per-tile edges
                let _ = dir_idx;
            }
        }

        // Build pieces
        let mut pieces: Vec<Piece> = Vec::with_capacity(coords.len());
        let mut all_internal_edge_ids_abs: Vec<i32> = Vec::new();

        for (idx, &c) in coords.iter().enumerate() {
            let mut edges = [0i32; 6];
            for (dir_idx, d) in DIRS.iter().enumerate() {
                let n = c.add(*d);
                if !coord_set.contains(&n) {
                    edges[dir_idx] = 0;
                    continue;
                }
                let (minc, maxc) = if c < n { (c, n) } else { (n, c) };
                let id = *edge_id_map.get(&(minc, maxc)).expect("assigned");
                // sign convention: min gets +id, max gets -id
                edges[dir_idx] = if c == minc { id } else { -id };
                all_internal_edge_ids_abs.push(id.abs());
            }

            pieces.push(Piece {
                id: idx,
                kind: PieceKind::True,
                cluster_id: None,
                edges,
            });
        }

        // Optional uniqueness filter on canonical shapes (up to rotation).
        let mut true_shape_set: HashSet<[i32; 6]> = HashSet::new();
        for p in &pieces {
            true_shape_set.insert(canonical_shape(&p.edges));
        }

        if enforce_unique_shapes && true_shape_set.len() != pieces.len() {
            if attempt >= max_regen {
                // Give up: return the best we have (still valid, just not uniqueness-safe)
                let stats = compute_stats(&pieces, &[]);
                return Ok((pieces, all_internal_edge_ids_abs, stats));
            }
            continue;
        }

        // Compute stats (multiplicity histogram etc.)
        let stats = compute_stats(&pieces, &[]);
        return Ok((pieces, all_internal_edge_ids_abs, stats));
    }
}

/// Generate false clusters:
/// - Each cluster is built on a small local hex layout (connected set of coords).
/// - Internal edges are assigned fresh unique ids (so the false island is internally "beautiful").
/// - Boundary edges: 0 (flat), reused true ids (attachable), or unmatched fresh ids.
fn generate_false_pieces(
    rng: &mut StdRng,
    edge_alphabet: i32,
    true_edge_ids_abs: &[i32],
    cluster_spec: &[(usize, usize)],
    flat_ratio: f64,
    attach_ratio: f64,
    start_piece_id: usize,
) -> Vec<Piece> {
    // Fresh ids for unmatched/internal false edges, avoid colliding with small alphabet.
    // Start above max(true alphabet) + 1_000_000 to be safe.
    let mut next_fresh_id: i32 = edge_alphabet + 1_000_000;

    let mut pieces: Vec<Piece> = Vec::new();
    let mut pid = start_piece_id;
    let mut cluster_id_counter: usize = 0;

    for &(size, count) in cluster_spec {
        for _ in 0..count {
            cluster_id_counter += 1;
            let coords = random_connected_cluster_coords(rng, size);
            let coord_to_idx: HashMap<Axial, usize> =
                coords.iter().enumerate().map(|(i, &c)| (c, i)).collect();

            // Create edges array for each node; fill 0 initially (we'll override)
            let mut edge_arrays: Vec<[i32; 6]> = vec![[0i32; 6]; size];

            // Assign internal edges (between false pieces) with fresh unique ids.
            // For each adjacency inside the cluster, set opposite signs.
            // We'll assign an undirected fresh id per adjacency.
            let mut internal_map: HashMap<(Axial, Axial), i32> = HashMap::new();

            for &a in &coords {
                for (dir_idx, d) in DIRS.iter().enumerate() {
                    let b = a.add(*d);
                    if !coord_to_idx.contains_key(&b) {
                        continue;
                    }
                    let (minc, maxc) = if a < b { (a, b) } else { (b, a) };
                    let id = *internal_map.entry((minc, maxc)).or_insert_with(|| {
                        let v = next_fresh_id;
                        next_fresh_id += 1;
                        v
                    });

                    // sign convention within cluster (same as true): min +id, max -id
                    let a_idx = coord_to_idx[&a];
                    let b_idx = coord_to_idx[&b];
                    edge_arrays[a_idx][dir_idx] = if a == minc { id } else { -id };
                    edge_arrays[b_idx][opposite_dir(dir_idx)] = if b == minc { id } else { -id };
                }
            }

            // Fill boundary edges for cluster nodes (edges still 0 after internal fill)
            for i in 0..size {
                for dir_idx in 0..6 {
                    if edge_arrays[i][dir_idx] != 0 {
                        continue; // internal edge already set
                    }

                    let roll: f64 = rng.random();
                    if roll < flat_ratio {
                        // Looks like boundary
                        edge_arrays[i][dir_idx] = 0;
                    } else if roll < flat_ratio + attach_ratio && !true_edge_ids_abs.is_empty() {
                        // Reuse an existing true edge id to create "warped match"
                        let v =
                            true_edge_ids_abs[rng.random_range(0..true_edge_ids_abs.len())].abs();
                        // Randomize sign; it will match somewhere with opposite sign in true set.
                        edge_arrays[i][dir_idx] = if rng.random_bool(0.5) { v } else { -v };
                    } else {
                        // Unmatched fresh id: appears only once => no counterpart => "適合ピースなし"
                        let v = next_fresh_id;
                        next_fresh_id += 1;
                        edge_arrays[i][dir_idx] = if rng.random_bool(0.5) { v } else { -v };
                    }
                }
            }

            // Add pieces
            for i in 0..size {
                pieces.push(Piece {
                    id: pid,
                    kind: PieceKind::False,
                    cluster_id: Some(cluster_id_counter),
                    edges: edge_arrays[i],
                });
                pid += 1;
            }
        }
    }

    pieces
}

/// Create a random connected set of `size` axial coords using growth on hex grid.
/// Starts at (0,0) and expands by attaching neighbors.
fn random_connected_cluster_coords(rng: &mut StdRng, size: usize) -> Vec<Axial> {
    let mut set: HashSet<Axial> = HashSet::new();
    let mut vec: Vec<Axial> = Vec::new();

    let start = Axial::new(0, 0);
    set.insert(start);
    vec.push(start);

    while vec.len() < size {
        // pick an existing node and try to add a neighbor not in set
        let base = vec[rng.random_range(0..vec.len())];
        let dir = rng.random_range(0..6);
        let cand = base.add(DIRS[dir]);
        if set.insert(cand) {
            vec.push(cand);
        }
    }
    vec
}

fn compute_stats(true_pieces: &[Piece], all_pieces: &[Piece]) -> Stats {
    // Multiplicity of abs edge ids in true set (exclude 0)
    let mut freq: HashMap<i32, usize> = HashMap::new();
    for p in true_pieces {
        for &e in &p.edges {
            if e == 0 {
                continue;
            }
            *freq.entry(e.abs()).or_insert(0) += 1;
        }
    }
    let mut hist: Vec<(i32, usize)> = freq.into_iter().collect();
    hist.sort_by_key(|(id, _)| *id);

    let mut true_shapes: HashSet<[i32; 6]> = HashSet::new();
    for p in true_pieces {
        true_shapes.insert(canonical_shape(&p.edges));
    }

    let mut all_shapes: HashSet<[i32; 6]> = HashSet::new();
    for p in true_pieces.iter().chain(all_pieces.iter()) {
        all_shapes.insert(canonical_shape(&p.edges));
    }

    Stats {
        true_edge_multiplicity_hist: hist,
        true_unique_shapes: true_shapes.len(),
        all_unique_shapes: all_shapes.len(),
    }
}

fn generate_puzzle(cfg: &Config) -> Result<Puzzle> {
    let mut rng = StdRng::seed_from_u64(cfg.seed);

    let (mut true_pieces, true_edge_ids_abs, _true_stats) = generate_true_pieces(
        &mut rng,
        cfg.side_len,
        cfg.edge_alphabet,
        cfg.anchor_ratio,
        cfg.enforce_unique_true_shapes,
        cfg.max_true_regen,
    )?;

    let false_pieces = generate_false_pieces(
        &mut rng,
        cfg.edge_alphabet,
        &true_edge_ids_abs,
        &cfg.false_clusters,
        cfg.false_flat_ratio,
        cfg.false_attach_ratio,
        true_pieces.len(),
    );

    // Update stats including false
    let stats = compute_stats(&true_pieces, &false_pieces);

    let mut pieces = Vec::new();
    pieces.append(&mut true_pieces);
    pieces.extend(false_pieces);

    Ok(Puzzle {
        side_len: cfg.side_len,
        edge_alphabet: cfg.edge_alphabet,
        true_piece_count: hex_board_coords(cfg.side_len).len(),
        false_piece_count: pieces.len() - hex_board_coords(cfg.side_len).len(),
        pieces,
        stats,
    })
}

#[derive(Debug, Clone)]
pub struct UniqueReport {
    /// 回転同値を除いて一意なら true
    pub unique_mod_rotation: bool,
    /// if non-unique, true when uniqueness is broken only by reflection (D6)
    pub nonunique_only_by_reflection: bool,
    /// 見つかった解の総数（回転同値も含むことがある）
    pub solutions_found: u64,
    /// 探索ノード数（難易度の目安に使える）
    pub nodes: u64,
}

/// axial (q,r) を60度回転（原点中心）
/// (q,r) -> (-r, q+r)
fn rot60(c: Axial) -> Axial {
    Axial::new(-c.r, c.q + c.r)
}

fn rot_k(mut c: Axial, k: usize) -> Axial {
    for _ in 0..k {
        c = rot60(c);
    }
    c
}

/// 盤面座標（正六角形 side_len）を生成してソート
fn board_coords_sorted(side_len: i32) -> Vec<Axial> {
    let mut coords = hex_board_coords(side_len);
    // 安定な順序（任意だが固定であればOK）
    coords.sort_by_key(|c| (c.r, c.q));
    coords
}

#[derive(Clone)]
struct Board {
    coords: Vec<Axial>,
    index: HashMap<Axial, usize>,
    neighbors: Vec<[Option<usize>; 6]>,
    boundary: Vec<[bool; 6]>,
}

impl Board {
    fn new(side_len: i32) -> Self {
        let coords = board_coords_sorted(side_len);
        let mut index = HashMap::new();
        for (i, &c) in coords.iter().enumerate() {
            index.insert(c, i);
        }

        let mut neighbors = vec![[None; 6]; coords.len()];
        let mut boundary = vec![[false; 6]; coords.len()];

        for (i, &c) in coords.iter().enumerate() {
            for (d, dv) in DIRS.iter().enumerate() {
                let n = c.add(*dv);
                if let Some(&j) = index.get(&n) {
                    neighbors[i][d] = Some(j);
                } else {
                    boundary[i][d] = true;
                }
            }
        }

        Self {
            coords,
            index,
            neighbors,
            boundary,
        }
    }

    fn n_cells(&self) -> usize {
        self.coords.len()
    }
}

// ---------------------------------------------------------------------------
// UNIQUE CHECKER: fix "always 2 solutions" caused by swapping identical pieces.
//
// Previously fingerprint encoded (piece_idx, rot). If two pieces are identical (same rotated edge labels),
// swapping them yields a "different" solution for the solver, even though it's indistinguishable physically.
//
// NEW: fingerprint encodes the *placed rotated edge array* (hashed), so identical pieces collapse.
// Also add an optional D6 (rotation+reflection) canonicalizer for diagnosing mirror-pairs.
// ---------------------------------------------------------------------------

fn fnv1a_u32_i32x6(edges: &[i32; 6]) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for &x in edges.iter() {
        for b in x.to_le_bytes() {
            h ^= b as u32;
            h = h.wrapping_mul(0x01000193);
        }
    }
    h
}

// Reflection across an axis (cube swap y<->z): axial (q,r) -> (q, -q-r)
fn reflect_axial(c: Axial) -> Axial {
    Axial::new(c.q, -c.q - c.r)
}

// Direction mapping under reflect_axial.
// Derived by applying the linear map (dq,dr)->(dq,-dq-dr) to DIRS.
const REFLECT_DIR_MAP: [usize; 6] = [1, 0, 5, 4, 3, 2];

fn reflect_edges(edges: &[i32; 6]) -> [i32; 6] {
    // new_dir = REFLECT_DIR_MAP[old_dir]
    // so for each old_dir, place edges[old_dir] into out[new_dir]
    let mut out = [0i32; 6];
    for old in 0..6 {
        let newd = REFLECT_DIR_MAP[old];
        out[newd] = edges[old];
    }
    out
}

fn canonical_solution_fingerprint_rot_only(
    board: &Board,
    assignment: &[Option<(usize, usize)>], // cell -> (piece_idx, rot)
    edges_rot: &Vec<[[i32; 6]; 6]>,
) -> Vec<u32> {
    let n = board.n_cells();
    let mut best: Option<Vec<u32>> = None;

    for k in 0..6 {
        let mut tmp = vec![0u32; n];
        for (i, &cell) in board.coords.iter().enumerate() {
            let Some((p, r)) = assignment[i] else {
                continue;
            };
            let cell2 = rot_k(cell, k);
            let j = board.index[&cell2];
            let r2 = (r + 6 - k) % 6;
            tmp[j] = fnv1a_u32_i32x6(&edges_rot[p][r2]);
        }
        match &best {
            None => best = Some(tmp),
            Some(b) => {
                if tmp < *b {
                    best = Some(tmp);
                }
            }
        }
    }
    best.unwrap()
}

fn canonical_solution_fingerprint_dihedral(
    board: &Board,
    assignment: &[Option<(usize, usize)>],
    edges_rot: &Vec<[[i32; 6]; 6]>,
) -> Vec<u32> {
    let n = board.n_cells();
    let mut best: Option<Vec<u32>> = None;

    for k in 0..6 {
        // rotation
        let mut tmp_r = vec![0u32; n];
        for (i, &cell) in board.coords.iter().enumerate() {
            let Some((p, r)) = assignment[i] else {
                continue;
            };
            let cell2 = rot_k(cell, k);
            let j = board.index[&cell2];
            let r2 = (r + 6 - k) % 6;
            tmp_r[j] = fnv1a_u32_i32x6(&edges_rot[p][r2]);
        }
        best = match &best {
            None => Some(tmp_r),
            Some(b) => Some(if tmp_r < *b { tmp_r } else { b.clone() }),
        };

        // rotation + reflection
        let mut tmp_rf = vec![0u32; n];
        for (i, &cell) in board.coords.iter().enumerate() {
            let Some((p, r)) = assignment[i] else {
                continue;
            };
            let cell2 = reflect_axial(rot_k(cell, k));
            let j = board.index[&cell2];
            let r2 = (r + 6 - k) % 6;
            let mirrored = reflect_edges(&edges_rot[p][r2]);
            tmp_rf[j] = fnv1a_u32_i32x6(&mirrored);
        }
        best = match &best {
            None => Some(tmp_rf),
            Some(b) => Some(if tmp_rf < *b { tmp_rf } else { b.clone() }),
        };
    }

    best.unwrap()
}

struct UniquenessSolver {
    board: Board,
    pieces: Vec<Piece>, // 真ピースのみを渡す想定

    // edges_rot[piece_idx][rot][dir]
    edges_rot: Vec<[[i32; 6]; 6]>,

    // (dir, value) -> placements (piece_idx*6+rot)
    idx: HashMap<(u8, i32), Vec<u16>>,

    used: Vec<bool>,
    assignment: Vec<Option<(usize, usize)>>,

    first_fingerprint: Option<Vec<u32>>,
    first_fingerprint_d6: Option<Vec<u32>>,
    nonunique_only_by_reflection: bool,
    nonunique: bool,

    solutions_found: u64,
    nodes: u64,
}

impl UniquenessSolver {
    fn new(side_len: i32, pieces: &[Piece]) -> Self {
        let board = Board::new(side_len);
        let pieces = pieces.to_vec();

        let n_p = pieces.len();
        let mut edges_rot = vec![[[0i32; 6]; 6]; n_p];
        for p in 0..n_p {
            for r in 0..6 {
                edges_rot[p][r] = rotate_edges(&pieces[p].edges, r);
            }
        }

        // 制約インデックス
        let mut idx: HashMap<(u8, i32), Vec<u16>> = HashMap::new();
        for p in 0..n_p {
            for r in 0..6 {
                let placement = (p * 6 + r) as u16;
                for d in 0..6 {
                    let v = edges_rot[p][r][d];
                    idx.entry((d as u8, v)).or_default().push(placement);
                }
            }
        }

        Self {
            board,
            pieces,
            edges_rot,
            idx,
            used: vec![false; n_p],
            assignment: vec![None; Board::new(side_len).n_cells()],
            first_fingerprint: None,
            first_fingerprint_d6: None,
            nonunique_only_by_reflection: false,
            nonunique: false,
            solutions_found: 0,
            nodes: 0,
        }
    }

    fn constraints_for_cell(&self, cell: usize) -> Vec<(u8, i32)> {
        let mut cons = Vec::new();

        // 外周は0必須
        for d in 0..6 {
            if self.board.boundary[cell][d] {
                cons.push((d as u8, 0));
            }
        }

        // 近傍の確定セルから要求値を作る
        for d in 0..6 {
            if let Some(nb) = self.board.neighbors[cell][d] {
                if let Some((p2, r2)) = self.assignment[nb] {
                    let v_nb = self.edges_rot[p2][r2][opposite_dir(d)];
                    cons.push((d as u8, -v_nb));
                }
            }
        }

        cons
    }

    fn candidates_for_cell(&self, cell: usize) -> Vec<u16> {
        let cons = self.constraints_for_cell(cell);
        if cons.is_empty() {
            // 制約が何も無い場合：全unused×6 が候補（ただし現実には起きにくい）
            let mut out = Vec::new();
            for p in 0..self.pieces.len() {
                if self.used[p] {
                    continue;
                }
                for r in 0..6 {
                    out.push((p * 6 + r) as u16);
                }
            }
            return out;
        }

        // 制約リストを「候補が一番少ないインデックス」順に
        let mut cons_sorted = cons;
        cons_sorted.sort_by_key(|&(d, v)| self.idx.get(&(d, v)).map(|x| x.len()).unwrap_or(0));

        let (d0, v0) = cons_sorted[0];
        let Some(base) = self.idx.get(&(d0, v0)) else {
            return vec![];
        };

        let mut out = Vec::new();
        'cand: for &pl in base.iter() {
            let pl_us = pl as usize;
            let p = pl_us / 6;
            let r = pl_us % 6;
            if self.used[p] {
                continue;
            }
            // すべての制約を満たすかチェック
            for &(d, v) in cons_sorted.iter().skip(1) {
                if self.edges_rot[p][r][d as usize] != v {
                    continue 'cand;
                }
            }
            out.push(pl);
        }

        out
    }

    fn pick_mrv_cell(&self) -> Option<(usize, Vec<u16>)> {
        let mut best_cell: Option<usize> = None;
        let mut best_cands: Vec<u16> = Vec::new();
        let mut best_count: usize = usize::MAX;

        for cell in 0..self.board.n_cells() {
            if self.assignment[cell].is_some() {
                continue;
            }
            let cands = self.candidates_for_cell(cell);
            let cnt = cands.len();
            if cnt == 0 {
                return Some((cell, cands)); // 即死（枝刈り）
            }
            if cnt < best_count {
                best_count = cnt;
                best_cell = Some(cell);
                best_cands = cands;
                if best_count == 1 {
                    // これ以上小さいのは0だけなので早期終了
                    //（0は上で返っている）
                    break;
                }
            }
        }

        best_cell.map(|c| (c, best_cands))
    }

    fn handle_solution(&mut self) {
        self.solutions_found += 1;

        let fp =
            canonical_solution_fingerprint_rot_only(&self.board, &self.assignment, &self.edges_rot);
        let fp_d6 =
            canonical_solution_fingerprint_dihedral(&self.board, &self.assignment, &self.edges_rot);

        match &self.first_fingerprint {
            None => {
                self.first_fingerprint = Some(fp);
                self.first_fingerprint_d6 = Some(fp_d6);
            }
            Some(first) => {
                if &fp != first {
                    self.nonunique = true;
                    // Diagnose whether difference disappears under reflection
                    if let Some(first_d6) = &self.first_fingerprint_d6 {
                        if &fp_d6 == first_d6 {
                            self.nonunique_only_by_reflection = true;
                        }
                    }
                }
            }
        }
    }

    fn dfs(&mut self, filled: usize) {
        if self.nonunique {
            return;
        }
        if filled == self.board.n_cells() {
            self.handle_solution();
            return;
        }

        let Some((cell, cands)) = self.pick_mrv_cell() else {
            return;
        };
        if cands.is_empty() {
            return;
        }

        // 分岐：候補を試す
        for pl in cands {
            if self.nonunique {
                return;
            }
            self.nodes += 1;

            let pl_us = pl as usize;
            let p = pl_us / 6;
            let r = pl_us % 6;

            // 置く
            self.used[p] = true;
            self.assignment[cell] = Some((p, r));

            self.dfs(filled + 1);

            // 戻す
            self.assignment[cell] = None;
            self.used[p] = false;
        }
    }

    fn run(mut self) -> UniqueReport {
        self.dfs(0);

        UniqueReport {
            unique_mod_rotation: !self.nonunique,
            nonunique_only_by_reflection: self.nonunique_only_by_reflection,
            solutions_found: self.solutions_found,
            nodes: self.nodes,
        }
    }
}

/// 真ピースのみで、回転同値まで一意かを判定
fn check_unique_true(side_len: i32, true_pieces: &[Piece]) -> UniqueReport {
    UniquenessSolver::new(side_len, true_pieces).run()
}

#[derive(Debug, Clone)]
pub struct ExistenceReport {
    pub exists: bool,
    pub nodes: u64,
}

/// 「偽を1枚以上使う完全敷き詰め解」が存在するかをチェック
fn exists_solution_using_false(side_len: i32, pieces: &[Piece]) -> ExistenceReport {
    let mut solver = ExistenceSolver::new(side_len, pieces, true);
    solver.run()
}

/// 「（偽/真問わず）何らかの完全敷き詰め解」が存在するかをチェック
/// ※真だけで解があるなら、通常こちらは必ず true になる
#[allow(dead_code)]
fn exists_any_solution(side_len: i32, pieces: &[Piece]) -> ExistenceReport {
    let mut solver = ExistenceSolver::new(side_len, pieces, false);
    solver.run()
}

struct ExistenceSolver {
    board: Board,
    pieces: Vec<Piece>,
    edges_rot: Vec<[[i32; 6]; 6]>,
    idx: HashMap<(u8, i32), Vec<u16>>,

    used: Vec<bool>,
    assignment: Vec<Option<(usize, usize)>>,

    require_at_least_one_false: bool,
    found: bool,
    nodes: u64,
}

impl ExistenceSolver {
    fn new(side_len: i32, pieces: &[Piece], require_at_least_one_false: bool) -> Self {
        let board = Board::new(side_len);
        let pieces = pieces.to_vec();

        let n_p = pieces.len();
        let mut edges_rot = vec![[[0i32; 6]; 6]; n_p];
        for p in 0..n_p {
            for r in 0..6 {
                edges_rot[p][r] = rotate_edges(&pieces[p].edges, r);
            }
        }

        // (dir,value) -> placements
        let mut idx: HashMap<(u8, i32), Vec<u16>> = HashMap::new();
        for p in 0..n_p {
            for r in 0..6 {
                let pl = (p * 6 + r) as u16;
                for d in 0..6 {
                    idx.entry((d as u8, edges_rot[p][r][d]))
                        .or_default()
                        .push(pl);
                }
            }
        }

        Self {
            board,
            pieces,
            edges_rot,
            idx,
            used: vec![false; n_p],
            assignment: vec![None; Board::new(side_len).n_cells()],
            require_at_least_one_false,
            found: false,
            nodes: 0,
        }
    }

    fn constraints_for_cell(&self, cell: usize) -> Vec<(u8, i32)> {
        let mut cons = Vec::new();

        // boundary edges must be 0
        for d in 0..6 {
            if self.board.boundary[cell][d] {
                cons.push((d as u8, 0));
            }
        }

        // neighbor-implied constraints
        for d in 0..6 {
            if let Some(nb) = self.board.neighbors[cell][d] {
                if let Some((p2, r2)) = self.assignment[nb] {
                    let v_nb = self.edges_rot[p2][r2][opposite_dir(d)];
                    cons.push((d as u8, -v_nb));
                }
            }
        }

        cons
    }

    fn candidates_for_cell(&self, cell: usize) -> Vec<u16> {
        let mut cons = self.constraints_for_cell(cell);
        if cons.is_empty() {
            // no constraints: all unused placements
            let mut out = Vec::new();
            for p in 0..self.pieces.len() {
                if self.used[p] {
                    continue;
                }
                for r in 0..6 {
                    out.push((p * 6 + r) as u16);
                }
            }
            return out;
        }

        // sort constraints by smallest posting list
        cons.sort_by_key(|&(d, v)| self.idx.get(&(d, v)).map(|x| x.len()).unwrap_or(0));

        let (d0, v0) = cons[0];
        let Some(base) = self.idx.get(&(d0, v0)) else {
            return vec![];
        };

        let mut out = Vec::new();
        'cand: for &pl in base.iter() {
            let pu = pl as usize;
            let p = pu / 6;
            let r = pu % 6;
            if self.used[p] {
                continue;
            }
            for &(d, v) in cons.iter().skip(1) {
                if self.edges_rot[p][r][d as usize] != v {
                    continue 'cand;
                }
            }
            out.push(pl);
        }
        out
    }

    fn pick_mrv_cell(&self) -> Option<(usize, Vec<u16>)> {
        let mut best_cell: Option<usize> = None;
        let mut best_cands: Vec<u16> = Vec::new();
        let mut best_cnt = usize::MAX;

        for cell in 0..self.board.n_cells() {
            if self.assignment[cell].is_some() {
                continue;
            }
            let cands = self.candidates_for_cell(cell);
            let cnt = cands.len();
            if cnt == 0 {
                return Some((cell, cands)); // dead end
            }
            if cnt < best_cnt {
                best_cnt = cnt;
                best_cell = Some(cell);
                best_cands = cands;
                if best_cnt == 1 {
                    break;
                }
            }
        }

        best_cell.map(|c| (c, best_cands))
    }

    fn dfs(&mut self, filled: usize, used_false: bool) {
        if self.found {
            return;
        }
        if filled == self.board.n_cells() {
            if !self.require_at_least_one_false || used_false {
                self.found = true;
            }
            return;
        }

        let Some((cell, cands)) = self.pick_mrv_cell() else {
            return;
        };
        if cands.is_empty() {
            return;
        }

        for pl in cands {
            if self.found {
                return;
            }
            self.nodes += 1;

            let pu = pl as usize;
            let p = pu / 6;
            let r = pu % 6;

            // place
            self.used[p] = true;
            self.assignment[cell] = Some((p, r));

            let used_false2 = used_false || matches!(self.pieces[p].kind, PieceKind::False);
            self.dfs(filled + 1, used_false2);

            // undo
            self.assignment[cell] = None;
            self.used[p] = false;
        }
    }

    fn run(&mut self) -> ExistenceReport {
        self.dfs(0, false);
        ExistenceReport {
            exists: self.found,
            nodes: self.nodes,
        }
    }
}
