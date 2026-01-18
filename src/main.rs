use anyhow::Result;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
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

    /// False cluster spec: list of (cluster_size, count)
    false_clusters: Vec<(usize, usize)>,

    /// For each boundary edge of a false cluster (an edge not connected to another false piece in same cluster):
    /// - with prob flat_ratio => set to 0 (looks like boundary)
    /// - else with prob attach_ratio => reuse some existing true edge id (to create "warped match")
    /// - else => unmatched unique id (no counterpart)
    false_flat_ratio: f64,
    false_attach_ratio: f64,
}

fn main() -> Result<()> {
    // Example: "one step harder" than side_len=7
    // True tiles: 3*s*(s-1)+1 = 169 when s=8
    // False: bump them; example uses your previous cluster pattern + extras
    let cfg = Config {
        side_len: 8,
        edge_alphabet: 14, // SMALL => very ambiguous => "全探索っぽい"へ寄る
        seed: 0x1A11_4D3C_2B1A_0F00u64, // pick any u64; hex-literal not required

        enforce_unique_true_shapes: true,
        max_true_regen: 500,

        // You can keep your earlier pattern and add more clusters for more pain:
        // 1*6 + 2*6 + 4*3 + 7*2 = 6 + 12 + 12 + 14 = 44 false pieces
        false_clusters: vec![(1, 6), (2, 6), (4, 3), (7, 2)],
        false_flat_ratio: 0.25,
        false_attach_ratio: 0.55,
    };

    let puzzle = generate_puzzle(&cfg)?;
    println!("{}", serde_json::to_string_pretty(&puzzle)?);
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
    enforce_unique_shapes: bool,
    max_regen: usize,
) -> Result<(Vec<Piece>, Vec<i32>, Stats)> {
    let coords = hex_board_coords(side_len);
    let coord_set: HashSet<Axial> = coords.iter().copied().collect();

    // Precompute all undirected adjacencies (a < b)
    // Store edge id per undirected pair.
    // Key: (min, max)
    let mut attempt = 0usize;

    loop {
        attempt += 1;
        let mut edge_id_map: HashMap<(Axial, Axial), i32> = HashMap::new();

        // Assign random id in 1..=edge_alphabet to each adjacency
        for &a in &coords {
            for (dir_idx, d) in DIRS.iter().enumerate() {
                let b = a.add(*d);
                if !coord_set.contains(&b) {
                    continue;
                }
                let (minc, maxc) = if a < b { (a, b) } else { (b, a) };
                edge_id_map
                    .entry((minc, maxc))
                    .or_insert_with(|| rng.random_range(1..=edge_alphabet));
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

    let (mut true_pieces, true_edge_ids_abs, mut stats) = generate_true_pieces(
        &mut rng,
        cfg.side_len,
        cfg.edge_alphabet,
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
    stats = compute_stats(&true_pieces, &false_pieces);

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
