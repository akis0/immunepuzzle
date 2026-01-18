use anyhow::{self, Result};
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
    side_len: i32,

    /// geometry dictionary max (<=295)
    total_alphabet: i32,

    /// true pieces use IDs in 1..=true_alphabet (<= total_alphabet)
    true_alphabet: i32,

    /// normal (collision) IDs are drawn from 1..=core_alphabet (<= true_alphabet)
    core_alphabet: i32,

    seed: u64,

    enforce_unique_true_shapes: bool,
    max_true_regen: usize,

    /// fraction of internal adjacencies to become "anchors" (unique IDs)
    anchor_ratio: f64,

    false_clusters: Vec<(usize, usize)>,
    false_flat_ratio: f64,
    false_attach_ratio: f64,
}

fn random_seed() -> u64 {
    rand::random::<u64>()
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
    // --alphabet は「真ピースが使うID上限」として解釈する
    let true_alphabet: i32 = parse_opt_i32("--alphabet").unwrap_or(30);
    // total は幾何辞書の上限(<=295)。未指定なら 295
    let total_alphabet: i32 = parse_opt_i32("--total-alphabet").unwrap_or(295);
    let core_alphabet: i32 = parse_opt_i32("--core-alphabet").unwrap_or(true_alphabet);

    let anchor_ratio: f64 = parse_opt_f64("--anchor-ratio").unwrap_or(0.15);

    // 偽の境界挙動
    let false_flat_ratio: f64 = parse_opt_f64("--false-flat").unwrap_or(0.35);
    let false_attach_ratio: f64 = parse_opt_f64("--false-attach").unwrap_or(0.65);

    // 偽塊スペック（例：強め。必要ならここもCLI化してOK）
    // 合計: 1*6 + 2*6 + 4*3 + 7*2 = 44
    let false_clusters: Vec<(usize, usize)> = vec![(1, 10), (2, 10), (4, 4), (7, 2)];

    // 真ピース生成時の「回転同値で形状が被らない」方向の選抜（保険）
    let enforce_unique_true_shapes = true;
    let max_true_regen = 500;

    // ---- ここから選抜ループ（免疫選抜） ----
    // best = (puzzle, seed, hardness_score, uniq_nodes, false_nodes)
    let mut best: Option<(Puzzle, u64, u64, u64, u64)> = None;

    let edge_meta = precompute_edge_meta(total_alphabet);

    // キャッシュは試行間で共有（大量ヒットして速くなる）
    // max_entries は環境に合わせて（10万〜50万くらいから）
    let mut geom_memo = GeomMemo::new(300_000);

    let min_gap = 0.2;

    for t in 0..trials {
        // seed：--seedが指定されていれば最初だけそれを使い、以降はランダム
        let seed = match fixed_seed {
            Some(s) if t == 0 => s,
            _ => random_seed(),
        };

        let cfg = Config {
            side_len,
            total_alphabet,
            true_alphabet,
            core_alphabet,
            seed,
            enforce_unique_true_shapes,
            max_true_regen,
            anchor_ratio,
            false_clusters: false_clusters.clone(),
            false_flat_ratio,
            false_attach_ratio,
        };

        let puzzle = generate_puzzle(&cfg, &mut geom_memo, &edge_meta, min_gap)?;

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

        let solution = build_solution(p.side_len, &p);

        let out = Output {
            puzzle: p,
            solution,
        };
        println!("{}", serde_json::to_string_pretty(&out)?);
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
    true_alphabet: i32,
    anchor_ratio: f64,
    enforce_unique_shapes: bool,
    max_regen: usize,
    geom_memo: &mut GeomMemo,
    edge_meta: &[EdgeMeta],
    min_gap: f64,
) -> Result<(Vec<Piece>, Vec<i32>, Stats)> {
    let coords = hex_board_coords(side_len);
    let coord_set: HashSet<Axial> = coords.iter().copied().collect();

    if true_alphabet <= 0 || true_alphabet > 295 {
        anyhow::bail!("true_alphabet must be in 1..=295 for geometry mapping");
    }

    let mut attempt = 0usize;

    loop {
        attempt += 1;
        let mut edge_id_map: HashMap<(Axial, Axial), i32> = HashMap::new();

        // Assign id in 1..=true_alphabet to each adjacency
        for &a in &coords {
            for d in DIRS.iter() {
                let b = a.add(*d);
                if !coord_set.contains(&b) {
                    continue;
                }
                let (minc, maxc) = if a < b { (a, b) } else { (b, a) };
                edge_id_map.entry((minc, maxc)).or_insert_with(|| {
                    let roll: f64 = rng.random();
                    if roll < anchor_ratio && true_alphabet >= 50 {
                        let lo = ((true_alphabet as f64) * 0.8).ceil() as i32;
                        let lo = lo.max(1).min(true_alphabet);
                        rng.random_range(lo..=true_alphabet)
                    } else {
                        rng.random_range(1..=true_alphabet)
                    }
                });
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

        // 幾何チェック（危険度が高い順に）＋メモ化
        {
            let mut order: Vec<usize> = (0..pieces.len()).collect();
            order.sort_by(|&i, &j| {
                let ri = piece_risk_score(&pieces[i].edges, edge_meta);
                let rj = piece_risk_score(&pieces[j].edges, edge_meta);
                rj.partial_cmp(&ri).unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut ok = true;
            for i in order {
                if !geom_memo.geom_ok(pieces[i].edges, min_gap) {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
        }

        // Optional uniqueness filter on canonical shapes
        if enforce_unique_shapes {
            let mut true_shape_set: HashSet<[i32; 6]> = HashSet::new();
            for p in &pieces {
                true_shape_set.insert(canonical_shape(&p.edges));
            }
            if true_shape_set.len() != pieces.len() {
                if attempt >= max_regen {
                    let stats = compute_stats(&pieces, &[]);
                    return Ok((pieces, all_internal_edge_ids_abs, stats));
                }
                continue;
            }
        }

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
    total_alphabet: i32,
    true_alphabet: i32,
    true_edge_ids_abs: &[i32],
    cluster_spec: &[(usize, usize)],
    flat_ratio: f64,
    attach_ratio: f64,
    start_piece_id: usize,
    geom_memo: &mut GeomMemo,
    edge_meta: &[EdgeMeta],
    min_gap: f64,
) -> Vec<Piece> {
    if total_alphabet <= 0 || total_alphabet > 295 {
        panic!("total_alphabet must be in 1..=295");
    }
    if true_alphabet <= 0 || true_alphabet > total_alphabet {
        panic!("true_alphabet must be in 1..=total_alphabet");
    }

    let unmatched_lo = true_alphabet + 1;
    let unmatched_hi = total_alphabet;

    let mut pieces: Vec<Piece> = Vec::new();
    let mut pid = start_piece_id;
    let mut cluster_id_counter: usize = 0;

    for &(size, count) in cluster_spec {
        for _ in 0..count {
            let mut tries = 0;

            let edge_arrays_opt: Option<Vec<[i32; 6]>> = loop {
                tries += 1;
                if tries > 400 {
                    break None;
                }

                let coords = random_connected_cluster_coords(rng, size);
                let coord_to_idx: HashMap<Axial, usize> =
                    coords.iter().enumerate().map(|(i, &c)| (c, i)).collect();

                let mut edge_arrays: Vec<[i32; 6]> = vec![[0i32; 6]; size];

                let mut internal_map: HashMap<(Axial, Axial), i32> = HashMap::new();
                for &a in &coords {
                    for (dir_idx, d) in DIRS.iter().enumerate() {
                        let b = a.add(*d);
                        if !coord_to_idx.contains_key(&b) {
                            continue;
                        }
                        let (minc, maxc) = if a < b { (a, b) } else { (b, a) };
                        let id = *internal_map.entry((minc, maxc)).or_insert_with(|| {
                            // まずは偽専用レンジ（真に噛みにくい）を優先
                            if unmatched_lo <= unmatched_hi {
                                rng.random_range(unmatched_lo..=unmatched_hi)
                            } else {
                                rng.random_range(1..=true_alphabet)
                            }
                        });

                        let a_idx = coord_to_idx[&a];
                        let b_idx = coord_to_idx[&b];
                        edge_arrays[a_idx][dir_idx] = if a == minc { id } else { -id };
                        edge_arrays[b_idx][opposite_dir(dir_idx)] =
                            if b == minc { id } else { -id };
                    }
                }

                for i in 0..size {
                    for dir_idx in 0..6 {
                        if edge_arrays[i][dir_idx] != 0 {
                            continue;
                        }

                        let roll: f64 = rng.random();
                        if roll < flat_ratio {
                            edge_arrays[i][dir_idx] = 0;
                        } else if roll < flat_ratio + attach_ratio && !true_edge_ids_abs.is_empty()
                        {
                            let v = true_edge_ids_abs[rng.random_range(0..true_edge_ids_abs.len())]
                                .abs();
                            edge_arrays[i][dir_idx] = if rng.gen_bool(0.5) { v } else { -v };
                        } else {
                            // 適合なし（真に噛まないレンジ）
                            if unmatched_lo <= unmatched_hi {
                                let v = rng.random_range(unmatched_lo..=unmatched_hi);
                                edge_arrays[i][dir_idx] = if rng.gen_bool(0.5) { v } else { -v };
                            } else {
                                let v = rng.random_range(1..=true_alphabet);
                                edge_arrays[i][dir_idx] = if rng.gen_bool(0.5) { v } else { -v };
                            }
                        }
                    }
                }

                // 幾何チェック（危険度順）＋メモ化
                let mut order: Vec<usize> = (0..size).collect();
                order.sort_by(|&i, &j| {
                    let ri = piece_risk_score(&edge_arrays[i], edge_meta);
                    let rj = piece_risk_score(&edge_arrays[j], edge_meta);
                    rj.partial_cmp(&ri).unwrap_or(std::cmp::Ordering::Equal)
                });

                let mut ok = true;
                for i in order {
                    if !geom_memo.geom_ok(edge_arrays[i], min_gap) {
                        ok = false;
                        break;
                    }
                }

                if ok {
                    break Some(edge_arrays);
                }
            };

            let Some(edge_arrays) = edge_arrays_opt else {
                continue;
            };

            cluster_id_counter += 1;
            for e in edge_arrays {
                pieces.push(Piece {
                    id: pid,
                    kind: PieceKind::False,
                    cluster_id: Some(cluster_id_counter),
                    edges: e,
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

fn generate_puzzle(
    cfg: &Config,
    geom_memo: &mut GeomMemo,
    edge_meta: &[EdgeMeta],
    min_gap: f64,
) -> Result<Puzzle> {
    let mut rng = StdRng::seed_from_u64(cfg.seed);

    let (mut true_pieces, true_edge_ids_abs, _true_stats) = generate_true_pieces(
        &mut rng,
        cfg.side_len,
        cfg.true_alphabet,
        cfg.anchor_ratio,
        cfg.enforce_unique_true_shapes,
        cfg.max_true_regen,
        geom_memo,
        edge_meta,
        min_gap,
    )?;

    let false_pieces = generate_false_pieces(
        &mut rng,
        cfg.total_alphabet,
        cfg.true_alphabet,
        &true_edge_ids_abs,
        &cfg.false_clusters,
        cfg.false_flat_ratio,
        cfg.false_attach_ratio,
        true_pieces.len(),
        geom_memo,
        edge_meta,
        min_gap,
    );

    let stats = compute_stats(&true_pieces, &false_pieces);

    let mut pieces = Vec::new();
    pieces.append(&mut true_pieces);
    pieces.extend(false_pieces);

    Ok(Puzzle {
        side_len: cfg.side_len,
        edge_alphabet: cfg.total_alphabet,
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
            out.sort_by_key(|&pl| {
                let p = (pl as usize) / 6;
                match self.pieces[p].kind {
                    PieceKind::False => 0, // 偽を先に試す
                    PieceKind::True => 1,
                }
            });
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

    fn dfs_from_state(&mut self, filled: usize, used_false: bool) {
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

            self.used[p] = true;
            self.assignment[cell] = Some((p, r));

            let used_false2 = used_false || matches!(self.pieces[p].kind, PieceKind::False);
            self.dfs_from_state(filled + 1, used_false2);

            self.assignment[cell] = None;
            self.used[p] = false;
        }
    }

    fn try_false_anchor_first(&mut self, max_anchors: usize) {
        if self.found {
            return;
        }

        // まず「候補が少ないセル」から false を試す方が効くので、
        // 未配置セルの中から MRV に近い順で先頭 max_anchors 個だけ試す。
        let mut cells: Vec<(usize, usize)> = Vec::new(); // (cell, cand_count)

        for cell in 0..self.board.n_cells() {
            if self.assignment[cell].is_some() {
                continue;
            }
            let cands = self.candidates_for_cell(cell);
            let cnt = cands.len();
            if cnt == 0 {
                return;
            }
            cells.push((cell, cnt));
        }

        cells.sort_by_key(|&(_, cnt)| cnt);

        for &(cell, _) in cells.iter().take(max_anchors) {
            if self.found {
                return;
            }

            let mut cands = self.candidates_for_cell(cell);

            // 偽だけに絞る
            cands.retain(|&pl| {
                let p = (pl as usize) / 6;
                matches!(self.pieces[p].kind, PieceKind::False)
            });

            for pl in cands {
                if self.found {
                    return;
                }
                self.nodes += 1;

                let pu = pl as usize;
                let p = pu / 6;
                let r = pu % 6;

                self.used[p] = true;
                self.assignment[cell] = Some((p, r));

                // ここからは used_false=true で探索
                self.dfs_from_state(1, true);

                self.assignment[cell] = None;
                self.used[p] = false;
            }
        }
    }

    fn run(&mut self) -> ExistenceReport {
        // 2段階：偽アンカー → 通常探索
        if self.require_at_least_one_false {
            self.try_false_anchor_first(12); // まず12セルだけ（調整可）
        }
        if !self.found {
            self.dfs_from_state(0, false);
        }
        ExistenceReport {
            exists: self.found,
            nodes: self.nodes,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Placement {
    q: i32,
    r: i32,
    piece_id: usize, // Piece.id を出す（インデックスではなく）
    rot: usize,      // 0..5
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Solution {
    placements: Vec<Placement>,   // 長さ = 盤面セル数（例: 169）
    unused_piece_ids: Vec<usize>, // 偽ピースなど（使わないピースのID）
}

fn solve_one_true(side_len: i32, true_pieces: &[Piece]) -> Option<Vec<Option<(usize, usize)>>> {
    // 返り値: cell -> (true_pieces index, rot)
    let board = Board::new(side_len);

    let n_p = true_pieces.len();
    let n_c = board.n_cells();

    // edges_rot[p][rot][dir]
    let mut edges_rot = vec![[[0i32; 6]; 6]; n_p];
    for p in 0..n_p {
        for rot in 0..6 {
            edges_rot[p][rot] = rotate_edges(&true_pieces[p].edges, rot);
        }
    }

    // (dir, value) -> placements
    let mut idx: HashMap<(u8, i32), Vec<u16>> = HashMap::new();
    for p in 0..n_p {
        for rot in 0..6 {
            let pl = (p * 6 + rot) as u16;
            for d in 0..6 {
                idx.entry((d as u8, edges_rot[p][rot][d]))
                    .or_default()
                    .push(pl);
            }
        }
    }

    let mut used = vec![false; n_p];
    let mut assign: Vec<Option<(usize, usize)>> = vec![None; n_c];

    fn constraints_for_cell(
        cell: usize,
        board: &Board,
        assign: &[Option<(usize, usize)>],
        edges_rot: &Vec<[[i32; 6]; 6]>,
    ) -> Vec<(u8, i32)> {
        let mut cons = Vec::new();

        // boundary => 0
        for d in 0..6 {
            if board.boundary[cell][d] {
                cons.push((d as u8, 0));
            }
        }
        // neighbors
        for d in 0..6 {
            if let Some(nb) = board.neighbors[cell][d] {
                if let Some((p2, r2)) = assign[nb] {
                    let v_nb = edges_rot[p2][r2][opposite_dir(d)];
                    cons.push((d as u8, -v_nb));
                }
            }
        }
        cons
    }

    fn candidates_for_cell(
        cell: usize,
        board: &Board,
        assign: &[Option<(usize, usize)>],
        edges_rot: &Vec<[[i32; 6]; 6]>,
        idx: &HashMap<(u8, i32), Vec<u16>>,
        used: &[bool],
        n_p: usize,
    ) -> Vec<u16> {
        let mut cons = constraints_for_cell(cell, board, assign, edges_rot);

        if cons.is_empty() {
            let mut out = Vec::new();
            for p in 0..n_p {
                if used[p] {
                    continue;
                }
                for r in 0..6 {
                    out.push((p * 6 + r) as u16);
                }
            }
            return out;
        }

        cons.sort_by_key(|&(d, v)| idx.get(&(d, v)).map(|x| x.len()).unwrap_or(0));

        let (d0, v0) = cons[0];
        let Some(base) = idx.get(&(d0, v0)) else {
            return vec![];
        };

        let mut out = Vec::new();
        'cand: for &pl in base.iter() {
            let pu = pl as usize;
            let p = pu / 6;
            let r = pu % 6;
            if used[p] {
                continue;
            }
            for &(d, v) in cons.iter().skip(1) {
                if edges_rot[p][r][d as usize] != v {
                    continue 'cand;
                }
            }
            out.push(pl);
        }
        out
    }

    fn pick_mrv_cell(
        board: &Board,
        assign: &[Option<(usize, usize)>],
        edges_rot: &Vec<[[i32; 6]; 6]>,
        idx: &HashMap<(u8, i32), Vec<u16>>,
        used: &[bool],
        n_p: usize,
    ) -> Option<(usize, Vec<u16>)> {
        let mut best_cell: Option<usize> = None;
        let mut best_cands: Vec<u16> = Vec::new();
        let mut best_cnt = usize::MAX;

        for cell in 0..board.n_cells() {
            if assign[cell].is_some() {
                continue;
            }
            let cands = candidates_for_cell(cell, board, assign, edges_rot, idx, used, n_p);
            let cnt = cands.len();
            if cnt == 0 {
                return Some((cell, cands));
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

    fn dfs(
        filled: usize,
        board: &Board,
        assign: &mut Vec<Option<(usize, usize)>>,
        used: &mut Vec<bool>,
        edges_rot: &Vec<[[i32; 6]; 6]>,
        idx: &HashMap<(u8, i32), Vec<u16>>,
        n_p: usize,
    ) -> bool {
        if filled == board.n_cells() {
            return true;
        }

        let Some((cell, cands)) = pick_mrv_cell(board, assign, edges_rot, idx, used, n_p) else {
            return false;
        };
        if cands.is_empty() {
            return false;
        }

        for pl in cands {
            let pu = pl as usize;
            let p = pu / 6;
            let r = pu % 6;

            used[p] = true;
            assign[cell] = Some((p, r));

            if dfs(filled + 1, board, assign, used, edges_rot, idx, n_p) {
                return true;
            }

            assign[cell] = None;
            used[p] = false;
        }
        false
    }

    if dfs(0, &board, &mut assign, &mut used, &edges_rot, &idx, n_p) {
        Some(assign)
    } else {
        None
    }
}

#[derive(Serialize)]
struct Output {
    puzzle: Puzzle,
    solution: Solution,
}

fn build_solution(side_len: i32, puzzle: &Puzzle) -> Solution {
    let true_n = puzzle.true_piece_count;
    let true_pieces = &puzzle.pieces[..true_n];

    let assign = solve_one_true(side_len, true_pieces).expect("true tiling should exist");

    let board = Board::new(side_len);

    let mut placements = Vec::with_capacity(board.n_cells());
    for (cell_idx, a) in assign.iter().enumerate() {
        let (pidx, rot) = a.expect("fully assigned");
        let c = board.coords[cell_idx];
        placements.push(Placement {
            q: c.q,
            r: c.r,
            piece_id: true_pieces[pidx].id, // Piece.id を出す
            rot,
        });
    }

    // 使わないのは偽ピース（true_n以降）。真は全部使う想定。
    let unused_piece_ids = puzzle.pieces[true_n..].iter().map(|p| p.id).collect();

    Solution {
        placements,
        unused_piece_ids,
    }
}

use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct Pt {
    pub x: f64,
    pub y: f64,
}

impl Pt {
    pub fn add(self, o: Pt) -> Pt {
        Pt {
            x: self.x + o.x,
            y: self.y + o.y,
        }
    }
    pub fn sub(self, o: Pt) -> Pt {
        Pt {
            x: self.x - o.x,
            y: self.y - o.y,
        }
    }
    pub fn mul(self, k: f64) -> Pt {
        Pt {
            x: self.x * k,
            y: self.y * k,
        }
    }
    pub fn dot(self, o: Pt) -> f64 {
        self.x * o.x + self.y * o.y
    }
    pub fn norm2(self) -> f64 {
        self.dot(self)
    }
    pub fn norm(self) -> f64 {
        self.norm2().sqrt()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Seg {
    pub a: Pt,
    pub b: Pt,
}

#[derive(Clone, Copy, Debug)]
pub enum EdgeSpec {
    None,
    Trapezoid { l: f64, t0: f64, h0: f64, h1: f64 }, // 7*5*5 = 175
    Triangle { l: f64, t0: f64, a: f64, h: f64 },    // 60+60 = 120
}

#[derive(Clone, Debug)]
pub struct PieceGeom {
    /// Outer boundary polyline (closed, CCW). First point not repeated at end.
    pub outline: Vec<Pt>,
    /// For each concave notch on this piece, store the "inner boundary" segments of that notch.
    pub concave_inner_segments: Vec<Vec<Seg>>,
}

/// ---- ID decode ----
/// abs_id in 1..=295
pub fn decode_edge_spec(abs_id: i32) -> Option<EdgeSpec> {
    if abs_id <= 0 || abs_id > 295 {
        return None;
    }
    if abs_id <= 175 {
        Some(decode_trapezoid(abs_id))
    } else {
        Some(decode_triangle(abs_id))
    }
}

fn decode_pos_idx(pos_idx: i32) -> (f64, f64) {
    // pos_idx 0..6 -> (l, t0)
    // 0..2: l=0.4, t0=0.2,0.3,0.4
    // 3..6: l=0.3, t0=0.2,0.3,0.4,0.5
    if pos_idx <= 2 {
        let t0 = 0.2 + 0.1 * (pos_idx as f64);
        (0.4, t0)
    } else {
        let t0 = 0.2 + 0.1 * ((pos_idx - 3) as f64);
        (0.3, t0)
    }
}

fn decode_len5(idx: i32) -> f64 {
    // 0..4 -> 0.2..0.6
    0.2 + 0.1 * (idx as f64)
}

fn decode_trapezoid(abs_id: i32) -> EdgeSpec {
    // 1..175
    let z = abs_id - 1;
    let pos_idx = z % 7; // 0..6
    let t = z / 7; // 0..24
    let h0_idx = t % 5; // 0..4
    let h1_idx = (t / 5) % 5; // 0..4

    let (l, t0) = decode_pos_idx(pos_idx);
    let h0 = decode_len5(h0_idx);
    let h1 = decode_len5(h1_idx);

    EdgeSpec::Trapezoid { l, t0, h0, h1 }
}

fn decode_triangle(abs_id: i32) -> EdgeSpec {
    // 176..295 => 0..119
    let t = abs_id - 176;
    if t < 60 {
        // l=0.4 group: pos3 * a4 * h5 = 60
        let pos_idx = t % 3; // 0..2 -> t0 0.2..0.4
        let a_idx = (t / 3) % 4; // 0..3 -> a 0.0..0.3
        let h_idx = (t / (3 * 4)) % 5; // 0..4 -> h 0.2..0.6
        let l = 0.4;
        let t0 = 0.2 + 0.1 * (pos_idx as f64);
        let a = 0.1 * (a_idx as f64);
        let h = decode_len5(h_idx);
        EdgeSpec::Triangle { l, t0, a, h }
    } else {
        // l=0.3 group: pos4 * a3 * h5 = 60
        let t2 = t - 60;
        let pos_idx = t2 % 4; // 0..3 -> t0 0.2..0.5
        let a_idx = (t2 / 4) % 3; // 0..2 -> a 0.0..0.2
        let h_idx = (t2 / (4 * 3)) % 5; // 0..4
        let l = 0.3;
        let t0 = 0.2 + 0.1 * (pos_idx as f64);
        let a = 0.1 * (a_idx as f64);
        let h = decode_len5(h_idx);
        EdgeSpec::Triangle { l, t0, a, h }
    }
}

/// ---- Regular hexagon vertices ----
/// side length = 1 => circumradius = 1
pub fn base_hex_vertices() -> [Pt; 6] {
    let mut v = [Pt { x: 0.0, y: 0.0 }; 6];
    for k in 0..6 {
        let th = (k as f64) * PI / 3.0;
        v[k] = Pt {
            x: th.cos(),
            y: th.sin(),
        };
    }
    v
}

fn unit(v: Pt) -> Pt {
    let n = v.norm();
    if n == 0.0 {
        Pt { x: 0.0, y: 0.0 }
    } else {
        v.mul(1.0 / n)
    }
}

/// For CCW polygon, moving A->B, outward is to the right (CW normal)
fn outward_normal(a: Pt, b: Pt) -> Pt {
    let u = unit(b.sub(a));
    Pt { x: u.y, y: -u.x } // rotate CW
}

fn lerp(a: Pt, b: Pt, t: f64) -> Pt {
    a.add(b.sub(a).mul(t))
}

fn approx_eq(a: Pt, b: Pt, eps: f64) -> bool {
    (a.x - b.x).abs() <= eps && (a.y - b.y).abs() <= eps
}

fn push_pt(out: &mut Vec<Pt>, p: Pt) {
    const EPS: f64 = 1e-9;
    if out.is_empty() || !approx_eq(out[out.len() - 1], p, EPS) {
        out.push(p);
    }
}

/// Build the polyline segment replacing [A..B] with an attachment.
/// Returns:
/// - points to append (starting after A, ending at B) in traversal order
/// - for concave (indent) only: the inner boundary segments of the notch
fn edge_attachment_polyline(
    a: Pt,
    b: Pt,
    sign: i32, // + for凸, - for凹
    spec: EdgeSpec,
) -> Option<(Vec<Pt>, Option<Vec<Seg>>)> {
    if let EdgeSpec::None = spec {
        return Some((vec![b], None));
    }

    let u_dir = unit(b.sub(a));
    let n_out = outward_normal(a, b);
    let n = if sign > 0 { n_out } else { n_out.mul(-1.0) }; // inward for concave

    // base segment is on the original edge
    match spec {
        EdgeSpec::Trapezoid { l, t0, h0, h1 } => {
            // validate
            if t0 < 0.0 || t0 + l > 1.0 {
                return None;
            }
            if l <= 0.0 {
                return None;
            }

            let p0 = lerp(a, b, t0);
            let p1 = lerp(a, b, t0 + l);

            // "perpendicular legs" from p0,p1 along normal by h0,h1
            let q0 = p0.add(n.mul(h0));
            let q1 = p1.add(n.mul(h1));

            // Build polyline A -> ... -> B. Caller already has A; we return after-A points.
            // Sequence: ... p0 -> q0 -> q1 -> p1 ... then to B
            let mut pts = Vec::new();
            pts.push(p0);
            pts.push(q0);
            pts.push(q1);
            pts.push(p1);
            pts.push(b);

            // For concave notch, the "inner boundary" excludes the shared base segment [p0..p1]
            let inner = if sign < 0 {
                Some(vec![
                    Seg { a: p0, b: q0 },
                    Seg { a: q0, b: q1 },
                    Seg { a: q1, b: p1 },
                ])
            } else {
                None
            };

            Some((pts, inner))
        }
        EdgeSpec::Triangle { l, t0, a: aa, h } => {
            if t0 < 0.0 || t0 + l > 1.0 {
                return None;
            }
            if aa < 0.0 || aa > l {
                return None;
            }

            let p0 = lerp(a, b, t0);
            let p1 = lerp(a, b, t0 + l);
            let apex = p0.add(u_dir.mul(aa)).add(n.mul(h));

            let mut pts = Vec::new();
            pts.push(p0);
            pts.push(apex);
            pts.push(p1);
            pts.push(b);

            let inner = if sign < 0 {
                Some(vec![Seg { a: p0, b: apex }, Seg { a: apex, b: p1 }])
            } else {
                None
            };

            Some((pts, inner))
        }
        EdgeSpec::None => unreachable!(),
    }
}

/// ---- Self-intersection check for simple polygon ----
fn orient(a: Pt, b: Pt, c: Pt) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn on_segment(a: Pt, b: Pt, p: Pt, eps: f64) -> bool {
    let minx = a.x.min(b.x) - eps;
    let maxx = a.x.max(b.x) + eps;
    let miny = a.y.min(b.y) - eps;
    let maxy = a.y.max(b.y) + eps;
    p.x >= minx && p.x <= maxx && p.y >= miny && p.y <= maxy
}

fn seg_intersect(s1: Seg, s2: Seg) -> bool {
    const EPS: f64 = 1e-10;
    let a = s1.a;
    let b = s1.b;
    let c = s2.a;
    let d = s2.b;

    let o1 = orient(a, b, c);
    let o2 = orient(a, b, d);
    let o3 = orient(c, d, a);
    let o4 = orient(c, d, b);

    // general case
    if (o1 > EPS && o2 < -EPS || o1 < -EPS && o2 > EPS)
        && (o3 > EPS && o4 < -EPS || o3 < -EPS && o4 > EPS)
    {
        return true;
    }

    // collinear / touching cases
    if o1.abs() <= EPS && on_segment(a, b, c, EPS) {
        return true;
    }
    if o2.abs() <= EPS && on_segment(a, b, d, EPS) {
        return true;
    }
    if o3.abs() <= EPS && on_segment(c, d, a, EPS) {
        return true;
    }
    if o4.abs() <= EPS && on_segment(c, d, b, EPS) {
        return true;
    }
    false
}

fn poly_self_intersects(poly: &[Pt]) -> bool {
    // poly: closed but last point NOT repeated. We'll consider segments i->i+1 (mod n).
    let n = poly.len();
    if n < 4 {
        return false;
    }

    let segs: Vec<Seg> = (0..n)
        .map(|i| Seg {
            a: poly[i],
            b: poly[(i + 1) % n],
        })
        .collect();

    // Check all non-adjacent segment pairs
    for i in 0..n {
        for j in (i + 1)..n {
            // skip same or adjacent, and skip first-last adjacency
            if j == i {
                continue;
            }
            if j == i + 1 {
                continue;
            }
            if i == 0 && j == n - 1 {
                continue;
            }

            if seg_intersect(segs[i], segs[j]) {
                return true;
            }
        }
    }
    false
}

/// ---- Segment distance (for concave inner boundary separation) ----
fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

fn dist_pt_seg(p: Pt, s: Seg) -> f64 {
    let v = s.b.sub(s.a);
    let w = p.sub(s.a);
    let vv = v.norm2();
    if vv == 0.0 {
        return p.sub(s.a).norm();
    }
    let t = clamp(w.dot(v) / vv, 0.0, 1.0);
    let proj = s.a.add(v.mul(t));
    p.sub(proj).norm()
}

fn dist_seg_seg(s1: Seg, s2: Seg) -> f64 {
    if seg_intersect(s1, s2) {
        return 0.0;
    }
    let d1 = dist_pt_seg(s1.a, s2);
    let d2 = dist_pt_seg(s1.b, s2);
    let d3 = dist_pt_seg(s2.a, s1);
    let d4 = dist_pt_seg(s2.b, s1);
    d1.min(d2).min(d3).min(d4)
}

fn min_dist_between_segment_sets(a: &[Seg], b: &[Seg]) -> f64 {
    let mut best = f64::INFINITY;
    for &sa in a {
        for &sb in b {
            best = best.min(dist_seg_seg(sa, sb));
        }
    }
    best
}

/// ---- Build full piece polygon from 6 edge ids ----
/// edges[i] : i32
///  0 => flat
/// +k =>凸, -k =>凹, abs(k) in 1..=295
///
/// Validation:
/// - polygon is simple (no self intersection)
/// - for any pair of concave notches on this piece, min distance between their inner boundaries >= min_concave_gap
pub fn build_piece_geom_if_valid(
    edges: [i32; 6],
    min_concave_gap: f64, // e.g. 0.2
) -> Option<PieceGeom> {
    let v = base_hex_vertices();

    let mut outline: Vec<Pt> = Vec::new();
    let mut concave_inner: Vec<Vec<Seg>> = Vec::new();

    // Start at vertex 0
    push_pt(&mut outline, v[0]);

    for i in 0..6 {
        let a = v[i];
        let b = v[(i + 1) % 6];
        let e = edges[i];

        if e == 0 {
            // straight
            push_pt(&mut outline, b);
            continue;
        }

        let sign = if e > 0 { 1 } else { -1 };
        let abs_id = e.abs();
        let spec = decode_edge_spec(abs_id)?;
        let (pts, inner_opt) = edge_attachment_polyline(a, b, sign, spec)?;

        // pts starts with p0; caller already has a, so just append sequentially
        for p in pts {
            push_pt(&mut outline, p);
        }

        if let Some(inner) = inner_opt {
            concave_inner.push(inner);
        }
    }

    // Close by removing duplicated last==first if any and ensuring closure implicit
    if outline.len() >= 2 && approx_eq(outline[0], outline[outline.len() - 1], 1e-9) {
        outline.pop();
    }

    // Basic sanity
    if outline.len() < 6 {
        return None;
    }

    // Self-intersection => invalid
    if poly_self_intersects(&outline) {
        return None;
    }

    // Concave inner boundary distance constraint
    if concave_inner.len() >= 2 {
        for i in 0..concave_inner.len() {
            for j in (i + 1)..concave_inner.len() {
                let d = min_dist_between_segment_sets(&concave_inner[i], &concave_inner[j]);
                if d < min_concave_gap {
                    return None;
                }
            }
        }
    }

    Some(PieceGeom {
        outline,
        concave_inner_segments: concave_inner,
    })
}

/// ---- SVG output ----
/// scale: multiply coordinates. margin adds whitespace.
pub fn piece_geom_to_svg(
    geom: &PieceGeom,
    scale: f64,
    margin: f64,
    stroke: &str,
    fill: &str,
    stroke_width: f64,
) -> String {
    // Compute bbox
    let mut minx = f64::INFINITY;
    let mut miny = f64::INFINITY;
    let mut maxx = -f64::INFINITY;
    let mut maxy = -f64::INFINITY;

    for p in &geom.outline {
        minx = minx.min(p.x);
        miny = miny.min(p.y);
        maxx = maxx.max(p.x);
        maxy = maxy.max(p.y);
    }

    // Apply scale and margin
    let w = (maxx - minx) * scale + 2.0 * margin;
    let h = (maxy - miny) * scale + 2.0 * margin;

    // SVG y-axis is downward; we flip y to keep CCW visually upright (optional).
    // Here: flip y for nicer view.
    let mut d = String::new();
    for (k, p) in geom.outline.iter().enumerate() {
        let x = (p.x - minx) * scale + margin;
        let y = (maxy - p.y) * scale + margin; // flip
        if k == 0 {
            d.push_str(&format!("M {:.4} {:.4} ", x, y));
        } else {
            d.push_str(&format!("L {:.4} {:.4} ", x, y));
        }
    }
    d.push_str("Z");

    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w:.2}" height="{h:.2}" viewBox="0 0 {w:.2} {h:.2}">
  <path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.3}"/>
</svg>"#,
        w = w,
        h = h,
        d = d,
        fill = fill,
        stroke = stroke,
        sw = stroke_width
    )
}

/// Convenience: build svg if valid
pub fn build_piece_svg_if_valid(edges: [i32; 6]) -> Option<String> {
    let geom = build_piece_geom_if_valid(edges, 0.2)?;
    Some(piece_geom_to_svg(&geom, 200.0, 20.0, "#111", "none", 1.5))
}

use std::fs;

fn dump_svgs(pieces: &[Piece]) {
    fs::create_dir_all("out_svg").unwrap();

    for p in pieces {
        if let Some(svg) = build_piece_svg_if_valid(p.edges) {
            let path = format!(
                "out_svg/piece_{:04}_{}.svg",
                p.id,
                match p.kind {
                    PieceKind::True => "T",
                    PieceKind::False => "F",
                }
            );
            fs::write(path, svg).unwrap();
        } else {
            eprintln!("SKIP invalid geometry piece_id={}", p.id);
        }
    }
}

#[derive(Default)]
struct GeomMemo {
    // key: rotation-canonical edge pattern, value: geometry valid?
    ok: HashMap<[i32; 6], bool>,
    // 任意: メモリ抑制
    max_entries: usize,
}

impl GeomMemo {
    fn new(max_entries: usize) -> Self {
        Self {
            ok: HashMap::new(),
            max_entries,
        }
    }

    #[inline]
    fn geom_ok(&mut self, edges: [i32; 6], min_gap: f64) -> bool {
        let key = canonical_shape(&edges); // rotation-invariant
        if let Some(&v) = self.ok.get(&key) {
            return v;
        }

        // キャッシュが巨大化しすぎたら掃除（LRUではないが実用上効く）
        if self.ok.len() > self.max_entries {
            self.ok.clear();
        }

        // 代表として canonical key をチェックしてしまう（回転不変なのでOK）
        let v = build_piece_geom_if_valid(key, min_gap).is_some();
        self.ok.insert(key, v);
        v
    }
}

#[derive(Clone, Copy, Debug)]
struct EdgeMeta {
    max_h: f64,
}

fn precompute_edge_meta(total_alphabet: i32) -> Vec<EdgeMeta> {
    let n = total_alphabet.max(0) as usize;
    let mut meta = vec![EdgeMeta { max_h: 0.0 }; n + 1];
    for id in 1..=n {
        let abs_id = id as i32;
        let max_h = match decode_edge_spec(abs_id) {
            Some(EdgeSpec::Trapezoid { h0, h1, .. }) => h0.max(h1),
            Some(EdgeSpec::Triangle { h, .. }) => h,
            _ => 0.0,
        };
        meta[id] = EdgeMeta { max_h };
    }
    meta
}

fn piece_risk_score(edges: &[i32; 6], meta: &[EdgeMeta]) -> f64 {
    let mut concave = 0.0;
    let mut sum_h = 0.0;
    let mut nonzero = 0.0;

    for &e in edges {
        if e == 0 {
            continue;
        }
        nonzero += 1.0;
        if e < 0 {
            concave += 1.0;
        }
        let id = e.abs() as usize;
        if id < meta.len() {
            sum_h += meta[id].max_h;
        }
    }

    // 重みは好みで調整（凹と高さを強めに）
    concave * 5.0 + sum_h * 3.0 + nonzero * 0.5
}
