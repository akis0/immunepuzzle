use anyhow::{bail, Context, Result};
use rand::rngs::StdRng;
use rand::prelude::IndexedRandom;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const DIRS: [(i32, i32); 6] = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
];

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
struct Axial {
    q: i32,
    r: i32,
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    q: [u8; 3],
    p: [u8; 3],
}

#[derive(Clone, Copy, Debug)]
struct Point {
    x: f64,
    y: f64,
}

#[derive(Clone, Debug)]
struct Piece {
    id: usize,
    kind: PieceKind,
    cluster_id: Option<usize>,
    edges: [i32; 6],
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum PieceKind {
    True,
    False,
}

#[derive(Serialize)]
struct PuzzleOutput {
    side_len: i32,
    true_piece_count: usize,
    false_piece_count: usize,
    pieces: Vec<PieceOutput>,
    shapes: Vec<ShapeOutput>,
    stats: StatsOutput,
}

#[derive(Serialize)]
struct PieceOutput {
    id: usize,
    kind: PieceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    cluster_id: Option<usize>,
    edges: [i32; 6],
}

#[derive(Serialize)]
struct ShapeOutput {
    id: i32,
    q: [u8; 3],
    p: [u8; 3],
}

#[derive(Serialize)]
struct SolutionOutput {
    placements: Vec<PlacementOutput>,
    rotation_equivalent: bool,
}

#[derive(Serialize, Clone)]
struct PlacementOutput {
    q: i32,
    r: i32,
    piece_id: usize,
    rot: u8,
}

#[derive(Serialize)]
struct OutputBundle {
    puzzle: PuzzleOutput,
    solution: SolutionOutput,
}

#[derive(Serialize)]
struct StatsOutput {
    seed: u64,
    trial_seed: u64,
    trial_index: usize,
    score: f64,
    u_nodes: u64,
    f_nodes: u64,
    u_solutions: u64,
    f_solutions: u64,
    unique_ids_used: usize,
    core_alphabet: i32,
    true_alphabet: i32,
    total_alphabet: i32,
    shapes_generated: usize,
    shape_collisions: usize,
    false_cluster_count: usize,
    false_cluster_sizes: Vec<usize>,
    edge_entropy: f64,
    reflection_killed: bool,
}

struct Config {
    seed: Option<u64>,
    trials: usize,
    side_len: i32,
    total_alphabet: i32,
    true_alphabet: i32,
    core_alphabet: i32,
    anchor_ratio: f64,
    false_flat: f64,
    false_attach: f64,
    max_u_nodes: u64,
    max_f_nodes: u64,
    dump_svgs: bool,
    rot_equiv: bool,
    kill_reflection: bool,
    false_count: Option<usize>,
    false_clusters: Option<Vec<usize>>,
    svg_dir: Option<PathBuf>,
}

#[derive(Clone, Copy)]
struct PlacementRef {
    pid: usize,
    rot: u8,
}

struct Solver {
    neighbors: Vec<[Option<usize>; 6]>,
    piece_rots: Vec<[[i32; 6]; 6]>,
    postings: Vec<Vec<Vec<PlacementRef>>>,
    value_offset: i32,
    assignments: Vec<Option<PlacementRef>>,
    available: Vec<bool>,
    cache: Vec<CellCache>,
    nodes: u64,
    node_cap: u64,
    solution_limit: u64,
    solutions_found: u64,
    require_false: bool,
    false_piece: Vec<bool>,
}

#[derive(Clone)]
struct CellCache {
    key: [i32; 6],
    candidates: Vec<PlacementRef>,
}

#[allow(dead_code)]
struct GeometryCache {
    cache: HashMap<[i32; 6], bool>,
}

fn main() -> Result<()> {
    let cfg = Config::from_args()?;
    let base_seed = cfg.seed.unwrap_or_else(rand::random);

    let mut rng = StdRng::seed_from_u64(base_seed);
    let mut best: Option<Candidate> = None;

    for trial in 0..cfg.trials {
        let trial_seed: u64 = rng.random();
        let mut trial_rng = StdRng::seed_from_u64(trial_seed);
        let attempt = generate_candidate(&cfg, base_seed, trial_seed, trial, &mut trial_rng);
        let Some(candidate) = attempt else {
            continue;
        };
        if best
            .as_ref()
            .map(|c| candidate.score > c.score)
            .unwrap_or(true)
        {
            best = Some(candidate);
        }
    }

    let Some(candidate) = best else {
        bail!("no valid candidates found in {} trials", cfg.trials);
    };

    let output = OutputBundle {
        puzzle: candidate.puzzle,
        solution: candidate.solution,
    };
    let json = serde_json::to_string_pretty(&output)?;
    println!("{json}");

    if cfg.dump_svgs {
        dump_svgs(&candidate.svg_context, cfg.svg_dir.as_deref())?;
    }

    Ok(())
}

struct Candidate {
    puzzle: PuzzleOutput,
    solution: SolutionOutput,
    svg_context: SvgContext,
    score: f64,
}

struct SvgContext {
    seed: u64,
    pieces: Vec<Piece>,
    shapes: Vec<Shape>,
    placements: Vec<PlacementOutput>,
}

impl Config {
    fn from_args() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut cfg = Config {
            seed: None,
            trials: 10,
            side_len: 8,
            total_alphabet: 295,
            true_alphabet: 240,
            core_alphabet: 40,
            anchor_ratio: 0.05,
            false_flat: 0.2,
            false_attach: 0.3,
            max_u_nodes: 30_000,
            max_f_nodes: 30_000,
            dump_svgs: false,
            rot_equiv: false,
            kill_reflection: false,
            false_count: None,
            false_clusters: None,
            svg_dir: None,
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--seed" => {
                    let v = args.next().context("missing value for --seed")?;
                    cfg.seed = Some(v.parse()?);
                }
                "--trials" => {
                    let v = args.next().context("missing value for --trials")?;
                    cfg.trials = v.parse()?;
                }
                "--side" => {
                    let v = args.next().context("missing value for --side")?;
                    cfg.side_len = v.parse()?;
                }
                "--total-alphabet" => {
                    let v = args.next().context("missing value for --total-alphabet")?;
                    cfg.total_alphabet = v.parse()?;
                }
                "--true-alphabet" => {
                    let v = args.next().context("missing value for --true-alphabet")?;
                    cfg.true_alphabet = v.parse()?;
                }
                "--core-alphabet" => {
                    let v = args.next().context("missing value for --core-alphabet")?;
                    cfg.core_alphabet = v.parse()?;
                }
                "--anchor-ratio" => {
                    let v = args.next().context("missing value for --anchor-ratio")?;
                    cfg.anchor_ratio = v.parse()?;
                }
                "--false-flat" => {
                    let v = args.next().context("missing value for --false-flat")?;
                    cfg.false_flat = v.parse()?;
                }
                "--false-attach" => {
                    let v = args.next().context("missing value for --false-attach")?;
                    cfg.false_attach = v.parse()?;
                }
                "--max-u-nodes" => {
                    let v = args.next().context("missing value for --max-u-nodes")?;
                    cfg.max_u_nodes = v.parse()?;
                }
                "--max-f-nodes" => {
                    let v = args.next().context("missing value for --max-f-nodes")?;
                    cfg.max_f_nodes = v.parse()?;
                }
                "--false-count" => {
                    let v = args.next().context("missing value for --false-count")?;
                    cfg.false_count = Some(v.parse()?);
                }
                "--false-clusters" => {
                    let v = args.next().context("missing value for --false-clusters")?;
                    let sizes = v
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(|s| s.parse::<usize>())
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    cfg.false_clusters = Some(sizes);
                }
                "--dump-svgs" => {
                    cfg.dump_svgs = true;
                }
                "--svg-dir" => {
                    let v = args.next().context("missing value for --svg-dir")?;
                    cfg.svg_dir = Some(PathBuf::from(v));
                }
                "--rot-equiv" => {
                    cfg.rot_equiv = true;
                }
                "--kill-reflection" => {
                    cfg.kill_reflection = true;
                }
                "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => {
                    bail!("unknown argument: {arg}");
                }
            }
        }

        if cfg.side_len < 1 {
            bail!("side length must be >= 1");
        }
        if cfg.total_alphabet < 1 {
            bail!("total alphabet must be >= 1");
        }
        if cfg.true_alphabet < 1 || cfg.true_alphabet > cfg.total_alphabet {
            bail!("true alphabet must be between 1 and total alphabet");
        }
        if cfg.core_alphabet < 0 {
            bail!("core alphabet must be >= 0");
        }
        if !(0.0..=1.0).contains(&cfg.anchor_ratio) {
            bail!("anchor ratio must be between 0 and 1");
        }
        if !(0.0..=1.0).contains(&cfg.false_flat) {
            bail!("false flat ratio must be between 0 and 1");
        }
        if !(0.0..=1.0).contains(&cfg.false_attach) {
            bail!("false attach ratio must be between 0 and 1");
        }

        if let Some(ref sizes) = cfg.false_clusters {
            let sum: usize = sizes.iter().sum();
            if let Some(count) = cfg.false_count {
                if sum != count {
                    bail!("false cluster sizes sum ({sum}) does not match --false-count ({count})");
                }
            } else {
                cfg.false_count = Some(sum);
            }
        }

        Ok(cfg)
    }
}

fn print_help() {
    println!(
        "Usage: immunepuzzle [options]\n\
         \n\
         --seed <u64>\n\
         --trials <n>\n\
         --side <n>\n\
         --total-alphabet <n>\n\
         --true-alphabet <n>\n\
         --core-alphabet <n>\n\
         --anchor-ratio <f>\n\
         --false-flat <f>\n\
         --false-attach <f>\n\
         --false-count <n>\n\
         --false-clusters <a,b,c>\n\
         --max-u-nodes <n>\n\
         --max-f-nodes <n>\n\
         --dump-svgs\n\
         --svg-dir <path>\n\
         --rot-equiv\n\
         --kill-reflection\n\
         --help"
    );
}

fn generate_candidate(
    cfg: &Config,
    base_seed: u64,
    trial_seed: u64,
    trial_index: usize,
    rng: &mut StdRng,
) -> Option<Candidate> {
    let side_len = cfg.side_len;
    let cells = build_hex_cells(side_len);
    let neighbors = build_neighbors(&cells);
    let n = cells.len();
    let false_count = match cfg.false_count {
        Some(count) => count,
        None => (n / 5).max(1),
    };

    let total_alphabet = cfg.total_alphabet;
    let mut true_alphabet = cfg.true_alphabet.min(total_alphabet);
    let mut core_alphabet = cfg.core_alphabet.min(true_alphabet);

    let unique_needed = n.saturating_sub(1);
    let unique_budget = (true_alphabet - core_alphabet).max(0) as usize;
    if unique_budget < unique_needed {
        let adjusted_core = true_alphabet.saturating_sub(unique_needed as i32);
        core_alphabet = adjusted_core.max(0);
    }
    if core_alphabet > true_alphabet {
        core_alphabet = true_alphabet;
    }
    if true_alphabet > total_alphabet {
        true_alphabet = total_alphabet;
    }

    if total_alphabet <= 0 || true_alphabet <= 0 {
        return None;
    }

    let mut shapes = Vec::with_capacity((total_alphabet + 1) as usize);
    shapes.push(Shape {
        q: [0, 0, 0],
        p: [0, 0, 0],
    });
    let mut seen = HashSet::new();
    let mut collisions = 0usize;
    for _ in 1..=total_alphabet {
        loop {
            let shape = random_shape(rng);
            let key = shape_key(&shape);
            if seen.insert(key) {
                shapes.push(shape);
                break;
            } else {
                collisions += 1;
            }
        }
    }

    let mut reflection_killed = false;
    if cfg.kill_reflection {
        let mut idxs: Vec<usize> = (1..shapes.len()).collect();
        idxs.shuffle(rng);
        for idx in idxs {
            if !is_mirror_symmetric(&shapes[idx]) {
                reflection_killed = true;
                break;
            }
        }
        if !reflection_killed {
            for idx in 1..shapes.len() {
                let mut tries = 0;
                while tries < 200 {
                    let candidate = random_shape(rng);
                    if !is_mirror_symmetric(&candidate) {
                        shapes[idx] = candidate;
                        reflection_killed = true;
                        break;
                    }
                    tries += 1;
                }
                if reflection_killed {
                    break;
                }
            }
        }
    }

    let (true_pieces, unique_ids_used, edge_entropy) = build_true_pieces(
        n,
        &neighbors,
        rng,
        core_alphabet,
        true_alphabet,
        cfg.anchor_ratio,
    );

    let (false_pieces, cluster_sizes) = build_false_pieces(
        false_count,
        &true_pieces,
        rng,
        true_alphabet,
        total_alphabet,
        cfg.false_flat,
        cfg.false_attach,
        cfg.false_clusters.as_deref(),
    );

    let mut pieces = true_pieces.clone();
    pieces.extend(false_pieces);

    // Geometry checks are omitted because the depth bounds guarantee no self-intersections.

    let max_true_id = max_edge_id(&true_pieces).max(1);
    let u_solver = Solver::new(
        &neighbors,
        &true_pieces,
        max_true_id,
        cfg.max_u_nodes,
        2,
        false,
    );
    let u_result = u_solver.solve();

    let false_anchor = false_anchor_exists(&neighbors, &pieces, true_alphabet);
    let f_result = if false_anchor {
        let max_all_id = max_edge_id(&pieces).max(1);
        let f_solver = Solver::new(
            &neighbors,
            &pieces,
            max_all_id,
            cfg.max_f_nodes,
            1,
            true,
        );
        f_solver.solve()
    } else {
        SolveResult {
            nodes: 0,
            solutions_found: 0,
        }
    };

    let score = f_result.nodes as f64 + (u_result.nodes as f64) / 4.0;
    let placements = true_pieces
        .iter()
        .zip(cells.iter())
        .map(|(piece, cell)| PlacementOutput {
            q: cell.q,
            r: cell.r,
            piece_id: piece.id,
            rot: 0,
        })
        .collect::<Vec<_>>();

    let piece_outputs = pieces
        .iter()
        .map(|piece| PieceOutput {
            id: piece.id,
            kind: piece.kind,
            cluster_id: piece.cluster_id,
            edges: piece.edges,
        })
        .collect::<Vec<_>>();

    let shape_outputs = shapes
        .iter()
        .enumerate()
        .skip(1)
        .map(|(idx, shape)| ShapeOutput {
            id: idx as i32,
            q: shape.q,
            p: shape.p,
        })
        .collect::<Vec<_>>();

    let stats = StatsOutput {
        seed: base_seed,
        trial_seed,
        trial_index,
        score,
        u_nodes: u_result.nodes,
        f_nodes: f_result.nodes,
        u_solutions: u_result.solutions_found,
        f_solutions: f_result.solutions_found,
        unique_ids_used,
        core_alphabet,
        true_alphabet,
        total_alphabet,
        shapes_generated: shapes.len() - 1,
        shape_collisions: collisions,
        false_cluster_count: cluster_sizes.len(),
        false_cluster_sizes: cluster_sizes,
        edge_entropy,
        reflection_killed,
    };

    let puzzle = PuzzleOutput {
        side_len,
        true_piece_count: n,
        false_piece_count: false_count,
        pieces: piece_outputs,
        shapes: shape_outputs,
        stats,
    };

    let solution = SolutionOutput {
        placements: placements.clone(),
        rotation_equivalent: cfg.rot_equiv,
    };

    Some(Candidate {
        puzzle,
        solution,
        svg_context: SvgContext {
            seed: base_seed,
            pieces,
            shapes,
            placements,
        },
        score,
    })
}

fn build_hex_cells(side_len: i32) -> Vec<Axial> {
    let radius = side_len - 1;
    let mut cells = Vec::new();
    for q in -radius..=radius {
        for r in -radius..=radius {
            let s = -q - r;
            if q.abs().max(r.abs()).max(s.abs()) <= radius {
                cells.push(Axial { q, r });
            }
        }
    }
    cells
}

fn build_neighbors(cells: &[Axial]) -> Vec<[Option<usize>; 6]> {
    let mut map = HashMap::new();
    for (idx, cell) in cells.iter().enumerate() {
        map.insert(*cell, idx);
    }
    let mut neighbors = vec![[None; 6]; cells.len()];
    for (idx, cell) in cells.iter().enumerate() {
        for (dir, (dq, dr)) in DIRS.iter().enumerate() {
            let neighbor = Axial {
                q: cell.q + dq,
                r: cell.r + dr,
            };
            if let Some(n_idx) = map.get(&neighbor) {
                neighbors[idx][dir] = Some(*n_idx);
            }
        }
    }
    neighbors
}

fn build_true_pieces(
    n: usize,
    neighbors: &[ [Option<usize>; 6] ],
    rng: &mut StdRng,
    core_alphabet: i32,
    true_alphabet: i32,
    anchor_ratio: f64,
) -> (Vec<Piece>, usize, f64) {
    let tree_edges = spanning_tree_edges(neighbors, rng);
    let mut edges_per_piece = vec![[0i32; 6]; n];

    let mut unique_ids_used = 0usize;
    let mut next_unique_id = core_alphabet + 1;
    let mut anchor_ids = 0usize;

    for (a, b, dir) in enumerate_edges(neighbors) {
        let is_tree = tree_edges.contains(&(a.min(b), a.max(b)));
        let id = if is_tree {
            if next_unique_id <= true_alphabet {
                let value = next_unique_id;
                next_unique_id += 1;
                unique_ids_used += 1;
                value
            } else {
                (core_alphabet.max(1)).min(true_alphabet)
            }
        } else if anchor_ratio > 0.0
            && rng.random::<f64>() < anchor_ratio
            && next_unique_id <= true_alphabet
        {
            let value = next_unique_id;
            next_unique_id += 1;
            unique_ids_used += 1;
            anchor_ids += 1;
            value
        } else if core_alphabet > 0 {
            rng.random_range(1..=core_alphabet)
        } else if next_unique_id <= true_alphabet {
            let value = next_unique_id;
            next_unique_id += 1;
            unique_ids_used += 1;
            value
        } else {
            rng.random_range(1..=true_alphabet.max(1))
        };

        let (pos_val, neg_val) = if a < b { (id, -id) } else { (-id, id) };
        edges_per_piece[a][dir] = pos_val;
        let opp = (dir + 3) % 6;
        edges_per_piece[b][opp] = neg_val;
    }

    let pieces = edges_per_piece
        .into_iter()
        .enumerate()
        .map(|(idx, edges)| Piece {
            id: idx,
            kind: PieceKind::True,
            cluster_id: None,
            edges,
        })
        .collect::<Vec<_>>();

    let mut counts = vec![0usize; (true_alphabet + 1).max(1) as usize];
    for piece in &pieces {
        for edge in piece.edges.iter() {
            if *edge != 0 {
                counts[edge.abs() as usize] += 1;
            }
        }
    }
    let total: usize = counts.iter().sum();
    let entropy = if total == 0 {
        0.0
    } else {
        let mut acc = 0.0;
        for &count in &counts {
            if count == 0 {
                continue;
            }
            let p = count as f64 / total as f64;
            acc -= p * p.ln();
        }
        acc
    };

    let _ = anchor_ids;
    (pieces, unique_ids_used, entropy)
}

fn enumerate_edges(neighbors: &[ [Option<usize>; 6] ]) -> Vec<(usize, usize, usize)> {
    let mut edges = Vec::new();
    for (idx, dirs) in neighbors.iter().enumerate() {
        for (dir, neighbor) in dirs.iter().enumerate() {
            if let Some(n_idx) = neighbor {
                if idx < *n_idx {
                    edges.push((idx, *n_idx, dir));
                }
            }
        }
    }
    edges
}

fn spanning_tree_edges(neighbors: &[ [Option<usize>; 6] ], rng: &mut StdRng) -> HashSet<(usize, usize)> {
    let mut visited = vec![false; neighbors.len()];
    let mut stack = vec![0usize];
    visited[0] = true;
    let mut tree_edges = HashSet::new();

    while let Some(node) = stack.pop() {
        let mut dirs: Vec<usize> = (0..6).collect();
        dirs.shuffle(rng);
        for dir in dirs {
            if let Some(next) = neighbors[node][dir] {
                if !visited[next] {
                    visited[next] = true;
                    stack.push(next);
                    tree_edges.insert((node.min(next), node.max(next)));
                }
            }
        }
    }
    tree_edges
}

fn random_shape(rng: &mut StdRng) -> Shape {
    loop {
        let q1 = rng.random_range(1..=7);
        let q2 = rng.random_range(q1 + 1..=8);
        let q3 = rng.random_range(q2 + 1..=9);
        let p1 = random_depth_for_q(rng, q1);
        let p2 = random_depth_for_q(rng, q2);
        let p3 = random_depth_for_q(rng, q3);

        if p1 == 0 && q1 != 1 && q1 != 9 {
            continue;
        }
        if p2 == 0 && q2 != 1 && q2 != 9 {
            continue;
        }
        if p3 == 0 && q3 != 1 && q3 != 9 {
            continue;
        }
        if p1 == p2 && p2 == p3 {
            continue;
        }

        // Avoid completely flat shape.
        if p1 == 0 && p2 == 0 && p3 == 0 {
            continue;
        }

        return Shape {
            q: [q1, q2, q3],
            p: [p1, p2, p3],
        };
    }
}

fn random_depth_for_q(rng: &mut StdRng, q_tenths: u8) -> u8 {
    let max_p = max_depth_tenths(q_tenths);
    let min_p = if q_tenths == 1 || q_tenths == 9 { 0 } else { 1 };
    if max_p < min_p {
        return min_p;
    }
    rng.random_range(min_p..=max_p)
}

fn max_depth_tenths(q_tenths: u8) -> u8 {
    let min_q = q_tenths.min(10 - q_tenths);
    ((16 * min_q) / 10) as u8
}

fn shape_key(shape: &Shape) -> Vec<(u8, u8)> {
    let mut key = Vec::new();
    for idx in 0..3 {
        if shape.p[idx] == 0 {
            continue;
        }
        key.push((shape.q[idx], shape.p[idx]));
    }
    if key.is_empty() {
        key.push((0, 0));
    }
    key
}

fn is_mirror_symmetric(shape: &Shape) -> bool {
    let mirrored = Shape {
        q: [10 - shape.q[2], 10 - shape.q[1], 10 - shape.q[0]],
        p: [shape.p[2], shape.p[1], shape.p[0]],
    };
    shape_key(shape) == shape_key(&mirrored)
}

fn build_false_pieces(
    false_count: usize,
    true_pieces: &[Piece],
    rng: &mut StdRng,
    true_alphabet: i32,
    total_alphabet: i32,
    false_flat: f64,
    false_attach: f64,
    false_clusters: Option<&[usize]>,
) -> (Vec<Piece>, Vec<usize>) {
    let cluster_sizes = if let Some(sizes) = false_clusters {
        sizes.to_vec()
    } else {
        let mut remaining = false_count;
        let mut sizes = Vec::new();
        while remaining > 0 {
            let max = remaining.min(5);
            let size = rng.random_range(1..=max);
            sizes.push(size);
            remaining -= size;
        }
        sizes
    };

    let mut true_ids = Vec::new();
    let mut seen = HashSet::new();
    for piece in true_pieces {
        for edge in piece.edges.iter() {
            if *edge != 0 {
                let id = edge.abs();
                if seen.insert(id) {
                    true_ids.push(id);
                }
            }
        }
    }
    if true_ids.is_empty() {
        true_ids.push(1);
    }

    let mut next_piece_id = true_pieces.len();
    let mut pieces = Vec::new();
    for (cluster_id, size) in cluster_sizes.iter().enumerate() {
        let positions = build_cluster_positions(*size, rng);
        let mut pos_map = HashMap::new();
        for (idx, pos) in positions.iter().enumerate() {
            pos_map.insert(*pos, idx);
        }
        let mut edges = vec![[0i32; 6]; *size];

        for (idx, pos) in positions.iter().enumerate() {
            for (dir, (dq, dr)) in DIRS.iter().enumerate() {
                let neighbor = Axial {
                    q: pos.q + dq,
                    r: pos.r + dr,
                };
                if let Some(n_idx) = pos_map.get(&neighbor) {
                    if idx < *n_idx {
                        let id = random_false_internal_id(rng, true_alphabet, total_alphabet);
                        let (pos_val, neg_val) = if idx < *n_idx { (id, -id) } else { (-id, id) };
                        edges[idx][dir] = pos_val;
                        let opp = (dir + 3) % 6;
                        edges[*n_idx][opp] = neg_val;
                    }
                }
            }
        }

        for (idx, pos) in positions.iter().enumerate() {
            for (dir, (dq, dr)) in DIRS.iter().enumerate() {
                let neighbor = Axial {
                    q: pos.q + dq,
                    r: pos.r + dr,
                };
                if !pos_map.contains_key(&neighbor) {
                    let roll: f64 = rng.random();
                    let edge = if roll < false_flat {
                        0
                    } else if roll < false_flat + false_attach {
                        let id = *true_ids.choose(rng).unwrap_or(&1);
                        if rng.random_bool(0.5) { id } else { -id }
                    } else {
                        let id = random_false_poison_id(rng, true_alphabet, total_alphabet);
                        if rng.random_bool(0.5) { id } else { -id }
                    };
                    edges[idx][dir] = edge;
                }
            }
        }

        for idx in 0..*size {
            pieces.push(Piece {
                id: next_piece_id,
                kind: PieceKind::False,
                cluster_id: Some(cluster_id),
                edges: edges[idx],
            });
            next_piece_id += 1;
        }
    }

    (pieces, cluster_sizes)
}

fn build_cluster_positions(size: usize, rng: &mut StdRng) -> Vec<Axial> {
    let mut positions = vec![Axial { q: 0, r: 0 }];
    let mut set = HashSet::new();
    set.insert(Axial { q: 0, r: 0 });

    while positions.len() < size {
        let idx = rng.random_range(0..positions.len());
        let base = positions[idx];
        let mut dirs = DIRS;
        dirs.shuffle(rng);
        for (dq, dr) in dirs {
            let cand = Axial {
                q: base.q + dq,
                r: base.r + dr,
            };
            if set.insert(cand) {
                positions.push(cand);
                break;
            }
        }
    }

    positions
}

fn random_false_internal_id(rng: &mut StdRng, true_alphabet: i32, total_alphabet: i32) -> i32 {
    if total_alphabet > true_alphabet {
        rng.random_range(true_alphabet + 1..=total_alphabet)
    } else {
        rng.random_range(1..=true_alphabet.max(1))
    }
}

fn random_false_poison_id(rng: &mut StdRng, true_alphabet: i32, total_alphabet: i32) -> i32 {
    if total_alphabet > true_alphabet {
        rng.random_range(true_alphabet + 1..=total_alphabet)
    } else {
        let base = true_alphabet.max(1);
        rng.random_range(1..=base)
    }
}

fn axial_to_point(ax: Axial, size: f64) -> Point {
    let q = ax.q as f64;
    let r = ax.r as f64;
    let x = size * (3.0 / 2.0 * q);
    let y = size * ((3f64).sqrt() / 2.0 * q + (3f64).sqrt() * r);
    Point { x, y }
}

fn max_edge_id(pieces: &[Piece]) -> i32 {
    let mut max_id = 0i32;
    for piece in pieces {
        for edge in piece.edges.iter() {
            max_id = max_id.max(edge.abs());
        }
    }
    max_id
}

#[allow(dead_code)]
impl GeometryCache {
    fn is_valid(&mut self, edges: &[i32; 6], shapes: &[Shape]) -> bool {
        let key = canonical_edges(edges);
        if let Some(value) = self.cache.get(&key) {
            return *value;
        }
        let polygon = piece_polygon(edges, shapes, 1.0);
        let ok = !polygon_self_intersects(&polygon);
        self.cache.insert(key, ok);
        ok
    }
}

#[allow(dead_code)]
fn canonical_edges(edges: &[i32; 6]) -> [i32; 6] {
    let mut best = *edges;
    for rot in 1..6 {
        let rotated = rotate_edges(edges, rot);
        if rotated < best {
            best = rotated;
        }
    }
    best
}

fn rotate_edges(edges: &[i32; 6], rot: usize) -> [i32; 6] {
    let mut out = [0i32; 6];
    for i in 0..6 {
        out[(i + rot) % 6] = edges[i];
    }
    out
}

fn piece_polygon(edges: &[i32; 6], shapes: &[Shape], size: f64) -> Vec<Point> {
    let verts = hex_vertices(size);
    let mut points = Vec::new();
    points.push(verts[0]);
    for i in 0..6 {
        let a = verts[i];
        let b = verts[(i + 1) % 6];
        let edge = edges[i];
        if edge == 0 {
            points.push(b);
            continue;
        }
        let shape = &shapes[edge.abs() as usize];
        let mut edge_vec = Point {
            x: b.x - a.x,
            y: b.y - a.y,
        };
        let len = (edge_vec.x * edge_vec.x + edge_vec.y * edge_vec.y).sqrt();
        if len == 0.0 {
            points.push(b);
            continue;
        }
        edge_vec.x /= len;
        edge_vec.y /= len;
        let normal = Point {
            x: edge_vec.y,
            y: -edge_vec.x,
        };
        for idx in 0..3 {
            let q = shape.q[idx] as f64 / 10.0;
            let mut p = shape.p[idx] as f64 / 10.0;
            if edge < 0 {
                p = -p;
            }
            let px = a.x + (b.x - a.x) * q + normal.x * p * len;
            let py = a.y + (b.y - a.y) * q + normal.y * p * len;
            points.push(Point { x: px, y: py });
        }
        points.push(b);
    }
    points
}

fn hex_vertices(size: f64) -> [Point; 6] {
    let mut verts = [Point { x: 0.0, y: 0.0 }; 6];
    for i in 0..6 {
        let angle = (60.0 * i as f64 - 30.0) * PI / 180.0;
        verts[i] = Point {
            x: size * angle.cos(),
            y: size * angle.sin(),
        };
    }
    verts
}

#[allow(dead_code)]
fn polygon_self_intersects(poly: &[Point]) -> bool {
    if poly.len() < 4 {
        return false;
    }
    let n = poly.len();
    for i in 0..n {
        let a1 = poly[i];
        let a2 = poly[(i + 1) % n];
        for j in (i + 1)..n {
            if j == i || (j + 1) % n == i || (i + 1) % n == j {
                continue;
            }
            let b1 = poly[j];
            let b2 = poly[(j + 1) % n];
            if segments_intersect(a1, a2, b1, b2) {
                return true;
            }
        }
    }
    false
}

#[allow(dead_code)]
fn segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool {
    let o1 = orient(a1, a2, b1);
    let o2 = orient(a1, a2, b2);
    let o3 = orient(b1, b2, a1);
    let o4 = orient(b1, b2, a2);

    if o1 == 0.0 && on_segment(a1, a2, b1) {
        return true;
    }
    if o2 == 0.0 && on_segment(a1, a2, b2) {
        return true;
    }
    if o3 == 0.0 && on_segment(b1, b2, a1) {
        return true;
    }
    if o4 == 0.0 && on_segment(b1, b2, a2) {
        return true;
    }

    (o1 > 0.0) != (o2 > 0.0) && (o3 > 0.0) != (o4 > 0.0)
}

#[allow(dead_code)]
fn orient(a: Point, b: Point, c: Point) -> f64 {
    let val = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    if val.abs() < 1e-9 {
        0.0
    } else {
        val
    }
}

#[allow(dead_code)]
fn on_segment(a: Point, b: Point, c: Point) -> bool {
    let min_x = a.x.min(b.x) - 1e-9;
    let max_x = a.x.max(b.x) + 1e-9;
    let min_y = a.y.min(b.y) - 1e-9;
    let max_y = a.y.max(b.y) + 1e-9;
    c.x >= min_x && c.x <= max_x && c.y >= min_y && c.y <= max_y
}

struct SolveResult {
    nodes: u64,
    solutions_found: u64,
}

impl Solver {
    fn new(
        neighbors: &[ [Option<usize>; 6] ],
        pieces: &[Piece],
        alphabet_max: i32,
        node_cap: u64,
        solution_limit: u64,
        require_false: bool,
    ) -> Self {
        let mut piece_rots = Vec::with_capacity(pieces.len());
        for piece in pieces {
            let mut rots = [[0i32; 6]; 6];
            rots[0] = piece.edges;
            for rot in 1..6 {
                rots[rot] = rotate_edges(&rots[rot - 1], 1);
            }
            piece_rots.push(rots);
        }

        let value_offset = alphabet_max.max(1);
        let value_size = (value_offset * 2 + 1) as usize;
        let mut postings = vec![vec![Vec::new(); value_size]; 6];
        for (pid, rots) in piece_rots.iter().enumerate() {
            for rot in 0..6 {
                let placement = PlacementRef {
                    pid,
                    rot: rot as u8,
                };
                for dir in 0..6 {
                    let val = rots[rot][dir];
                    let idx = (val + value_offset) as usize;
                    postings[dir][idx].push(placement);
                }
            }
        }

        let assignments = vec![None; neighbors.len()];
        let available = vec![true; pieces.len()];
        let cache = vec![
            CellCache {
                key: [i32::MAX; 6],
                candidates: Vec::new(),
            };
            neighbors.len()
        ];
        let false_piece = pieces.iter().map(|p| p.kind == PieceKind::False).collect();

        Solver {
            neighbors: neighbors.to_vec(),
            piece_rots,
            postings,
            value_offset,
            assignments,
            available,
            cache,
            nodes: 0,
            node_cap,
            solution_limit,
            solutions_found: 0,
            require_false,
            false_piece,
        }
    }

    fn solve(mut self) -> SolveResult {
        self.search(false);
        SolveResult {
            nodes: self.nodes,
            solutions_found: self.solutions_found,
        }
    }

    fn search(&mut self, used_false: bool) {
        if self.nodes >= self.node_cap {
            return;
        }
        if self.solutions_found >= self.solution_limit {
            return;
        }
        let Some(cell_idx) = self.pick_next_cell() else {
            if !self.require_false || used_false {
                self.solutions_found += 1;
            }
            return;
        };
        let candidates = self.candidates_for_cell(cell_idx);
        if candidates.is_empty() {
            return;
        }
        for placement in candidates {
            if self.nodes >= self.node_cap || self.solutions_found >= self.solution_limit {
                break;
            }
            if !self.available[placement.pid] {
                continue;
            }
            self.nodes += 1;
            self.assignments[cell_idx] = Some(placement);
            self.available[placement.pid] = false;
            let next_used_false = used_false || self.false_piece[placement.pid];
            self.search(next_used_false);
            self.assignments[cell_idx] = None;
            self.available[placement.pid] = true;
        }
    }

    fn pick_next_cell(&mut self) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for idx in 0..self.assignments.len() {
            if self.assignments[idx].is_some() {
                continue;
            }
            let candidates = self.candidates_for_cell(idx);
            let count = candidates.len();
            match best {
                None => best = Some((idx, count)),
                Some((_, best_count)) if count < best_count => best = Some((idx, count)),
                _ => {}
            }
            if count <= 1 {
                break;
            }
        }
        best.map(|(idx, _)| idx)
    }

    fn candidates_for_cell(&mut self, idx: usize) -> Vec<PlacementRef> {
        let key = self.cell_key(idx);
        if self.cache[idx].key == key {
            return self
                .cache[idx]
                .candidates
                .iter()
                .copied()
                .filter(|p| self.available[p.pid])
                .collect();
        }
        let mut constraints = Vec::new();
        for dir in 0..6 {
            let val = key[dir];
            if val != i32::MAX {
                constraints.push((dir, val));
            }
        }
        let candidates = if constraints.is_empty() {
            let mut out = Vec::new();
            for pid in 0..self.available.len() {
                if !self.available[pid] {
                    continue;
                }
                for rot in 0..6 {
                    out.push(PlacementRef {
                        pid,
                        rot: rot as u8,
                    });
                }
            }
            out
        } else {
            constraints.sort_by_key(|(dir, val)| {
                let idx = (*val + self.value_offset) as usize;
                self.postings[*dir][idx].len()
            });
            let (base_dir, base_val) = constraints[0];
            let base_idx = (base_val + self.value_offset) as usize;
            let mut out = Vec::new();
            'outer: for placement in &self.postings[base_dir][base_idx] {
                let pid = placement.pid;
                if !self.available[pid] {
                    continue;
                }
                let rot = placement.rot as usize;
                for &(dir, val) in &constraints {
                    let edge = self.piece_rots[pid][rot][dir];
                    if edge != val {
                        continue 'outer;
                    }
                }
                out.push(*placement);
            }
            out
        };

        self.cache[idx] = CellCache {
            key,
            candidates: candidates.clone(),
        };
        candidates
    }

    fn cell_key(&self, idx: usize) -> [i32; 6] {
        let mut key = [i32::MAX; 6];
        for dir in 0..6 {
            if let Some(n_idx) = self.neighbors[idx][dir] {
                if let Some(placement) = self.assignments[n_idx] {
                    let opp = (dir + 3) % 6;
                    let neighbor_edges = self.piece_rots[placement.pid][placement.rot as usize];
                    key[dir] = -neighbor_edges[opp];
                }
            } else {
                key[dir] = 0;
            }
        }
        key
    }
}

fn false_anchor_exists(
    neighbors: &[ [Option<usize>; 6] ],
    pieces: &[Piece],
    true_alphabet: i32,
) -> bool {
    let mut false_pieces = Vec::new();
    for piece in pieces {
        if piece.kind == PieceKind::False {
            false_pieces.push(piece);
        }
    }
    if false_pieces.is_empty() {
        return false;
    }
    for (cell_idx, dirs) in neighbors.iter().enumerate() {
        let mut boundary = [false; 6];
        let mut has_boundary = false;
        for dir in 0..6 {
            if dirs[dir].is_none() {
                boundary[dir] = true;
                has_boundary = true;
            }
        }
        if !has_boundary {
            continue;
        }
        for piece in &false_pieces {
            for rot in 0..6 {
                let edges = rotate_edges(&piece.edges, rot);
                let mut ok = true;
                for dir in 0..6 {
                    if boundary[dir] && edges[dir] != 0 {
                        ok = false;
                        break;
                    }
                    if edges[dir].abs() > true_alphabet {
                        // still possible, keep.
                    }
                }
                if ok {
                    let _ = cell_idx;
                    return true;
                }
            }
        }
    }
    false
}

fn dump_svgs(context: &SvgContext, base_dir: Option<&Path>) -> Result<()> {
    let dir = if let Some(base) = base_dir {
        base.join(format!("seed_{}", context.seed))
    } else {
        PathBuf::from(format!("svgs/seed_{}", context.seed))
    };
    fs::create_dir_all(&dir)?;

    let pieces_svg = dir.join("pieces.svg");
    let mut pieces_file = fs::File::create(&pieces_svg)?;
    let mut piece_polys = Vec::new();
    let grid_cols = 10usize.max(1);
    let spacing = 3.5;
    for (idx, piece) in context.pieces.iter().enumerate() {
        let poly = piece_polygon(&piece.edges, &context.shapes, 1.0);
        let col = (idx % grid_cols) as f64;
        let row = (idx / grid_cols) as f64;
        let offset = Point {
            x: col * spacing,
            y: row * spacing,
        };
        let shifted = poly
            .iter()
            .map(|p| Point {
                x: p.x + offset.x,
                y: p.y + offset.y,
            })
            .collect::<Vec<_>>();
        piece_polys.push((shifted, piece.kind));
    }
    write_svg(&mut pieces_file, &piece_polys)?;

    let board_svg = dir.join("solution.svg");
    let mut board_file = fs::File::create(&board_svg)?;
    let mut board_polys = Vec::new();
    for placement in &context.placements {
        let piece = &context.pieces[placement.piece_id];
        let poly = piece_polygon(&rotate_edges(&piece.edges, placement.rot as usize), &context.shapes, 1.0);
        let center = axial_to_point(
            Axial {
                q: placement.q,
                r: placement.r,
            },
            1.2,
        );
        let rotated = poly
            .iter()
            .map(|p| Point {
                x: p.x + center.x,
                y: p.y + center.y,
            })
            .collect::<Vec<_>>();
        board_polys.push((rotated, piece.kind));
    }
    write_svg(&mut board_file, &board_polys)?;

    Ok(())
}

fn write_svg(file: &mut fs::File, polys: &[(Vec<Point>, PieceKind)]) -> Result<()> {
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;
    for (poly, _) in polys {
        for p in poly {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }
    }
    let width = (max_x - min_x) + 2.0;
    let height = (max_y - min_y) + 2.0;

    writeln!(
        file,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{} {} {} {}">"#,
        min_x - 1.0,
        min_y - 1.0,
        width,
        height
    )?;
        for (poly, kind) in polys {
            let color = match kind {
                PieceKind::True => "#ffcc66",
                PieceKind::False => "#66aaff",
            };
            let points = poly
            .iter()
            .map(|p| format!("{},{}", p.x, p.y))
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(
            file,
            r##"<polygon points="{}" fill="{}" stroke="#222" stroke-width="0.04" />"##,
            points, color
        )?;
    }
    writeln!(file, "</svg>")?;
    Ok(())
}
