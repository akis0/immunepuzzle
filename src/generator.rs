use anyhow::{bail, Result};
use rand::prelude::IndexedRandom;
use rand::rngs::StdRng;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::geometry::{build_piece_geom_if_valid, EdgeMeta, GeomMemo};
use crate::grid::{opposite_dir, rotate_edges, Axial, Board, DIRS};

#[derive(Clone, Debug, Serialize)]
pub struct Piece {
    pub id: usize,
    pub edges: [i32; 6],
    pub is_true: bool,
    pub cluster_id: Option<usize>,
    pub back_id: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SolutionEntry {
    pub q: i32,
    pub r: i32,
    pub piece_id: usize,
    pub rot: usize,
}

#[derive(Clone, Debug)]
pub struct GeneratedPuzzle {
    pub pieces: Vec<Piece>,
    pub solution: Vec<SolutionEntry>,
    pub seed: u64,
    pub retries: usize,
}

#[derive(Clone, Debug)]
pub struct GeneratorConfig {
    pub total_alphabet: i32,
    pub core_alphabet: i32,
    pub anchor_ratio: f64,
    pub false_flat_ratio: f64,
    pub false_attach_ratio: f64,
    pub false_clusters: Vec<(usize, usize)>,
    pub min_gap: f64,
    pub chiral: bool,
    pub back_id: bool,
    pub max_retries: usize,
}

pub fn generate_puzzle(
    board: &Board,
    cfg: &GeneratorConfig,
    memo: &mut GeomMemo,
    seed: u64,
) -> Result<GeneratedPuzzle> {
    let mut rng = StdRng::seed_from_u64(seed);
    let (true_pieces, solution, used_ids, repair_steps) = assign_true_pieces(board, cfg, &mut rng, memo)?;
    let mut pieces = true_pieces;
    let start_id = pieces.len();
    let decoys = generate_decoy_clusters(board, cfg, &mut rng, memo, &used_ids, start_id)?;
    pieces.extend(decoys);

    Ok(GeneratedPuzzle {
        pieces,
        solution,
        seed,
        retries: repair_steps,
    })
}

fn assign_true_pieces(
    board: &Board,
    cfg: &GeneratorConfig,
    rng: &mut impl Rng,
    memo: &mut GeomMemo,
) -> Result<(Vec<Piece>, Vec<SolutionEntry>, HashSet<i32>, usize)> {
    if cfg.core_alphabet < 1 {
        bail!("core_alphabet must be >= 1");
    }
    if cfg.total_alphabet < cfg.core_alphabet {
        bail!("total_alphabet must be >= core_alphabet");
    }

    let n = board.cell_count();
    let anchor_target = if n > 1 {
        let target = ((n - 1) as f64 * cfg.anchor_ratio).round() as usize;
        target.max(1).min(n - 1)
    } else {
        0
    };
    let max_anchor = (cfg.total_alphabet - cfg.core_alphabet) as usize;
    if anchor_target > max_anchor {
        bail!(
            "anchor_target {} exceeds total_alphabet capacity {}",
            anchor_target,
            max_anchor
        );
    }

    let sorted_ids = sorted_ids_by_height(cfg.total_alphabet, &memo.edge_meta);
    let core_count = cfg.core_alphabet as usize;
    if core_count == 0 || core_count > sorted_ids.len() {
        bail!("core_alphabet must be within 1..=total_alphabet");
    }
    let core_ids: Vec<i32> = sorted_ids[..core_count].to_vec();
    let core_set: HashSet<i32> = core_ids.iter().cloned().collect();
    let remaining_ids: Vec<i32> = sorted_ids[core_count..].to_vec();
    if anchor_target > remaining_ids.len() {
        bail!("not enough ids available for anchors");
    }
    let anchor_ids: Vec<i32> = remaining_ids[..anchor_target].to_vec();
    let chiral_ids: Vec<i32> = remaining_ids[anchor_target..]
        .iter()
        .cloned()
        .take(2)
        .collect();

    let tree_edges = random_spanning_tree(board, rng);
    let mut anchor_edges = tree_edges.clone();
    anchor_edges.shuffle(rng);
    anchor_edges.truncate(anchor_target);
    let mut anchor_edge_set = HashSet::new();
    for (a, b, _) in &anchor_edges {
        anchor_edge_set.insert(edge_key(*a, *b));
    }

    let mut abs_edges = vec![[0; 6]; n];
    let mut next_anchor = 0usize;
    for (a, b, dir) in &tree_edges {
        if anchor_edge_set.contains(&edge_key(*a, *b)) {
            let id = anchor_ids[next_anchor];
            next_anchor += 1;
            set_abs_edge(&mut abs_edges, *a, *b, *dir, id);
        }
    }

    for (a, b, dir) in adjacency_edges(board) {
        if abs_edges[a][dir] != 0 {
            continue;
        }
        let id = *core_ids.choose(rng).unwrap();
        set_abs_edge(&mut abs_edges, a, b, dir, id);
    }

    if cfg.chiral && !chiral_ids.is_empty() {
        let mut candidates: Vec<(usize, usize, usize)> = adjacency_edges(board)
            .into_iter()
            .filter(|(a, b, _)| !anchor_edge_set.contains(&edge_key(*a, *b)))
            .collect();
        candidates.shuffle(rng);
        for (id, (a, b, dir)) in chiral_ids.into_iter().zip(candidates.into_iter()) {
            set_abs_edge(&mut abs_edges, a, b, dir, id);
        }
    }

    let targets = target_concaves(board);
    let mut edges = vec![[0; 6]; n];
    let mut internal_edges = adjacency_edges(board);
    internal_edges.shuffle(rng);
    for (a, b, dir) in internal_edges {
        let id = abs_edges[a][dir];
        let opp = opposite_dir(dir);

        edges[a][dir] = -id;
        edges[b][opp] = id;
        let penalty_a = cell_penalty(&edges[a], targets[a], &memo.edge_meta);
        let penalty_b = cell_penalty(&edges[b], targets[b], &memo.edge_meta);
        let pen1 = penalty_a + penalty_b;

        edges[a][dir] = id;
        edges[b][opp] = -id;
        let penalty_a2 = cell_penalty(&edges[a], targets[a], &memo.edge_meta);
        let penalty_b2 = cell_penalty(&edges[b], targets[b], &memo.edge_meta);
        let pen2 = penalty_a2 + penalty_b2;

        let concave_on_a = if pen1 < pen2 {
            true
        } else if pen2 < pen1 {
            false
        } else {
            rng.random_bool(0.5)
        };
        if concave_on_a {
            edges[a][dir] = -id;
            edges[b][opp] = id;
        } else {
            edges[a][dir] = id;
            edges[b][opp] = -id;
        }
    }

    let repair_steps = repair_geometry(board, &mut edges, cfg, rng, memo, &core_ids, &core_set)?;

    let mut used_ids = HashSet::new();
    for cell_edges in &edges {
        for &e in cell_edges {
            if e != 0 {
                used_ids.insert(e.abs());
            }
        }
    }

    let mut pieces = Vec::with_capacity(n);
    let mut solution = Vec::with_capacity(n);
    for (idx, cell) in board.cells.iter().enumerate() {
        let rot = rng.random_range(0..6);
        let rotated = rotate_edges(&edges[idx], rot);
        if build_piece_geom_if_valid(&rotated, cfg.min_gap, memo).is_none() {
            bail!("geometry invalid after repair");
        }
        pieces.push(Piece {
            id: idx,
            edges: rotated,
            is_true: true,
            cluster_id: None,
            back_id: if cfg.back_id { Some(rng.random()) } else { None },
        });
        solution.push(SolutionEntry {
            q: cell.q,
            r: cell.r,
            piece_id: idx,
            rot: (6 - rot) % 6,
        });
    }

    Ok((pieces, solution, used_ids, repair_steps))
}

fn sorted_ids_by_height(total_alphabet: i32, meta: &[EdgeMeta]) -> Vec<i32> {
    let mut ids: Vec<i32> = (1..=total_alphabet).collect();
    ids.sort_by(|a, b| {
        let ha = meta[*a as usize].max_height;
        let hb = meta[*b as usize].max_height;
        ha.partial_cmp(&hb)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(b))
    });
    ids
}

fn core_id_pool(cfg: &GeneratorConfig, memo: &GeomMemo) -> Result<Vec<i32>> {
    let sorted = sorted_ids_by_height(cfg.total_alphabet, &memo.edge_meta);
    let core_count = cfg.core_alphabet as usize;
    if core_count == 0 || core_count > sorted.len() {
        bail!("core_alphabet must be within 1..=total_alphabet");
    }
    Ok(sorted[..core_count].to_vec())
}

fn set_abs_edge(abs_edges: &mut [[i32; 6]], a: usize, b: usize, dir: usize, id: i32) {
    abs_edges[a][dir] = id;
    abs_edges[b][opposite_dir(dir)] = id;
}

fn target_concaves(board: &Board) -> Vec<usize> {
    let mut targets = Vec::with_capacity(board.cell_count());
    for cell in 0..board.cell_count() {
        let deg = board.neighbors[cell].iter().filter(|n| n.is_some()).count();
        targets.push(deg / 2);
    }
    targets
}

fn cell_penalty(edges: &[i32; 6], target: usize, meta: &[EdgeMeta]) -> f64 {
    let mut concave = [false; 6];
    let mut concave_count = 0usize;
    let mut concave_height_sum = 0.0;
    for dir in 0..6 {
        if edges[dir] < 0 {
            concave[dir] = true;
            concave_count += 1;
            concave_height_sum += meta[edges[dir].abs() as usize].max_height;
        }
    }

    let mut adjacent = 0usize;
    for dir in 0..6 {
        if concave[dir] && concave[(dir + 1) % 6] {
            adjacent += 1;
        }
    }

    let count_diff = (concave_count as i32 - target as i32).abs() as f64;
    (adjacent as f64) * 80.0 + count_diff * 12.0 + concave_height_sum * 6.0
}

fn is_piece_valid(edges: &[i32; 6], cfg: &GeneratorConfig, memo: &mut GeomMemo) -> bool {
    build_piece_geom_if_valid(edges, cfg.min_gap, memo).is_some()
}

fn repair_geometry(
    board: &Board,
    edges: &mut [[i32; 6]],
    cfg: &GeneratorConfig,
    rng: &mut impl Rng,
    memo: &mut GeomMemo,
    core_ids: &[i32],
    core_set: &HashSet<i32>,
) -> Result<usize> {
    let n = board.cell_count();
    let targets = target_concaves(board);

    let mut valid = vec![false; n];
    let mut invalid_cells = Vec::new();
    for cell in 0..n {
        let ok = is_piece_valid(&edges[cell], cfg, memo);
        valid[cell] = ok;
        if !ok {
            invalid_cells.push(cell);
        }
    }
    let mut invalid_count = invalid_cells.len();

    struct BestMove {
        cell: usize,
        nb: usize,
        dir: usize,
        new_a: i32,
        new_b: i32,
        new_valid_cell: bool,
        new_valid_nb: bool,
        delta: i32,
        score: f64,
    }

    let mut steps = 0usize;
    let mut stagnation = 0usize;

    while invalid_count > 0 && steps < cfg.max_retries {
        steps += 1;

        let cell = if let Some(c) = invalid_cells.pop() {
            c
        } else {
            (0..n)
                .find(|&i| !valid[i])
                .ok_or_else(|| anyhow::anyhow!("invalid state"))?
        };
        if valid[cell] {
            continue;
        }

        let mut best: Option<BestMove> = None;
        for dir in 0..6 {
            let Some(nb) = board.neighbors[cell][dir] else {
                continue;
            };
            let opp = opposite_dir(dir);
            let old_a = edges[cell][dir];
            if old_a == 0 {
                continue;
            }
            let old_b = edges[nb][opp];

            let old_valid_cell = valid[cell];
            let old_valid_nb = valid[nb];
            let old_invalid = (!old_valid_cell as i32) + (!old_valid_nb as i32);

            let mut consider = |new_a: i32, new_b: i32| {
                edges[cell][dir] = new_a;
                edges[nb][opp] = new_b;
                let new_valid_cell = is_piece_valid(&edges[cell], cfg, memo);
                let new_valid_nb = is_piece_valid(&edges[nb], cfg, memo);
                let new_invalid = (!new_valid_cell as i32) + (!new_valid_nb as i32);
                let delta = old_invalid - new_invalid;
                let penalty = cell_penalty(&edges[cell], targets[cell], &memo.edge_meta)
                    + cell_penalty(&edges[nb], targets[nb], &memo.edge_meta);
                edges[cell][dir] = old_a;
                edges[nb][opp] = old_b;

                let mut score = (delta as f64) * 1_000_000.0;
                if new_valid_cell {
                    score += 200_000.0;
                }
                if new_valid_nb {
                    score += 20_000.0;
                }
                score -= penalty;

                let replace = match &best {
                    None => true,
                    Some(b) => score > b.score,
                };
                if replace {
                    best = Some(BestMove {
                        cell,
                        nb,
                        dir,
                        new_a,
                        new_b,
                        new_valid_cell,
                        new_valid_nb,
                        delta,
                        score,
                    });
                }
            };

            consider(-old_a, -old_b);

            let old_id = old_a.abs();
            if core_set.contains(&old_id) {
                let trials = core_ids.len().min(8).max(1);
                for _ in 0..3 {
                    let new_id = core_ids[..trials].choose(rng).copied().unwrap();
                    if new_id == old_id {
                        continue;
                    }
                    consider(-new_id, new_id);
                    consider(new_id, -new_id);
                }
            }

        }

        let Some(best) = best else {
            bail!("repair failed: no candidate moves");
        };

        let old_valid_cell = valid[best.cell];
        let old_valid_nb = valid[best.nb];
        let old_invalid = (!old_valid_cell as i32) + (!old_valid_nb as i32);

        edges[best.cell][best.dir] = best.new_a;
        edges[best.nb][opposite_dir(best.dir)] = best.new_b;

        let new_valid_cell = is_piece_valid(&edges[best.cell], cfg, memo);
        let new_valid_nb = is_piece_valid(&edges[best.nb], cfg, memo);
        valid[best.cell] = new_valid_cell;
        valid[best.nb] = new_valid_nb;

        let new_invalid = (!new_valid_cell as i32) + (!new_valid_nb as i32);
        invalid_count = (invalid_count as i32 - old_invalid + new_invalid) as usize;

        if new_valid_cell {
            stagnation = 0;
        } else {
            stagnation += 1;
        }

        if !new_valid_cell {
            invalid_cells.push(best.cell);
        }
        if !new_valid_nb {
            invalid_cells.push(best.nb);
        }

        if stagnation > 500 {
            stagnation = 0;
            let mut candidates = Vec::new();
            for dir in 0..6 {
                if let Some(nb) = board.neighbors[cell][dir] {
                    let id = edges[cell][dir].abs();
                    if id != 0 && core_set.contains(&id) {
                        candidates.push((cell, nb, dir));
                    }
                }
            }
            if let Some((a, b, dir)) = candidates.choose(rng).copied() {
                let opp = opposite_dir(dir);
                let new_id = core_ids.choose(rng).copied().unwrap();
                if rng.random_bool(0.5) {
                    edges[a][dir] = -new_id;
                    edges[b][opp] = new_id;
                } else {
                    edges[a][dir] = new_id;
                    edges[b][opp] = -new_id;
                }
                valid[a] = is_piece_valid(&edges[a], cfg, memo);
                valid[b] = is_piece_valid(&edges[b], cfg, memo);
            }
            invalid_cells.clear();
            for i in 0..n {
                if !valid[i] {
                    invalid_cells.push(i);
                }
            }
            invalid_count = invalid_cells.len();
        }
    }

    if invalid_count == 0 {
        Ok(steps)
    } else {
        bail!("repair failed: {} invalid pieces remain", invalid_count)
    }
}

fn generate_decoy_clusters(
    _board: &Board,
    cfg: &GeneratorConfig,
    rng: &mut impl Rng,
    memo: &mut GeomMemo,
    used_ids: &HashSet<i32>,
    start_id: usize,
) -> Result<Vec<Piece>> {
    let core_ids = core_id_pool(cfg, memo)?;
    let mut pieces = Vec::new();
    let mut next_piece_id = start_id;
    let true_ids: Vec<i32> = used_ids.iter().cloned().collect();
    let unused_ids: Vec<i32> = (1..=cfg.total_alphabet)
        .filter(|id| !used_ids.contains(id))
        .collect();
    let mut cluster_id = 0;

    for (size, count) in &cfg.false_clusters {
        for _ in 0..*count {
            let mut success = false;
            for _ in 0..cfg.max_retries {
                let cells = random_cluster(*size, rng);
                let cluster =
                    assign_decoy_cluster(&cells, cfg, rng, &core_ids, &true_ids, &unused_ids);

                let mut temp_id = next_piece_id;
                let mut cluster_pieces = Vec::new();
                let mut ok = true;
                for edges in cluster {
                    let rot = rng.random_range(0..6);
                    let rotated = rotate_edges(&edges, rot);
                    if build_piece_geom_if_valid(&rotated, cfg.min_gap, memo).is_none() {
                        ok = false;
                        break;
                    }
                    cluster_pieces.push(Piece {
                        id: temp_id,
                        edges: rotated,
                        is_true: false,
                        cluster_id: Some(cluster_id),
                        back_id: if cfg.back_id {
                            Some(rng.random())
                        } else {
                            None
                        },
                    });
                    temp_id += 1;
                }

                if ok {
                    pieces.extend(cluster_pieces);
                    next_piece_id = temp_id;
                    success = true;
                    break;
                }
            }

            if !success {
                bail!("failed to generate valid decoy cluster");
            }
            cluster_id += 1;
        }
    }

    Ok(pieces)
}

fn assign_decoy_cluster(
    cells: &[Axial],
    cfg: &GeneratorConfig,
    rng: &mut impl Rng,
    core_ids: &[i32],
    true_ids: &[i32],
    unused_ids: &[i32],
) -> Vec<[i32; 6]> {
    let mut index = HashMap::new();
    for (i, cell) in cells.iter().enumerate() {
        index.insert(*cell, i);
    }

    let mut edges = vec![[0; 6]; cells.len()];

    for i in 0..cells.len() {
        let cell = cells[i];
        for dir in 0..6 {
            let nb = cell.add(DIRS[dir]);
            if let Some(&j) = index.get(&nb) {
                if j < i {
                    continue;
                }
                let id = *core_ids.choose(rng).unwrap();
                assign_edge(&mut edges, i, j, dir, id, rng);
            }
        }
    }

    for i in 0..cells.len() {
        for dir in 0..6 {
            if edges[i][dir] != 0 {
                continue;
            }
            let roll: f64 = rng.random();
            let val = if roll < cfg.false_flat_ratio {
                0
            } else if roll < cfg.false_flat_ratio + cfg.false_attach_ratio {
                pick_id(true_ids, core_ids, rng)
            } else {
                pick_id(unused_ids, core_ids, rng)
            };
            if val == 0 {
                edges[i][dir] = 0;
            } else {
                let sign = if rng.random_bool(0.5) { 1 } else { -1 };
                edges[i][dir] = sign * val;
            }
        }
    }

    edges
}

fn pick_id(pool: &[i32], fallback: &[i32], rng: &mut impl Rng) -> i32 {
    if pool.is_empty() {
        return *fallback.choose(rng).unwrap();
    }
    *pool.choose(rng).unwrap()
}

fn random_spanning_tree(board: &Board, rng: &mut impl Rng) -> Vec<(usize, usize, usize)> {
    let n = board.cell_count();
    let mut visited = vec![false; n];
    let mut stack = vec![0usize];
    visited[0] = true;
    let mut edges = Vec::with_capacity(n - 1);

    while let Some(&current) = stack.last() {
        let mut dirs: Vec<usize> = (0..6).collect();
        dirs.shuffle(rng);
        let mut advanced = false;
        for dir in dirs {
            if let Some(next) = board.neighbors[current][dir] {
                if !visited[next] {
                    visited[next] = true;
                    edges.push((current, next, dir));
                    stack.push(next);
                    advanced = true;
                    break;
                }
            }
        }
        if !advanced {
            stack.pop();
        }
    }

    edges
}

fn adjacency_edges(board: &Board) -> Vec<(usize, usize, usize)> {
    let mut edges = Vec::new();
    for a in 0..board.cell_count() {
        for dir in 0..6 {
            if let Some(b) = board.neighbors[a][dir] {
                if b > a {
                    edges.push((a, b, dir));
                }
            }
        }
    }
    edges
}

fn assign_edge(
    edges: &mut [[i32; 6]],
    a: usize,
    b: usize,
    dir: usize,
    id: i32,
    rng: &mut impl Rng,
) {
    let sign = if rng.random_bool(0.5) { 1 } else { -1 };
    edges[a][dir] = sign * id;
    edges[b][opposite_dir(dir)] = -sign * id;
}

fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn random_cluster(size: usize, rng: &mut impl Rng) -> Vec<Axial> {
    let mut cells = Vec::with_capacity(size);
    let mut set = HashSet::new();
    let start = Axial::new(0, 0);
    cells.push(start);
    set.insert(start);

    while cells.len() < size {
        let base = *cells.choose(rng).unwrap();
        let dir = rng.random_range(0..6);
        let next = base.add(DIRS[dir]);
        if set.insert(next) {
            cells.push(next);
        }
    }

    cells
}
