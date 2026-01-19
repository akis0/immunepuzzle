use anyhow::{bail, Result};
use rand::prelude::IndexedRandom;
use rand::{seq::SliceRandom, Rng};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

use crate::geometry::{build_piece_geom_if_valid, GeomMemo};
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
    rng: &mut impl Rng,
    memo: &mut GeomMemo,
    seed: u64,
) -> Result<GeneratedPuzzle> {
    let mut retries = 0;
    let (true_pieces, solution, used_ids) = loop {
        let attempt = assign_true_pieces(board, cfg, rng, memo);
        match attempt {
            Ok(ok) => break ok,
            Err(_) => {
                retries += 1;
                if retries >= cfg.max_retries {
                    bail!("failed to generate valid true pieces after {} retries", retries);
                }
            }
        }
    };

    let mut pieces = true_pieces;
    let start_id = pieces.len();
    let decoys = generate_decoy_clusters(board, cfg, rng, memo, &used_ids, start_id)?;
    pieces.extend(decoys);

    Ok(GeneratedPuzzle {
        pieces,
        solution,
        seed,
        retries,
    })
}

fn assign_true_pieces(
    board: &Board,
    cfg: &GeneratorConfig,
    rng: &mut impl Rng,
    memo: &mut GeomMemo,
) -> Result<(Vec<Piece>, Vec<SolutionEntry>, HashSet<i32>)> {
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

    let tree_edges = random_spanning_tree(board, rng);
    let mut anchor_edges = tree_edges.clone();
    anchor_edges.shuffle(rng);
    anchor_edges.truncate(anchor_target);
    let mut anchor_set = HashSet::new();
    for (a, b, _) in &anchor_edges {
        anchor_set.insert(edge_key(*a, *b));
    }

    let mut edges = vec![[0; 6]; n];
    let mut used_ids = HashSet::new();
    let mut next_anchor = cfg.core_alphabet + 1;

    for (a, b, dir) in &tree_edges {
        if anchor_set.contains(&edge_key(*a, *b)) {
            assign_edge(&mut edges, *a, *b, *dir, next_anchor, rng);
            used_ids.insert(next_anchor.abs());
            next_anchor += 1;
        }
    }

    for a in 0..n {
        for dir in 0..6 {
            if let Some(b) = board.neighbors[a][dir] {
                if b < a {
                    continue;
                }
                if edges[a][dir] != 0 {
                    continue;
                }
                let id = rng.random_range(1..=cfg.core_alphabet);
                assign_edge(&mut edges, a, b, dir, id, rng);
                used_ids.insert(id.abs());
            }
        }
    }

    if cfg.chiral {
        let mut available: Vec<i32> = ((cfg.core_alphabet + anchor_target as i32 + 1)
            ..=cfg.total_alphabet)
            .collect();
        if !available.is_empty() {
            available.shuffle(rng);
            let mut internal_edges = adjacency_edges(board);
            internal_edges.shuffle(rng);
            for (idx, (a, b, dir)) in internal_edges.into_iter().take(2).enumerate() {
                if idx >= available.len() {
                    break;
                }
                let id = available[idx];
                assign_edge(&mut edges, a, b, dir, id, rng);
                used_ids.insert(id.abs());
            }
        }
    }

    let mut pieces = Vec::with_capacity(n);
    let mut solution = Vec::with_capacity(n);
    for (idx, cell) in board.cells.iter().enumerate() {
        let rot = rng.random_range(0..6);
        let rotated = rotate_edges(&edges[idx], rot);
        if build_piece_geom_if_valid(&rotated, cfg.min_gap, memo).is_none() {
            bail!("invalid geometry for true piece");
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

    Ok((pieces, solution, used_ids))
}

fn generate_decoy_clusters(
    _board: &Board,
    cfg: &GeneratorConfig,
    rng: &mut impl Rng,
    memo: &mut GeomMemo,
    used_ids: &HashSet<i32>,
    start_id: usize,
) -> Result<Vec<Piece>> {
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
                let cluster = assign_decoy_cluster(
                    &cells,
                    cfg,
                    rng,
                    &true_ids,
                    &unused_ids,
                );

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
                        back_id: if cfg.back_id { Some(rng.random()) } else { None },
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
                let id = rng.random_range(1..=cfg.core_alphabet);
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
                pick_id(true_ids, cfg, rng)
            } else {
                pick_id(unused_ids, cfg, rng)
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

fn pick_id(pool: &[i32], cfg: &GeneratorConfig, rng: &mut impl Rng) -> i32 {
    if pool.is_empty() {
        return rng.random_range(1..=cfg.core_alphabet);
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
