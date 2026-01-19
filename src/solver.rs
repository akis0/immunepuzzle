use serde::Serialize;

use crate::generator::Piece;
use crate::grid::{opposite_dir, rotate_edges, Board};

#[derive(Clone, Debug)]
pub struct SolverResult {
    pub nodes: u64,
    pub aborted: bool,
    pub solutions: usize,
    pub witness: Option<Vec<WitnessEntry>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WitnessEntry {
    pub q: i32,
    pub r: i32,
    pub piece_id: usize,
    pub rot: usize,
}

struct Solver<'a> {
    board: &'a Board,
    total_alphabet: i32,
    placement_edges: Vec<[i32; 6]>,
    placement_piece: Vec<usize>,
    placement_is_decoy: Vec<bool>,
    placement_rot: Vec<usize>,
    postings: Vec<Vec<Vec<usize>>>,
    all_placements: Vec<usize>,
    has_decoys: bool,
}

struct SolverState {
    assigned: Vec<Option<usize>>,
    used_piece: Vec<bool>,
    dirty: Vec<bool>,
    cache: Vec<Vec<usize>>,
    decoy_used: bool,
}

struct SearchConfig {
    max_nodes: u64,
    max_solutions: usize,
    prefer_decoy: bool,
    require_decoy: bool,
}

impl<'a> Solver<'a> {
    fn new(board: &'a Board, pieces: &[Piece], total_alphabet: i32) -> Self {
        let mut placement_edges = Vec::new();
        let mut placement_piece = Vec::new();
        let mut placement_is_decoy = Vec::new();
        let mut placement_rot = Vec::new();
        let mut has_decoys = false;

        for (piece_idx, piece) in pieces.iter().enumerate() {
            if !piece.is_true {
                has_decoys = true;
            }
            for rot in 0..6 {
                let rotated = rotate_edges(&piece.edges, rot);
                placement_edges.push(rotated);
                placement_piece.push(piece_idx);
                placement_is_decoy.push(!piece.is_true);
                placement_rot.push(rot);
            }
        }

        let range = (2 * total_alphabet + 1) as usize;
        let mut postings = vec![vec![Vec::new(); range]; 6];
        for (idx, edges) in placement_edges.iter().enumerate() {
            for dir in 0..6 {
                let val = edges[dir];
                if val.abs() > total_alphabet {
                    continue;
                }
                let v = (val + total_alphabet) as usize;
                postings[dir][v].push(idx);
            }
        }

        let all_placements = (0..placement_edges.len()).collect();

        Self {
            board,
            total_alphabet,
            placement_edges,
            placement_piece,
            placement_is_decoy,
            placement_rot,
            postings,
            all_placements,
            has_decoys,
        }
    }

    fn initial_state(&self, piece_count: usize) -> SolverState {
        let n = self.board.cell_count();
        SolverState {
            assigned: vec![None; n],
            used_piece: vec![false; piece_count],
            dirty: vec![true; n],
            cache: vec![Vec::new(); n],
            decoy_used: false,
        }
    }

    fn mark_dirty(&self, state: &mut SolverState, cell: usize) {
        state.dirty[cell] = true;
        for dir in 0..6 {
            if let Some(nb) = self.board.neighbors[cell][dir] {
                state.dirty[nb] = true;
            }
        }
    }

    fn place(&self, state: &mut SolverState, cell: usize, placement: usize) {
        state.assigned[cell] = Some(placement);
        let piece_idx = self.placement_piece[placement];
        state.used_piece[piece_idx] = true;
        self.mark_dirty(state, cell);
    }

    fn unplace(&self, state: &mut SolverState, cell: usize, placement: usize) {
        state.assigned[cell] = None;
        let piece_idx = self.placement_piece[placement];
        state.used_piece[piece_idx] = false;
        self.mark_dirty(state, cell);
    }

    fn compute_structural_candidates(
        &self,
        cell: usize,
        assigned: &[Option<usize>],
    ) -> Vec<usize> {
        let mut req: [Option<i32>; 6] = [None, None, None, None, None, None];
        for dir in 0..6 {
            if self.board.boundary[cell][dir] {
                req[dir] = Some(0);
            }
            if let Some(nb) = self.board.neighbors[cell][dir] {
                if let Some(pl) = assigned[nb] {
                    let nb_edge = self.placement_edges[pl][opposite_dir(dir)];
                    req[dir] = Some(-nb_edge);
                }
            }
        }

        let mut best_dir = None;
        let mut best_len = usize::MAX;
        for dir in 0..6 {
            if let Some(val) = req[dir] {
                let idx = value_index(val, self.total_alphabet);
                let len = self.postings[dir][idx].len();
                if len < best_len {
                    best_len = len;
                    best_dir = Some(dir);
                }
            }
        }

        let base_list: &[usize] = if let Some(dir) = best_dir {
            let idx = value_index(req[dir].unwrap(), self.total_alphabet);
            &self.postings[dir][idx]
        } else {
            &self.all_placements
        };

        let mut out = Vec::new();
        'outer: for &pl in base_list {
            let edges = &self.placement_edges[pl];
            for dir in 0..6 {
                if let Some(val) = req[dir] {
                    if edges[dir] != val {
                        continue 'outer;
                    }
                }
            }
            out.push(pl);
        }
        out
    }

    fn get_structural_candidates<'b>(
        &self,
        state: &'b mut SolverState,
        cell: usize,
    ) -> &'b [usize] {
        if state.dirty[cell] {
            let candidates = self.compute_structural_candidates(cell, &state.assigned);
            state.cache[cell] = candidates;
            state.dirty[cell] = false;
        }
        &state.cache[cell]
    }

    fn available_count(&self, state: &mut SolverState, cell: usize) -> usize {
        let candidates = self.get_structural_candidates(state, cell).to_vec();
        let mut count = 0;
        for pl in candidates {
            let piece_idx = self.placement_piece[pl];
            if !state.used_piece[piece_idx] {
                count += 1;
            }
        }
        count
    }

    fn ordered_candidates(
        &self,
        state: &mut SolverState,
        cell: usize,
        prefer_decoy: bool,
    ) -> Vec<usize> {
        let candidates = self.get_structural_candidates(state, cell).to_vec();
        let mut out = Vec::new();
        if prefer_decoy {
            for pl in &candidates {
                let pl = *pl;
                let piece_idx = self.placement_piece[pl];
                if !state.used_piece[piece_idx] && self.placement_is_decoy[pl] {
                    out.push(pl);
                }
            }
            for pl in &candidates {
                let pl = *pl;
                let piece_idx = self.placement_piece[pl];
                if !state.used_piece[piece_idx] && !self.placement_is_decoy[pl] {
                    out.push(pl);
                }
            }
        } else {
            for pl in candidates {
                let piece_idx = self.placement_piece[pl];
                if !state.used_piece[piece_idx] {
                    out.push(pl);
                }
            }
        }
        out
    }

    fn choose_cell(&self, state: &mut SolverState) -> Option<(usize, usize)> {
        let mut best_cell = None;
        let mut best_count = usize::MAX;
        for cell in 0..self.board.cell_count() {
            if state.assigned[cell].is_some() {
                continue;
            }
            let count = self.available_count(state, cell);
            if count == 0 {
                return Some((cell, 0));
            }
            if count < best_count {
                best_count = count;
                best_cell = Some(cell);
            }
        }
        best_cell.map(|cell| (cell, best_count))
    }

    fn dfs(
        &self,
        state: &mut SolverState,
        cfg: &SearchConfig,
        nodes: &mut u64,
        solutions: &mut usize,
        witness: &mut Option<Vec<WitnessEntry>>,
    ) -> bool {
        if *nodes >= cfg.max_nodes {
            return true;
        }

        let mut finished = true;
        for cell in 0..self.board.cell_count() {
            if state.assigned[cell].is_none() {
                finished = false;
                break;
            }
        }
        if finished {
            if !cfg.require_decoy || state.decoy_used {
                *solutions += 1;
                if witness.is_none() {
                    *witness = Some(self.extract_witness(state));
                }
            }
            return false;
        }

        let (cell, count) = match self.choose_cell(state) {
            Some(v) => v,
            None => return false,
        };
        if count == 0 {
            return false;
        }

        let candidates = self.ordered_candidates(state, cell, cfg.prefer_decoy);
        for placement in candidates {
            if *nodes >= cfg.max_nodes {
                return true;
            }
            *nodes += 1;
            let prev_decoy = state.decoy_used;
            state.decoy_used = prev_decoy || self.placement_is_decoy[placement];
            self.place(state, cell, placement);
            let aborted = self.dfs(state, cfg, nodes, solutions, witness);
            self.unplace(state, cell, placement);
            state.decoy_used = prev_decoy;
            if aborted {
                return true;
            }
            if *solutions >= cfg.max_solutions {
                return false;
            }
        }
        false
    }

    fn extract_witness(&self, state: &SolverState) -> Vec<WitnessEntry> {
        let mut entries = Vec::with_capacity(self.board.cell_count());
        for (idx, cell) in self.board.cells.iter().enumerate() {
            let placement = state.assigned[idx].unwrap();
            let piece_idx = self.placement_piece[placement];
            entries.push(WitnessEntry {
                q: cell.q,
                r: cell.r,
                piece_id: piece_idx,
                rot: self.placement_rot[placement],
            });
        }
        entries
    }

    fn fits_boundary(&self, cell: usize, placement: usize) -> bool {
        for dir in 0..6 {
            if self.board.boundary[cell][dir] {
                if self.placement_edges[placement][dir] != 0 {
                    return false;
                }
            }
        }
        true
    }
}

fn value_index(value: i32, total_alphabet: i32) -> usize {
    (value + total_alphabet) as usize
}

pub fn uniqueness_solver(
    board: &Board,
    pieces: &[Piece],
    total_alphabet: i32,
    max_nodes: u64,
) -> SolverResult {
    let solver = Solver::new(board, pieces, total_alphabet);
    let mut state = solver.initial_state(pieces.len());
    let cfg = SearchConfig {
        max_nodes,
        max_solutions: 2,
        prefer_decoy: false,
        require_decoy: false,
    };
    let mut nodes = 0;
    let mut solutions = 0;
    let mut witness = None;
    let aborted = solver.dfs(&mut state, &cfg, &mut nodes, &mut solutions, &mut witness);
    SolverResult {
        nodes,
        aborted,
        solutions,
        witness,
    }
}

pub fn existence_solver(
    board: &Board,
    pieces: &[Piece],
    total_alphabet: i32,
    max_nodes: u64,
) -> SolverResult {
    let solver = Solver::new(board, pieces, total_alphabet);
    if !solver.has_decoys {
        return SolverResult {
            nodes: 0,
            aborted: false,
            solutions: 0,
            witness: None,
        };
    }

    let cfg = SearchConfig {
        max_nodes,
        max_solutions: 1,
        prefer_decoy: true,
        require_decoy: true,
    };

    let mut nodes = 0;
    let mut solutions = 0;
    let mut witness = None;

    let mut state = solver.initial_state(pieces.len());
    let decoy_placements: Vec<usize> = solver
        .placement_is_decoy
        .iter()
        .enumerate()
        .filter_map(|(i, &is_decoy)| if is_decoy { Some(i) } else { None })
        .collect();

    let max_anchor_attempts = 200usize;
    let mut attempts = 0usize;
    for &placement in &decoy_placements {
        if attempts >= max_anchor_attempts {
            break;
        }
        for cell in 0..board.cell_count() {
            if attempts >= max_anchor_attempts {
                break;
            }
            if !solver.fits_boundary(cell, placement) {
                continue;
            }
            attempts += 1;
            solver.place(&mut state, cell, placement);
            state.decoy_used = true;
            let aborted = solver.dfs(&mut state, &cfg, &mut nodes, &mut solutions, &mut witness);
            solver.unplace(&mut state, cell, placement);
            state.decoy_used = false;
            if solutions > 0 {
                return SolverResult {
                    nodes,
                    aborted,
                    solutions,
                    witness,
                };
            }
            if aborted {
                return SolverResult {
                    nodes,
                    aborted: true,
                    solutions,
                    witness,
                };
            }
        }
    }

    let aborted = solver.dfs(&mut state, &cfg, &mut nodes, &mut solutions, &mut witness);
    SolverResult {
        nodes,
        aborted,
        solutions,
        witness,
    }
}
