use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Axial {
    pub q: i32,
    pub r: i32,
}

impl Axial {
    pub fn new(q: i32, r: i32) -> Self {
        Self { q, r }
    }

    pub fn add(&self, other: Axial) -> Self {
        Self {
            q: self.q + other.q,
            r: self.r + other.r,
        }
    }
}

pub const DIRS: [Axial; 6] = [
    Axial { q: 1, r: 0 },
    Axial { q: 1, r: -1 },
    Axial { q: 0, r: -1 },
    Axial { q: -1, r: 0 },
    Axial { q: -1, r: 1 },
    Axial { q: 0, r: 1 },
];

#[derive(Debug, Clone)]
pub struct Board {
    pub cells: Vec<Axial>,
    pub neighbors: Vec<[Option<usize>; 6]>,
    pub boundary: Vec<[bool; 6]>,
}

impl Board {
    pub fn new(side_len: i32) -> Self {
        let mut cells = Vec::new();
        let mut index = HashMap::new();
        let s = side_len - 1;
        for q in -s..=s {
            for r in -s..=s {
                let qr = q + r;
                if qr < -s || qr > s {
                    continue;
                }
                let axial = Axial::new(q, r);
                index.insert(axial, cells.len());
                cells.push(axial);
            }
        }

        let mut neighbors = vec![[None; 6]; cells.len()];
        let mut boundary = vec![[false; 6]; cells.len()];
        for (i, cell) in cells.iter().enumerate() {
            for dir in 0..6 {
                let nb = cell.add(DIRS[dir]);
                if let Some(&j) = index.get(&nb) {
                    neighbors[i][dir] = Some(j);
                } else {
                    boundary[i][dir] = true;
                }
            }
        }

        Self {
            cells,
            neighbors,
            boundary,
        }
    }

    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }
}

pub fn rotate_edges(edges: &[i32; 6], rot: usize) -> [i32; 6] {
    let mut out = [0; 6];
    for i in 0..6 {
        out[i] = edges[(i + rot) % 6];
    }
    out
}

pub fn opposite_dir(dir: usize) -> usize {
    (dir + 3) % 6
}
