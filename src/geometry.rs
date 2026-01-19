use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::grid::rotate_edges;

#[derive(Clone, Copy, Debug, Default)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    fn add(self, other: Point) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    fn sub(self, other: Point) -> Point {
        Point {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    fn mul(self, k: f64) -> Point {
        Point {
            x: self.x * k,
            y: self.y * k,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PieceGeom {
    pub points: Vec<Point>,
}

#[derive(Clone, Copy, Debug)]
struct EdgeShape {
    kind: ShapeKind,
}

#[derive(Clone, Copy, Debug)]
enum ShapeKind {
    Trapezoid {
        t0: f64,
        length: f64,
        h0: f64,
        h1: f64,
    },
    Triangle {
        t0: f64,
        length: f64,
        apex_offset: f64,
        height: f64,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct EdgeMeta {
    pub max_height: f64,
}

#[derive(Debug)]
pub struct GeomMemo {
    cache: HashMap<[i32; 6], Option<PieceGeom>>,
    pub edge_meta: Vec<EdgeMeta>,
}

impl GeomMemo {
    pub fn new(total_alphabet: i32) -> Self {
        let mut edge_meta = vec![EdgeMeta { max_height: 0.0 }; (total_alphabet + 1) as usize];
        for id in 1..=total_alphabet {
            if let Some(shape) = edge_shape_from_id(id) {
                edge_meta[id as usize] = EdgeMeta {
                    max_height: shape.max_height(),
                };
            }
        }
        Self {
            cache: HashMap::new(),
            edge_meta,
        }
    }
}

pub fn canonical_shape(edges: &[i32; 6]) -> ([i32; 6], usize) {
    let mut best = *edges;
    let mut best_rot = 0;
    for rot in 1..6 {
        let cand = rotate_edges(edges, rot);
        if cand < best {
            best = cand;
            best_rot = rot;
        }
    }
    (best, best_rot)
}

pub fn build_piece_geom_if_valid(
    edges: &[i32; 6],
    min_gap: f64,
    memo: &mut GeomMemo,
) -> Option<PieceGeom> {
    let (canon, rot) = canonical_shape(edges);
    if let Some(cached) = memo.cache.get(&canon) {
        return cached.as_ref().map(|geom| rotate_geom(geom, (6 - rot) % 6));
    }

    let (geom, concave_segments, concave_ids) = build_piece_geom_raw(&canon);
    if !is_simple_polygon(&geom.points) {
        memo.cache.insert(canon, None);
        return None;
    }

    if concave_segments.len() >= 2 {
        let mut order: Vec<usize> = (0..concave_segments.len()).collect();
        order.sort_by(|&a, &b| {
            let ha = memo.edge_meta[concave_ids[a] as usize].max_height;
            let hb = memo.edge_meta[concave_ids[b] as usize].max_height;
            hb.partial_cmp(&ha).unwrap_or(std::cmp::Ordering::Equal)
        });

        for i in 0..order.len() {
            for j in (i + 1)..order.len() {
                let a = &concave_segments[order[i]];
                let b = &concave_segments[order[j]];
                if min_distance_segment_sets(a, b) < min_gap {
                    memo.cache.insert(canon, None);
                    return None;
                }
            }
        }
    }

    memo.cache.insert(canon, Some(geom.clone()));
    Some(rotate_geom(&geom, (6 - rot) % 6))
}

fn build_piece_geom_raw(edges: &[i32; 6]) -> (PieceGeom, Vec<Vec<(Point, Point)>>, Vec<i32>) {
    let verts = base_hex();
    let mut points = Vec::new();
    let mut concave_segments: Vec<Vec<(Point, Point)>> = Vec::new();
    let mut concave_ids: Vec<i32> = Vec::new();

    for i in 0..6 {
        let p0 = verts[i];
        let p1 = verts[(i + 1) % 6];
        let edge = edges[i];
        if i == 0 {
            points.push(p0);
        }

        if edge == 0 {
            push_point(&mut points, p1);
            continue;
        }

        let shape = match edge_shape_from_id(edge.abs()) {
            Some(s) => s,
            None => {
                push_point(&mut points, p1);
                continue;
            }
        };

        let edge_vec = p1.sub(p0);
        let edge_unit = unit(edge_vec);
        let outward = rotate_cw90(edge_unit);
        let sign = if edge > 0 { 1.0 } else { -1.0 };

        let mut segments = Vec::new();
        match shape.kind {
            ShapeKind::Trapezoid { t0, length, h0, h1 } => {
                let s = p0.add(edge_vec.mul(t0));
                let t = s.add(edge_vec.mul(length));
                let s_off = s.add(outward.mul(sign * h0));
                let t_off = t.add(outward.mul(sign * h1));
                push_point(&mut points, s);
                push_point(&mut points, s_off);
                push_point(&mut points, t_off);
                push_point(&mut points, t);
                push_point(&mut points, p1);
                if edge < 0 {
                    segments.push((s, s_off));
                    segments.push((s_off, t_off));
                    segments.push((t_off, t));
                }
            }
            ShapeKind::Triangle {
                t0,
                length,
                apex_offset,
                height,
            } => {
                let s = p0.add(edge_vec.mul(t0));
                let t = s.add(edge_vec.mul(length));
                let apex = s.add(edge_vec.mul(apex_offset));
                let apex_off = apex.add(outward.mul(sign * height));
                push_point(&mut points, s);
                push_point(&mut points, apex_off);
                push_point(&mut points, t);
                push_point(&mut points, p1);
                if edge < 0 {
                    segments.push((s, apex_off));
                    segments.push((apex_off, t));
                }
            }
        }

        if edge < 0 {
            concave_segments.push(segments);
            concave_ids.push(edge.abs());
        }
    }

    (PieceGeom { points }, concave_segments, concave_ids)
}

fn push_point(points: &mut Vec<Point>, p: Point) {
    if let Some(last) = points.last() {
        if dist2(*last, p) < 1e-12 {
            return;
        }
    }
    points.push(p);
}

fn base_hex() -> [Point; 6] {
    static HEX: OnceLock<[Point; 6]> = OnceLock::new();
    *HEX.get_or_init(|| {
        let mut verts = [Point::default(); 6];
        for i in 0..6 {
            let angle = (i as f64) * (PI / 3.0);
            verts[i] = Point {
                x: angle.cos(),
                y: angle.sin(),
            };
        }
        verts
    })
}

fn unit(v: Point) -> Point {
    let len = (v.x * v.x + v.y * v.y).sqrt();
    if len == 0.0 {
        return v;
    }
    v.mul(1.0 / len)
}

fn rotate_cw90(v: Point) -> Point {
    Point { x: v.y, y: -v.x }
}

fn edge_shape_from_id(id: i32) -> Option<EdgeShape> {
    if id < 1 || id > 295 {
        return None;
    }
    if id <= 175 {
        let idx = (id - 1) as usize;
        let pos_index = idx / (5 * 5);
        let rem = idx % (5 * 5);
        let h0_index = rem / 5;
        let h1_index = rem % 5;
        let (t0, length) = position_pattern(pos_index)?;
        let h0 = 0.2 + 0.1 * (h0_index as f64);
        let h1 = 0.2 + 0.1 * (h1_index as f64);
        return Some(EdgeShape {
            kind: ShapeKind::Trapezoid { t0, length, h0, h1 },
        });
    }

    let idx = (id - 176) as usize;
    if idx < 60 {
        let pos_index = idx / (4 * 5);
        let rem = idx % (4 * 5);
        let apex_index = rem / 5;
        let height_index = rem % 5;
        let (t0, length) = position_pattern(pos_index)?;
        let apex_offset = 0.1 * ((apex_index + 1) as f64);
        let height = 0.2 + 0.1 * (height_index as f64);
        return Some(EdgeShape {
            kind: ShapeKind::Triangle {
                t0,
                length,
                apex_offset,
                height,
            },
        });
    }

    let idx = idx - 60;
    let pos_index = idx / (3 * 5) + 3;
    let rem = idx % (3 * 5);
    let apex_index = rem / 5;
    let height_index = rem % 5;
    let (t0, length) = position_pattern(pos_index)?;
    let apex_offset = 0.1 * ((apex_index + 1) as f64);
    let height = 0.2 + 0.1 * (height_index as f64);
    Some(EdgeShape {
        kind: ShapeKind::Triangle {
            t0,
            length,
            apex_offset,
            height,
        },
    })
}

fn position_pattern(index: usize) -> Option<(f64, f64)> {
    match index {
        0 => Some((0.2, 0.4)),
        1 => Some((0.3, 0.4)),
        2 => Some((0.4, 0.4)),
        3 => Some((0.2, 0.3)),
        4 => Some((0.3, 0.3)),
        5 => Some((0.4, 0.3)),
        6 => Some((0.5, 0.3)),
        _ => None,
    }
}

impl EdgeShape {
    fn max_height(&self) -> f64 {
        match self.kind {
            ShapeKind::Trapezoid { h0, h1, .. } => h0.max(h1),
            ShapeKind::Triangle { height, .. } => height,
        }
    }
}

pub fn rotate_geom(geom: &PieceGeom, rot: usize) -> PieceGeom {
    if rot == 0 {
        return geom.clone();
    }
    let angle = (rot as f64) * (PI / 3.0);
    let cos = angle.cos();
    let sin = angle.sin();
    let mut points = Vec::with_capacity(geom.points.len());
    for p in &geom.points {
        points.push(Point {
            x: p.x * cos - p.y * sin,
            y: p.x * sin + p.y * cos,
        });
    }
    PieceGeom { points }
}

fn is_simple_polygon(points: &[Point]) -> bool {
    if points.len() < 3 {
        return false;
    }
    let n = points.len();
    for i in 0..n {
        let a1 = points[i];
        let a2 = points[(i + 1) % n];
        for j in (i + 1)..n {
            let b1 = points[j];
            let b2 = points[(j + 1) % n];
            if i == j {
                continue;
            }
            if (i + 1) % n == j || (j + 1) % n == i {
                continue;
            }
            if segments_intersect(a1, a2, b1, b2) {
                return false;
            }
        }
    }
    true
}

fn segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool {
    let o1 = orient(a1, a2, b1);
    let o2 = orient(a1, a2, b2);
    let o3 = orient(b1, b2, a1);
    let o4 = orient(b1, b2, a2);

    if o1 == 0 && on_segment(a1, a2, b1) {
        return true;
    }
    if o2 == 0 && on_segment(a1, a2, b2) {
        return true;
    }
    if o3 == 0 && on_segment(b1, b2, a1) {
        return true;
    }
    if o4 == 0 && on_segment(b1, b2, a2) {
        return true;
    }

    (o1 > 0 && o2 < 0 || o1 < 0 && o2 > 0) && (o3 > 0 && o4 < 0 || o3 < 0 && o4 > 0)
}

fn orient(a: Point, b: Point, c: Point) -> i32 {
    let val = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    if val.abs() < 1e-9 {
        0
    } else if val > 0.0 {
        1
    } else {
        -1
    }
}

fn on_segment(a: Point, b: Point, c: Point) -> bool {
    let min_x = a.x.min(b.x) - 1e-9;
    let max_x = a.x.max(b.x) + 1e-9;
    let min_y = a.y.min(b.y) - 1e-9;
    let max_y = a.y.max(b.y) + 1e-9;
    c.x >= min_x && c.x <= max_x && c.y >= min_y && c.y <= max_y
}

fn dist2(a: Point, b: Point) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy
}

fn min_distance_segment_sets(a: &[(Point, Point)], b: &[(Point, Point)]) -> f64 {
    let mut best = f64::INFINITY;
    for &(a1, a2) in a {
        for &(b1, b2) in b {
            let d = segment_distance(a1, a2, b1, b2);
            if d < best {
                best = d;
            }
        }
    }
    best
}

fn segment_distance(a1: Point, a2: Point, b1: Point, b2: Point) -> f64 {
    if segments_intersect(a1, a2, b1, b2) {
        return 0.0;
    }
    let d1 = point_to_segment_distance(a1, b1, b2);
    let d2 = point_to_segment_distance(a2, b1, b2);
    let d3 = point_to_segment_distance(b1, a1, a2);
    let d4 = point_to_segment_distance(b2, a1, a2);
    d1.min(d2).min(d3).min(d4)
}

fn point_to_segment_distance(p: Point, a: Point, b: Point) -> f64 {
    let ab = b.sub(a);
    let ap = p.sub(a);
    let ab_len2 = ab.x * ab.x + ab.y * ab.y;
    if ab_len2 == 0.0 {
        return (ap.x * ap.x + ap.y * ap.y).sqrt();
    }
    let mut t = (ap.x * ab.x + ap.y * ab.y) / ab_len2;
    if t < 0.0 {
        t = 0.0;
    } else if t > 1.0 {
        t = 1.0;
    }
    let proj = a.add(ab.mul(t));
    ((p.x - proj.x).powi(2) + (p.y - proj.y).powi(2)).sqrt()
}

pub fn svg_path(points: &[Point]) -> String {
    if points.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push_str(&format!("M {:.4} {:.4}", points[0].x, points[0].y));
    for p in points.iter().skip(1) {
        out.push_str(&format!(" L {:.4} {:.4}", p.x, p.y));
    }
    out.push_str(" Z");
    out
}

pub fn bounds(points: &[Point]) -> Option<(Point, Point)> {
    if points.is_empty() {
        return None;
    }
    let mut min = points[0];
    let mut max = points[0];
    for p in points.iter().skip(1) {
        if p.x < min.x {
            min.x = p.x;
        }
        if p.y < min.y {
            min.y = p.y;
        }
        if p.x > max.x {
            max.x = p.x;
        }
        if p.y > max.y {
            max.y = p.y;
        }
    }
    Some((min, max))
}
