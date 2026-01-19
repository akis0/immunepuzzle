use anyhow::{bail, Result};
use serde::Serialize;
use serde_json::json;
use std::env;
use std::fs;
use std::path::Path;

mod generator;
mod geometry;
mod grid;
mod solver;

use generator::{generate_puzzle, GeneratedPuzzle, GeneratorConfig, Piece, SolutionEntry};
use geometry::{bounds, build_piece_geom_if_valid, svg_path, GeomMemo};
use grid::Board;
use solver::{existence_solver, uniqueness_solver, SolverResult, WitnessEntry};

#[derive(Clone, Debug)]
struct Cli {
    side: i32,
    total_alphabet: i32,
    core_alphabet: i32,
    anchor_ratio: f64,
    false_flat: f64,
    false_attach: f64,
    false_clusters: Vec<(usize, usize)>,
    trials: usize,
    seed: Option<u64>,
    max_u: u64,
    max_f: u64,
    dump_svg: bool,
    svg_dir: String,
    svg_filter: String,
    min_gap: f64,
    chiral: bool,
    back_id: bool,
    dump_witness: bool,
}

#[derive(Clone, Debug, Serialize)]
struct PuzzleStats {
    seed: u64,
    score: u64,
    u_nodes: u64,
    f_nodes: u64,
    aborted: bool,
    valid: bool,
    retries: usize,
    decoy_solution_found: bool,
}

#[derive(Clone, Debug, Serialize)]
struct PuzzleOut {
    side_len: i32,
    pieces: Vec<Piece>,
    stats: PuzzleStats,
}

#[derive(Clone, Debug, Serialize)]
struct Output {
    puzzle: PuzzleOut,
    solution: Vec<SolutionEntry>,
    unused_piece_ids: Vec<usize>,
    witness: Option<Vec<WitnessEntry>>,
}

#[derive(Clone, Debug)]
struct Candidate {
    puzzle: GeneratedPuzzle,
    u_res: SolverResult,
    f_res: SolverResult,
    score: u64,
    valid: bool,
    aborted: bool,
}

fn main() -> Result<()> {
    let cli = parse_args()?;
    let board = Board::new(cli.side);
    if cli.total_alphabet > 295 {
        bail!("total_alphabet must be <= 295");
    }

    let mut memo = GeomMemo::new(cli.total_alphabet);
    let mut best_valid: Option<Candidate> = None;
    let mut best_any: Option<Candidate> = None;

    for trial in 0..cli.trials {
        let seed = if trial == 0 {
            cli.seed.unwrap_or_else(rand::random)
        } else {
            rand::random()
        };
        let cfg = GeneratorConfig {
            total_alphabet: cli.total_alphabet,
            core_alphabet: cli.core_alphabet,
            anchor_ratio: cli.anchor_ratio,
            false_flat_ratio: cli.false_flat,
            false_attach_ratio: cli.false_attach,
            false_clusters: cli.false_clusters.clone(),
            min_gap: cli.min_gap,
            chiral: cli.chiral,
            back_id: cli.back_id,
            max_retries: 200000,
        };

        let puzzle = generate_puzzle(&board, &cfg, &mut memo, seed)?;

        let true_pieces: Vec<Piece> = puzzle
            .pieces
            .iter()
            .cloned()
            .filter(|p| p.is_true)
            .collect();
        let u_res = uniqueness_solver(&board, &true_pieces, cli.total_alphabet, cli.max_u);
        let f_res = existence_solver(&board, &puzzle.pieces, cli.total_alphabet, cli.max_f);
        let score = f_res.nodes + u_res.nodes / 4;

        let unique_ok = u_res.solutions <= 1;
        let decoy_ok = f_res.solutions == 0;
        let valid = unique_ok && decoy_ok;
        let aborted = u_res.aborted || f_res.aborted;

        let candidate = Candidate {
            puzzle,
            u_res,
            f_res,
            score,
            valid,
            aborted,
        };

        best_any = pick_better(best_any, &candidate, false);
        if valid {
            best_valid = pick_better(best_valid, &candidate, true);
        }

        eprintln!(
            "trial {} seed={} score={} u_nodes={} f_nodes={} aborted={} valid={}",
            trial + 1,
            candidate.puzzle.seed,
            score,
            candidate.u_res.nodes,
            candidate.f_res.nodes,
            aborted,
            valid
        );
    }

    let selected = if let Some(best) = best_valid {
        best
    } else if let Some(best) = best_any {
        best
    } else {
        bail!("no candidates generated");
    };

    let unused_piece_ids: Vec<usize> = selected
        .puzzle
        .pieces
        .iter()
        .filter(|p| !p.is_true)
        .map(|p| p.id)
        .collect();

    let stats = PuzzleStats {
        seed: selected.puzzle.seed,
        score: selected.score,
        u_nodes: selected.u_res.nodes,
        f_nodes: selected.f_res.nodes,
        aborted: selected.aborted,
        valid: selected.valid,
        retries: selected.puzzle.retries,
        decoy_solution_found: selected.f_res.solutions > 0,
    };

    let output = Output {
        puzzle: PuzzleOut {
            side_len: cli.side,
            pieces: selected.puzzle.pieces.clone(),
            stats,
        },
        solution: selected.puzzle.solution.clone(),
        unused_piece_ids,
        witness: if cli.dump_witness {
            selected.f_res.witness.clone()
        } else {
            None
        },
    };

    let json = serde_json::to_string_pretty(&output)?;
    println!("{}", json);

    if cli.dump_svg && selected.valid {
        dump_svgs(
            &cli.svg_dir,
            &selected.puzzle.pieces,
            &cli.svg_filter,
            cli.min_gap,
            &mut memo,
        )?;
    }

    Ok(())
}

fn pick_better(
    current: Option<Candidate>,
    next: &Candidate,
    prefer_valid: bool,
) -> Option<Candidate> {
    match current {
        None => Some(next.clone()),
        Some(best) => {
            if prefer_valid && best.valid != next.valid {
                return if next.valid {
                    Some(next.clone())
                } else {
                    Some(best)
                };
            }
            if next.score > best.score {
                Some(next.clone())
            } else if next.score == best.score && best.aborted && !next.aborted {
                Some(next.clone())
            } else {
                Some(best)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SvgFilter {
    All,
    TrueOnly,
    FalseOnly,
}

fn dump_svgs(
    out_dir: &str,
    pieces: &[Piece],
    filter: &str,
    min_gap: f64,
    memo: &mut GeomMemo,
) -> Result<()> {
    let filter = match filter {
        "all" => SvgFilter::All,
        "false" => SvgFilter::FalseOnly,
        _ => SvgFilter::TrueOnly,
    };

    fs::create_dir_all(out_dir)?;
    for piece in pieces {
        let include = match filter {
            SvgFilter::All => true,
            SvgFilter::TrueOnly => piece.is_true,
            SvgFilter::FalseOnly => !piece.is_true,
        };
        if !include {
            continue;
        }
        let geom = match build_piece_geom_if_valid(&piece.edges, min_gap, memo) {
            Some(g) => g,
            None => continue,
        };
        let (min, max) = match bounds(&geom.points) {
            Some(b) => b,
            None => continue,
        };
        let pad = 0.2;
        let width = (max.x - min.x) + pad * 2.0;
        let height = (max.y - min.y) + pad * 2.0;
        let view_box = format!(
            "{:.4} {:.4} {:.4} {:.4}",
            min.x - pad,
            min.y - pad,
            width,
            height
        );
        let color = if piece.is_true { "#111111" } else { "#b22222" };
        let svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{}\">\n  <path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"0.02\"/>\n</svg>\n",
            view_box,
            svg_path(&geom.points),
            color
        );
        let filename = format!("piece_{:04}.svg", piece.id);
        let path = Path::new(out_dir).join(filename);
        fs::write(path, svg)?;
    }
    Ok(())
}

fn parse_args() -> Result<Cli> {
    let mut side = 8;
    let mut total_alphabet = 295;
    let mut core_alphabet = 15;
    let mut anchor_ratio = 0.2;
    let mut false_flat = 0.1;
    let mut false_attach = 0.2;
    let mut false_clusters = parse_clusters("1x10,2x8,4x4")?;
    let mut trials = 1000;
    let mut seed = None;
    let mut max_u = 200_000;
    let mut max_f = 300_000;
    let mut dump_svg = false;
    let mut svg_dir = String::from("out/svg");
    let mut svg_filter = String::from("true");
    let mut min_gap = 0.2;
    let mut chiral = false;
    let mut back_id = false;
    let mut dump_witness = false;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let (key, inline) = split_arg(&arg);
        match key.as_str() {
            "--side" => side = take_value(inline, &mut args)?.parse()?,
            "--alphabet" => core_alphabet = take_value(inline, &mut args)?.parse()?,
            "--core-alphabet" => core_alphabet = take_value(inline, &mut args)?.parse()?,
            "--total-alphabet" => total_alphabet = take_value(inline, &mut args)?.parse()?,
            "--anchor-ratio" => anchor_ratio = take_value(inline, &mut args)?.parse()?,
            "--false-flat" => false_flat = take_value(inline, &mut args)?.parse()?,
            "--false-attach" => false_attach = take_value(inline, &mut args)?.parse()?,
            "--false-clusters" => false_clusters = parse_clusters(&take_value(inline, &mut args)?)?,
            "--trials" => trials = take_value(inline, &mut args)?.parse()?,
            "--seed" => seed = Some(take_value(inline, &mut args)?.parse()?),
            "--max-u" => max_u = take_value(inline, &mut args)?.parse()?,
            "--max-f" => max_f = take_value(inline, &mut args)?.parse()?,
            "--dump-svg" => dump_svg = true,
            "--svg-dir" => svg_dir = take_value(inline, &mut args)?,
            "--svg-filter" => svg_filter = take_value(inline, &mut args)?,
            "--min-gap" => min_gap = take_value(inline, &mut args)?.parse()?,
            "--chiral" => chiral = true,
            "--back-id" => back_id = true,
            "--dump-witness" => dump_witness = true,
            "--help" => {
                print_help();
                std::process::exit(0);
            }
            _ => bail!("unknown argument: {}", arg),
        }
    }

    Ok(Cli {
        side,
        total_alphabet,
        core_alphabet,
        anchor_ratio,
        false_flat,
        false_attach,
        false_clusters,
        trials,
        seed,
        max_u,
        max_f,
        dump_svg,
        svg_dir,
        svg_filter,
        min_gap,
        chiral,
        back_id,
        dump_witness,
    })
}

fn split_arg(arg: &str) -> (String, Option<String>) {
    if let Some(pos) = arg.find('=') {
        (arg[..pos].to_string(), Some(arg[pos + 1..].to_string()))
    } else {
        (arg.to_string(), None)
    }
}

fn take_value(inline: Option<String>, args: &mut impl Iterator<Item = String>) -> Result<String> {
    if let Some(value) = inline {
        Ok(value)
    } else if let Some(next) = args.next() {
        Ok(next)
    } else {
        bail!("missing value for argument")
    }
}

fn parse_clusters(input: &str) -> Result<Vec<(usize, usize)>> {
    let mut out = Vec::new();
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(out);
    }
    for part in trimmed.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let part = part.trim_start_matches("size");
        let mut split = part.split(|c| c == 'x' || c == 'X' || c == '*');
        let size = split
            .next()
            .ok_or_else(|| anyhow::anyhow!("bad cluster spec"))?
            .trim()
            .parse()?;
        let count = split
            .next()
            .ok_or_else(|| anyhow::anyhow!("bad cluster spec"))?
            .trim()
            .parse()?;
        out.push((size, count));
    }
    Ok(out)
}

fn print_help() {
    let msg = json!({
        "usage": "puzzle [options]",
        "options": [
            "--side <i32>",
            "--core-alphabet <i32>",
            "--total-alphabet <i32>",
            "--anchor-ratio <f64>",
            "--false-flat <f64>",
            "--false-attach <f64>",
            "--false-clusters <spec>",
            "--trials <usize>",
            "--seed <u64>",
            "--max-u <u64>",
            "--max-f <u64>",
            "--dump-svg",
            "--svg-dir <path>",
            "--svg-filter <true|false|all>",
            "--min-gap <f64>",
            "--chiral",
            "--back-id",
            "--dump-witness"
        ]
    });
    eprintln!("{}", serde_json::to_string_pretty(&msg).unwrap());
}
