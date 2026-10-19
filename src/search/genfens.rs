use anyhow::{Context, bail};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use shakmaty::fen::{Epd, Fen};
use shakmaty::{CastlingMode, Chess, EnPassantMode, Position};
use std::fs;
use std::time::Instant;

pub const BASE_MOVES: usize = 6;
pub const MAX_RANDOM_MOVES: usize = 4;
const MAX_ACCEPTED_SCORE: i32 = 2000;
const VALIDATION_DEPTH: usize = crate::bench::BENCH_DEPTH - 1;

pub fn generate_fens(
    n: usize,
    seed: u64,
    book_path: Option<&str>,
    _extra: Option<&str>,
) -> anyhow::Result<()> {
    let start_time = Instant::now();
    let starts = match book_path {
        Some(path) => load_book(path)?,
        None => vec![Chess::default()],
    };

    let offset = (seed & u64::from(u32::MAX)) as usize % starts.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut book_index = offset;
    let mut accepted = 0;

    while accepted < n {
        let start = &starts[book_index];
        book_index = (book_index + 1) % starts.len();

        let Some((position, _move_count)) = generate_candidate(start, &mut rng) else {
            continue;
        };

        let (_, _, score) = crate::search::search(
            &position,
            None,
            crate::search::SearchOptions {
                max_depth: Some(VALIDATION_DEPTH),
                bench: true,
                ..Default::default()
            },
        );
        if !score_is_acceptable(score) {
            continue;
        }

        let fen = Fen::from_position(&position, EnPassantMode::Legal);
        println!("info string genfens {fen}");
        accepted += 1;
    }

    let elapsed = start_time.elapsed();
    let elapsed_seconds = elapsed.as_secs_f64();
    let positions_per_second = n as f64 / elapsed_seconds.max(f64::EPSILON);
    println!(
        "info string genfens generated {n} positions in {elapsed_seconds:.3} seconds ({positions_per_second:.2} positions/second)"
    );

    Ok(())
}

fn generate_candidate(start: &Chess, rng: &mut StdRng) -> Option<(Chess, usize)> {
    if start.is_game_over() {
        return None;
    }

    let move_count = BASE_MOVES + rng.random_range(0..=MAX_RANDOM_MOVES);
    let mut position = start.clone();

    for _ in 0..move_count * 2 {
        let moves = position.legal_moves();
        if moves.is_empty() {
            return None;
        }

        let selected_move = moves[rng.random_range(0..moves.len())];
        position.play_unchecked(selected_move);
        if position.is_game_over() {
            return None;
        }
    }

    Some((position, move_count))
}

fn score_is_acceptable(score: i32) -> bool {
    (-MAX_ACCEPTED_SCORE..=MAX_ACCEPTED_SCORE).contains(&score)
}

pub fn parse_genfens_args(
    tokens: &[String],
) -> anyhow::Result<(usize, u64, Option<&str>, Option<String>)> {
    let n = tokens
        .first()
        .context("genfens requires a position count N")?
        .parse::<usize>()
        .context("genfens N must be a positive integer")?;
    if n == 0 {
        bail!("genfens N must be greater than zero");
    }

    if tokens.get(1).map(String::as_str) != Some("seed") {
        bail!("genfens syntax: genfens N seed S book <None|path> [extra args...]");
    }

    let seed = tokens
        .get(2)
        .context("genfens requires a seed S after `seed`")?
        .parse::<u64>()
        .context("genfens seed S must be an unsigned 64-bit integer")?;

    if tokens.get(3).map(String::as_str) != Some("book") {
        bail!("genfens syntax: genfens N seed S book <None|path> [extra args...]");
    }

    let book = tokens
        .get(4)
        .context("genfens requires `None` or a book path after `book`")?;
    let book_path = if book == "None" {
        None
    } else {
        Some(book.as_str())
    };

    let extra = tokens[5..]
        .iter()
        .take_while(|token| token.as_str() != "quit")
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(" ");
    let extra = (!extra.is_empty()).then_some(extra);

    Ok((n, seed, book_path, extra))
}

fn load_book(path: &str) -> anyhow::Result<Vec<Chess>> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read genfens book `{path}`"))?;
    let mut positions = Vec::new();

    for (line_number, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<_> = line.split_ascii_whitespace().collect();
        if fields.len() < 4 {
            bail!(
                "invalid EPD/FEN in genfens book `{path}` at line {}",
                line_number + 1
            );
        }

        let position = if fields.len() >= 6
            && (fields[4].parse::<u32>().is_ok() || fields[5].parse::<u32>().is_ok())
        {
            let fen = fields[..6].join(" ").parse::<Fen>().with_context(|| {
                format!(
                    "invalid FEN in genfens book `{path}` at line {}",
                    line_number + 1
                )
            })?;
            fen.into_position::<Chess>(CastlingMode::Standard)
                .with_context(|| {
                    format!(
                        "invalid chess position in genfens book `{path}` at line {}",
                        line_number + 1
                    )
                })?
        } else {
            let epd = fields[..4].join(" ").parse::<Epd>().with_context(|| {
                format!(
                    "invalid EPD in genfens book `{path}` at line {}",
                    line_number + 1
                )
            })?;
            epd.into_position::<Chess>(CastlingMode::Standard)
                .with_context(|| {
                    format!(
                        "invalid chess position in genfens book `{path}` at line {}",
                        line_number + 1
                    )
                })?
        };

        positions.push(position);
    }

    if positions.is_empty() {
        bail!("genfens book `{path}` contains no EPD/FEN positions");
    }

    Ok(positions)
}
