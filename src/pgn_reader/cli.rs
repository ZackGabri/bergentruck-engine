use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::read_pgn;
use super::viriformat::write_game;

const USAGE: &str = "usage: packpgns --input <file.pgn | directory> [--output <file.vf>]";
const DEFAULT_OUTPUT: &str = "games.vf";

pub fn pack_files(args: &[&str]) {
    let mut input_path: Option<&Path> = None;
    let mut output_path: Option<&Path> = None;

    let split_args = args.chunks(2);
    for cli_argument in split_args {
        match cli_argument {
            ["--input" | "-i", input, ..] => {
                input_path = Some(Path::new(input));
            }
            ["--output" | "-o", output, ..] => {
                output_path = Some(Path::new(output));
            }
            // add option for viri/binpack later on, rn there's only viri
            ["--format" | "-f", format, ..] => {}
            other => {
                eprintln!(
                    "ignoring unrecognised or incomplete argument: {}",
                    other.join(" ")
                );
            }
        }
    }

    let Some(input_path) = input_path else {
        eprintln!("missing --input\n{USAGE}");
        return;
    };
    let output_path = output_path.unwrap_or_else(|| Path::new(DEFAULT_OUTPUT));

    let Ok(input) = std::path::absolute(input_path) else {
        eprintln!("Invalid input path provided!");
        return;
    };
    let Ok(output) = std::path::absolute(output_path) else {
        eprintln!("Invalid output path provided!");
        return;
    };

    println!("input file/directory: {}", input.display());
    println!("output file: {}", output.display());

    if let Err(e) = pack(input.as_path(), output.as_path()) {
        eprintln!("error: {e:#}");
    }
}

// A single file, or every .pgn file directly inside a directory
fn collect_pgn_files(input: &Path) -> anyhow::Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    anyhow::ensure!(input.is_dir(), "{} does not exist", input.display());

    let mut files: Vec<PathBuf> = std::fs::read_dir(input)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("pgn"))
        })
        .collect();
    files.sort();
    Ok(files)
}

/// Items per second, returning 0 instead of dividing by zero on very short runs.
fn rate(count: usize, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs > 0.0 { count as f64 / secs } else { 0.0 }
}

fn pack(input: &Path, output: &Path) -> anyhow::Result<()> {
    let files = collect_pgn_files(input)?;
    anyhow::ensure!(
        !files.is_empty(),
        "no .pgn files found in {}",
        input.display()
    );

    let start = Instant::now();

    let mut writer = BufWriter::new(File::create(output)?);

    let mut written_total = 0usize;
    let mut positions_total = 0usize; // every move written is one position
    let mut positions_scored = 0usize; // positions that actually have an eval
    let mut games_empty = 0usize;
    let mut files_failed = 0usize;

    for (i, file) in files.iter().enumerate() {
        let file_start = Instant::now();

        // One file's games are held in memory at a time, then written out.
        let games = match read_pgn(file) {
            Ok(games) => games,
            Err(e) => {
                eprintln!(
                    "[{}/{}] {}: failed to read: {e:#}",
                    i + 1,
                    files.len(),
                    file.display()
                );
                files_failed += 1;
                continue;
            }
        };

        let mut written = 0usize;
        let mut positions = 0usize;
        for game in &games {
            if game.moves.is_empty() {
                games_empty += 1;
                continue;
            }
            write_game(&mut writer, game)?;
            written += 1;
            positions += game.moves.len();
            positions_scored += game.moves.iter().filter(|m| m.score.is_some()).count();
        }

        written_total += written;
        positions_total += positions;

        let file_elapsed = file_start.elapsed();
        println!(
            "[{}/{}] {}: {} games, {} positions in {:.2?}",
            i + 1,
            files.len(),
            file.file_name().unwrap().display(),
            written,
            positions,
            file_elapsed,
        );
    }

    writer.flush()?;

    let elapsed = start.elapsed();
    println!(
        "wrote {written_total} games, {positions_total} positions ({positions_scored} scored) to {} ({games_empty} empty games skipped, {files_failed} files failed)",
        output.display()
    );
    println!(
        "total {:.2?} | {:.0} games/s, {:.0} positions/s",
        elapsed,
        rate(written_total, elapsed),
        rate(positions_total, elapsed),
    );
    Ok(())
}
