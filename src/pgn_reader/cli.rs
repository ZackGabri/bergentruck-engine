use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context;

use super::GameRecord;
use super::input::PgnInput;
use super::input::{collect_pgn_files, read_tar_entries};
use super::output::{OutputFormat, OutputWriter};

const USAGE: &str = "usage: packpgns --input <file.pgn | directory> [--output <file | folder>] [--format <viriformat/sfbinpack>] [--one-file <true/false>] [--threads <n>]";
const DEFAULT_OUTPUT: &str = "games";

#[derive(Default)]
struct Stats {
    games: usize,
    positions: usize,
    skipped_games: usize,
    files_failed: usize,
}

impl Stats {
    fn merge(&mut self, other: &Stats) {
        self.games += other.games;
        self.positions += other.positions;
        self.skipped_games += other.skipped_games;
        self.files_failed += other.files_failed;
    }
}

pub fn pack_files(args: &[&str]) -> anyhow::Result<()> {
    let mut input_path: Option<&Path> = None;
    let mut output_path: Option<&Path> = None;
    let mut output_format = OutputFormat::Viriformat;
    let mut as_one = true;
    let mut threads = thread::available_parallelism().map_or(1, |n| n.get());

    for cli_argument in args.chunks(2) {
        match cli_argument {
            ["--input" | "-i", input, ..] => input_path = Some(Path::new(input)),
            ["--output" | "-o", output, ..] => output_path = Some(Path::new(output)),
            ["--format" | "-f", format, ..] => output_format = format.parse()?,
            ["--as-one" | "--one-file", yesno] => as_one = yesno.parse()?,
            ["--threads" | "-t", n] => {
                threads = n.parse().context("invalid thread count")?;
                anyhow::ensure!(threads > 0, "--threads must be at least 1");
            }
            other => {
                anyhow::bail!("unrecognised or incomplete argument: {}", other.join(" "));
            }
        }
    }

    let input_path = input_path.context(format!("missing --input\n{USAGE}"))?;
    let output_path = output_path.unwrap_or_else(|| Path::new(DEFAULT_OUTPUT));

    let input = std::path::absolute(input_path).context("Invalid input path provided!")?;
    let output = std::path::absolute(output_path).context("Invalid output path provided!")?;

    println!("input file/directory: {}", input.display());
    println!(
        "output {}: {}",
        if as_one { "file" } else { "folder" },
        output.display()
    );
    println!("format: {output_format:?}");

    pack(&input, &output, output_format, as_one, threads)
}

/// Items per second, returning 0 instead of dividing by zero on very short runs.
fn rate(count: usize, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs > 0.0 { count as f64 / secs } else { 0.0 }
}

fn write_games(writer: &mut OutputWriter, games: &[GameRecord]) -> Stats {
    let mut stats = Stats::default();

    for game in games {
        if game.moves.is_empty() || game.moves.iter().any(|rm| rm.score.is_none()) {
            stats.skipped_games += 1;
            continue;
        }
        if let Err(e) = writer.write_game(game) {
            eprintln!("failed to write game: {e:#}");
            stats.skipped_games += 1;
            continue;
        }
        stats.games += 1;
        stats.positions += game.moves.len();
    }
    stats
}

fn report(index: usize, total: usize, input: &PgnInput, stats: &Stats, elapsed: Duration) {
    println!(
        "[{}/{}] {}: {} games, {} positions in {:.2?}",
        index + 1,
        total,
        input.file_name().unwrap_or_default().display(),
        stats.games,
        stats.positions,
        elapsed,
    );
}

// all files go into one shared output file.
fn pack_shared(inputs: &[PgnInput], output: &Path, format: OutputFormat) -> anyhow::Result<Stats> {
    let mut writer = OutputWriter::create(&output.with_extension(format.extension()), format)?;
    let mut total = Stats::default();

    for (i, input) in inputs.iter().enumerate() {
        let file_start = Instant::now();

        // One file's games are held in memory at a time, then written out.
        let games = match input.read() {
            Ok(games) => games,
            Err(e) => {
                eprintln!(
                    "[{}/{}] {}: failed to read: {e:#}",
                    i + 1,
                    inputs.len(),
                    input.path().display()
                );
                total.files_failed += 1;
                continue;
            }
        };

        let stats = write_games(&mut writer, &games);
        report(i, inputs.len(), input, &stats, file_start.elapsed());
        total.merge(&stats);
    }

    writer.finish()?;
    Ok(total)
}

// reads one pgn and writes it to its own file inside `out_dir`
fn pack_one(input: &PgnInput, out_dir: &Path, format: OutputFormat) -> anyhow::Result<Stats> {
    let games = input.read()?;
    let path = input.output_path(out_dir, format)?;

    let mut writer = OutputWriter::create(&path, format)
        .with_context(|| format!("failed to create {}", path.display()))?;
    let stats = write_games(&mut writer, &games);
    writer.finish()?;
    Ok(stats)
}

// yay, threading
fn pack_parallel(
    files: &[PgnInput],
    out_dir: &Path,
    format: OutputFormat,
    threads: usize,
) -> Stats {
    let next = AtomicUsize::new(0);
    let next = &next;
    let worker_count = threads.min(files.len());

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);

        for _ in 0..worker_count {
            handles.push(scope.spawn(move || {
                let mut total = Stats::default();

                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(file) = files.get(i) else { break };
                    let file_start = Instant::now();

                    match pack_one(file, out_dir, format) {
                        Ok(stats) => {
                            report(i, files.len(), file, &stats, file_start.elapsed());
                            total.merge(&stats);
                        }
                        Err(e) => {
                            eprintln!(
                                "[{}/{}] {}: failed: {e:#}",
                                i + 1,
                                files.len(),
                                file.path().display()
                            );
                            total.files_failed += 1;
                        }
                    }
                }
                total
            }));
        }

        let mut total = Stats::default();
        for handle in handles {
            total.merge(&handle.join().expect("worker thread panicked"));
        }
        total
    })
}

fn pack(
    input: &Path,
    output: &Path,
    format: OutputFormat,
    as_one: bool,
    threads: usize,
) -> anyhow::Result<()> {
    let is_tar = input.is_file() && input.extension().is_some_and(|v| v == "tar");
    let blobs = if is_tar {
        read_tar_entries(input)?
    } else {
        Vec::new()
    };

    let inputs: Vec<PgnInput> = if is_tar {
        blobs
            .iter()
            .map(|(name, data)| PgnInput::Bz2 {
                name: name.clone(),
                data,
            })
            .collect()
    } else {
        collect_pgn_files(input)?
            .into_iter()
            .map(PgnInput::File)
            .collect()
    };

    anyhow::ensure!(
        !inputs.is_empty(),
        "no .pgn/.bz2 files found in {}",
        input.display()
    );

    let start = Instant::now();

    let total = if as_one {
        pack_shared(&inputs, output, format)?
    } else {
        std::fs::create_dir_all(output)
            .with_context(|| format!("failed to create folder {}", output.display()))?;
        println!("using {} thread(s)", threads.min(inputs.len()));
        pack_parallel(&inputs, output, format, threads)
    };

    let elapsed = start.elapsed();
    println!(
        "wrote {} games, {} positions to {}",
        total.games,
        total.positions,
        output.display()
    );
    println!(
        "{} games skipped, {} files failed",
        total.skipped_games, total.files_failed
    );
    println!(
        "total {:.2?} | {:.0} games/s, {:.0} positions/s",
        elapsed,
        rate(total.games, elapsed),
        rate(total.positions, elapsed),
    );
    Ok(())
}
