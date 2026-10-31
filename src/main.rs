use std::io::{self, Write};
use std::sync::OnceLock;
use std::time::Duration;

use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{Chess, Position};

use crate::history::MoveHistory;
use crate::nn::data;
use crate::options::EngineOptions;
use crate::search::SearchOptions;
use crate::search::threads::SearchThread;

mod bench;
mod history;
mod nn;
mod options;
mod pgn_reader;
mod search;
mod transposition_table;

pub fn engine_options() -> &'static EngineOptions {
    static ENGINE_OPTIONS: OnceLock<EngineOptions> = OnceLock::new();

    ENGINE_OPTIONS.get_or_init(EngineOptions::default)
}

fn main() -> Result<(), anyhow::Error> {
    let mut pos = Chess::default();

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut history = MoveHistory::new();

    let mut search_thread: Option<SearchThread> = None;

    // Flatten all arguments, including space-separated string arguments
    let args: Vec<String> = std::env::args()
        .skip(1)
        .flat_map(|s| s.split_whitespace().map(String::from).collect::<Vec<_>>())
        .collect();

    let tokens: Vec<&str> = args.iter().map(String::as_str).collect();
    if !tokens.is_empty() {
        match tokens.as_slice() {
            ["bench", ..] => return bench::bench(),
            ["speedtest", iters @ ..] => return bench::speedtest(iters.join("").parse().ok()),
            ["aspiration" | "aspirations", ..] => return bench::bench_windows(),
            ["genfens", args @ ..] => {
                let (n, seed, book_path, extra) = search::genfens::parse_genfens_args(args)?;
                search::genfens::generate_fens(n, seed, book_path, extra.as_deref())?;
                return Ok(());
            }
            ["packpgns", args @ ..] => {
                pgn_reader::cli::pack_files(args);
                return Ok(());
            }
            ["balls", ..] => {
                use crate::nn::trainer::TrainingSteps;

                let games = pgn_reader::read_pgn("./71.281.1.pgn")?;

                let mut trainer = nn::trainer::NetworkTrainer::new();
                trainer.config.training_steps = TrainingSteps {
                    batch_size: 1,
                    end_superbatch: 1,
                    batches_per_superbatch: 6000,
                    ..Default::default()
                };
                trainer.train(data::games_to_samples(games))?;

                return Ok(());
            }
            ["quit", ..] => return Ok(()),
            _ => {}
        };
    }

    loop {
        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }

        let start_time = minstant::Instant::now();

        let tokens = line.split_ascii_whitespace().collect::<Vec<_>>();

        if let Some(&cmd) = tokens.first()
            && !matches!(cmd, "stop" | "quit")
            && search_thread.as_ref().is_some_and(SearchThread::is_running)
        {
            continue; // basically ignoring anything that's not listed above
        }

        match tokens.as_slice() {
            ["uci", ..] => {
                println!("id name Bürgentruck {}", env!("CARGO_PKG_VERSION"));
                println!("id author ZackGabri & TampliteSiphronKents");
                println!();
                EngineOptions::print_defaults();

                println!("uciok");
            }
            ["setoption", args @ ..] => {
                if let ["name", args @ ..] = args {
                    let mut split = args.split(|x| *x == "value");
                    let name = split.next().unwrap_or(&[""]).join(" ");
                    let value = split.next().unwrap_or(&[""]).join(" ");

                    let result = engine_options().set(&name, &value);
                    if let Err(err) = result {
                        println!("{err:?}")
                    }

                    if &name == "Hash" {
                        transposition_table::resize();
                    }
                };
            }
            ["ucinewgame", ..] => {
                transposition_table::clear();
            }
            ["isready", ..] => println!("readyok"),

            ["position"] => {}
            ["position", pos_type, args @ ..] => match *pos_type {
                "startpos" => {
                    history.reset();
                    pos = Chess::default();

                    if args.first().copied() == Some("moves") {
                        let moves = args.get(1..).unwrap_or_default();

                        for m in moves {
                            let uci: UciMove = m.trim().parse()?;
                            let m = uci.to_move(&pos).unwrap();

                            pos.play_unchecked(m);
                            history.push_position(&pos);
                        }
                    }
                }
                "fen" => {
                    history.reset();

                    let (fen_args, moves) = match args.iter().position(|&s| s == "moves") {
                        Some(idx) => (&args[..idx], &args[idx + 1..]),
                        None => (args, &[][..]),
                    };

                    let fen_str = fen_args.join(" ");
                    let fen = fen_str.parse::<Fen>();

                    if fen.is_err() {
                        println!("invalid fen");
                        continue;
                    }

                    let mode = if engine_options().get_bool("UCI_Chess960") {
                        shakmaty::CastlingMode::Chess960
                    } else {
                        shakmaty::CastlingMode::Standard
                    };

                    pos = fen?.into_position(mode)?;
                    history.push_position(&pos);

                    for m in moves {
                        let uci: UciMove = m.trim().parse()?;
                        let m = uci.to_move(&pos)?;

                        pos.play_unchecked(m);
                        history.push_position(&pos);
                    }
                }
                _ => {}
            },

            ["go", "infinite", ..] => {
                let search_options = SearchOptions {
                    infinite: true,
                    ..Default::default()
                };

                search_thread = Some(SearchThread::spawn(&pos, &history, search_options)?);
            }
            ["go", "movetime", val] => {
                // Fixed time per move, set both hard and soft deadlines to the same value
                let search_options = SearchOptions {
                    hard_deadline: val
                        .parse()
                        .map(|val| minstant::Instant::now() + Duration::from_millis(val))
                        .ok(),
                    soft_deadline: val
                        .parse()
                        .map(|val| minstant::Instant::now() + Duration::from_millis(val))
                        .ok(),
                    ..Default::default()
                };

                search_thread = Some(SearchThread::spawn(&pos, &history, search_options)?);
            }
            ["go", args @ ..] => {
                let search_args = args.chunks(2);

                let mut time = search::TimeControl::default();
                let mut max_depth = None;
                let mut max_soft_nodes = None;
                let mut max_hard_nodes = None;

                for arg in search_args {
                    match arg {
                        ["nodes", nodes] => {
                            // By default go nodes <> is soft nodes
                            // We set hard modes as double as a cap
                            if let Ok(nodes) = nodes.parse::<u64>() {
                                max_soft_nodes = Some(nodes);
                                max_hard_nodes = Some(nodes * 2);
                            }
                        }
                        ["depth", depth] => max_depth = depth.parse().ok(),

                        ["wtime", wtime] => time.w_time = wtime.parse().unwrap_or_default(),
                        ["btime", btime] => time.b_time = btime.parse().unwrap_or_default(),
                        ["winc", inc] => time.w_inc = inc.parse().unwrap_or_default(),
                        ["binc", inc] => time.b_inc = inc.parse().unwrap_or_default(),
                        _ => {}
                    }
                }

                let search_options = SearchOptions {
                    soft_deadline: time.soft_deadline(start_time, &pos),
                    hard_deadline: time.hard_deadline(start_time, &pos),
                    max_depth,
                    max_soft_nodes,
                    max_hard_nodes,
                    ..Default::default()
                };

                search_thread = Some(SearchThread::spawn(&pos, &history, search_options)?);
            }
            ["stop", ..] => search::threads::stop_search(&mut search_thread),

            ["d" | "display", ..] => {
                let mut board: [[char; 8]; 8] = [['.'; 8]; 8];
                pos.board().iter().for_each(|(square, piece)| {
                    let (file, rank) = square.coords();
                    board[rank as usize][file as usize] = piece.char();
                });

                board.reverse();

                board.iter().for_each(|chunk| {
                    println!(
                        "{}",
                        chunk
                            .iter()
                            .map(|c| format!("{c}"))
                            .collect::<Vec<String>>()
                            .join(" ")
                    );
                });
                println!("Fen: {}", pos.board().board_fen());
            }
            ["bench", ..] => {
                if let Err(err) = bench::bench() {
                    println!("Bench Failed! {:?}", err);
                }
            }
            ["speedtest", iters @ ..] => {
                if let Err(err) = bench::speedtest(iters.join("").parse().ok()) {
                    println!("Bench Failed! {:?}", err);
                }
            }
            ["aspiration", ..] => {
                if let Err(err) = bench::bench_windows() {
                    println!("Bench Failed! {:?}", err);
                }
            }
            ["test", ..] => {
                println!("Features: {:?}", data::parse_board_to_features(&pos));
            }

            [] => {}
            ["quit", ..] => break,
            _ => println!("Unkown Command: {line}"),
        }

        stdout.flush()?;
    }

    search::threads::stop_search(&mut search_thread);
    Ok(())
}
