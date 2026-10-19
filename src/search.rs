use crate::{history::MoveHistory, search::negamax::PVariation, transposition_table};
use negamax::Negamax;

use shakmaty::{Chess, Color, Move, Position};
use std::time::Duration;

use std::sync::atomic::{AtomicUsize, Ordering};

pub mod eval;
pub mod negamax;
pub mod see;
pub mod genfens;

#[cfg(test)]
mod tests;

pub static FAIL_LOWS: AtomicUsize = AtomicUsize::new(0);
pub static FAIL_HIGHS: AtomicUsize = AtomicUsize::new(0);
pub static ASP_ITERS: AtomicUsize = AtomicUsize::new(0);

pub const MATE_SCORE: i32 = 100_000;
pub const MATE_THRESHOLD: i32 = MATE_SCORE - 1000;
pub const MAX_PLY: usize = 128;

// Helper function to print search info
fn print_info(
    depth: usize,
    best_score: i32,
    duration: Duration,
    negamax: &Negamax,
    pv: &PVariation,
) {
    let Negamax {
        node_count,
        seldepth,
        ..
    } = negamax;

    let hashfull = transposition_table::hashfull();
    let nps = (*node_count as f64 / duration.as_secs_f64()) as usize;
    let time_ms = duration.as_millis();

    let score_string = if best_score.abs() >= MATE_THRESHOLD {
        // safe mate detection range
        let mate_in = if best_score > 0 {
            (MATE_SCORE - best_score + 1) / 2
        } else {
            -(MATE_SCORE + best_score + 1) / 2
        };
        format!("mate {mate_in}")
    } else {
        format!("cp {best_score}")
    };

    println!(
        "info depth {depth} seldepth {seldepth} score {score_string} nodes {node_count} nps {nps} hashfull {hashfull} time {time_ms} pv {pv}"
    );
}

#[derive(Default)]
pub struct SearchOptions {
    pub max_depth: Option<usize>,
    pub max_hard_nodes: Option<u64>,
    pub max_soft_nodes: Option<u64>,
    pub hard_deadline: Option<minstant::Instant>,
    pub soft_deadline: Option<minstant::Instant>,
    pub bench: bool,
}

pub fn search(
    position: &Chess,
    history: Option<&mut MoveHistory>,
    search_options: SearchOptions,
) -> (Option<Move>, u64) {
    let SearchOptions {
        max_depth,
        max_hard_nodes,
        max_soft_nodes,
        hard_deadline,
        soft_deadline,
        bench,
    } = search_options;

    let mut negamax = Negamax::new();
    negamax.max_hard_nodes = max_hard_nodes;
    negamax.hard_deadline = hard_deadline;

    let max_depth = max_depth.unwrap_or_else(|| {
        if max_hard_nodes.is_some() || hard_deadline.is_some() {
            100
        } else {
            crate::engine_options().get_number("Default Depth")
        }
    });

    let start = minstant::Instant::now();
    let (search_limit, is_node_search) = if let Some(soft_nodes) = max_soft_nodes {
        // Node search
        (soft_nodes, true)
    } else {
        // Time search
        let limit_ms = soft_deadline.map_or(u64::MAX, |deadline| {
            deadline.duration_since(start).as_millis() as u64
        });
        (limit_ms, false)
    };

    let mut default_history = MoveHistory::default();
    let history = history.unwrap_or(&mut default_history);

    // aspiration window variables
    let mut alpha = -MATE_SCORE;
    let mut beta = MATE_SCORE;
    const WINDOW_SIZE: i32 = 18;
    let mut delta = WINDOW_SIZE;

    let mut depth = 1;

    let mut pv = PVariation::default();
    while depth <= max_depth {
        let mut search_progress = if is_node_search {
            negamax.node_count
        } else {
            start.elapsed().as_millis() as u64
        };

        // Check soft / hard limits
        if (search_progress >= search_limit || negamax.has_reached_limit()) && depth > 1 {
            break;
        }

        let mut iter_pv = PVariation::default(); // don't reuse pv from previous iteration, since it 
        // may be invalid if the search fails
        let score = negamax.negamax(position, history, depth, 0, alpha, beta, &mut iter_pv, true);

        if negamax.has_reached_limit() {
            if depth == 1 {
                pv = iter_pv;
            }
            break;
        }

        // Aspiration windows
        if depth > 4 {
            ASP_ITERS.fetch_add(1, Ordering::Relaxed);
            if score <= alpha {
                beta = (alpha + beta) / 2;
                alpha = std::cmp::max(-MATE_SCORE, alpha - delta);
                delta += delta / 2;
                FAIL_LOWS.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if score >= beta {
                // fail high still has vaild best move but rest of pv is invalid
                iter_pv.length = 1;
                pv = iter_pv;

                if !bench {
                    let duration = start.elapsed();
                    print_info(depth, score, duration, &negamax, &pv);
                }
                beta = std::cmp::min(MATE_SCORE, beta + delta);
                delta += delta / 2;
                FAIL_HIGHS.fetch_add(1, Ordering::Relaxed);

                // print info after updating pv so it doesn't mismatch the output bestmove if we're cut off before finishing
                if !bench {
                    let duration = start.elapsed();
                    print_info(depth, score, duration, &negamax, &pv);
                }

                continue;
            }

            // Reset parameters if search passes
            delta = WINDOW_SIZE;
            alpha = std::cmp::max(-MATE_SCORE, score - WINDOW_SIZE);
            beta = std::cmp::min(score + WINDOW_SIZE, MATE_SCORE);
        }

        pv = iter_pv; // search passes, so we can update the pv

        if !bench {
            let duration = start.elapsed();
            print_info(depth, score, duration, &negamax, &pv);
        }

        search_progress = if is_node_search {
            negamax.node_count
        } else {
            start.elapsed().as_millis() as u64
        };

        // Check soft / hard limits
        if (search_progress >= search_limit || negamax.has_reached_limit()) && depth > 1 {
            break;
        }

        depth += 1;
    }

    // if negamax failed for whatever reason, then just play the first move in the position
    if pv.line[0].is_none() {
        pv.line[0] = position.legal_moves().first().copied();
        // just print invalid pv so engines can warn or smth
        println!("info depth 0 score cp 0 nodes 0 nps 0 hashfull -1 time -1 pv e2e2 e2e1 e1e2");
    }

    (pv.line[0], negamax.node_count)
}

#[derive(Default, Debug)]
pub struct TimeControl {
    pub w_time: u64,
    pub b_time: u64,
    pub w_inc: u64,
    pub b_inc: u64,
}

impl TimeControl {
    pub fn soft_deadline(
        &self,
        start_time: minstant::Instant,
        position: &Chess,
    ) -> Option<minstant::Instant> {
        let turn = position.turn();
        let played_moves = position.fullmoves().get() as u64;

        let divisor = 30.max(60_u64.saturating_sub(played_moves));

        let duration = match turn {
            Color::Black if self.b_time > 0 => (self.b_time / divisor) + (self.b_inc / 2),
            Color::White if self.w_time > 0 => (self.w_time / divisor) + (self.w_inc / 2),
            _ => 0,
        };

        if duration > 0 {
            Some(start_time + Duration::from_millis(duration))
        } else {
            None
        }
    }

    pub fn hard_deadline(
        &self,
        start_time: minstant::Instant,
        position: &Chess,
    ) -> Option<minstant::Instant> {
        let turn = position.turn();
        let duration = match turn {
            Color::Black if self.b_time > 0 => (self.b_time / 10) + self.b_inc,
            Color::White if self.w_time > 0 => (self.w_time / 10) + self.w_inc,
            _ => 0,
        };

        if duration > 0 {
            Some(start_time + Duration::from_millis(duration))
        } else {
            None
        }
    }
}
