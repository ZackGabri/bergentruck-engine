use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use shakmaty::{Chess, Move, MoveList, Position};

use crate::nn::network;
use crate::search::repetition::RepetitionTable;
use crate::search::{MATE_SCORE, MATE_THRESHOLD, MAX_PLY};
use crate::transposition_table::{self, TTBound, TTEntry};

const MAX_HISTORY: i32 = 16_384;
const CONT_HISTORY_PLIES: [usize; 3] = [1, 2, 4];
const CONT_HISTORY_SIZE: usize = CONT_HISTORY_PLIES.len();

pub struct PVariation {
    pub length: usize,
    pub line: [Option<Move>; MAX_PLY],
}

impl Default for PVariation {
    fn default() -> Self {
        Self {
            length: 0,
            line: [None; MAX_PLY],
        }
    }
}

impl std::fmt::Display for PVariation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output = String::new();
        let mode = if crate::engine_options().get_bool("UCI_Chess960") {
            shakmaty::CastlingMode::Chess960
        } else {
            shakmaty::CastlingMode::Standard
        };

        for i in 0..self.length {
            if let Some(m) = self.line[i] {
                output += &m.to_uci(mode).to_string();

                // add spaces between unless it's last move
                if i != self.length - 1 {
                    output += " ";
                }
            }
        }

        write!(f, "{output}")
    }
}

#[derive(Default, Clone, Copy)]
struct KillerMoves {
    moves: [Option<Move>; 2],
}

impl KillerMoves {
    #[inline(always)]
    fn insert_move(&mut self, m: Move) {
        if self.moves[0] == Some(m) {
            return;
        }
        self.moves[1] = self.moves[0];
        self.moves[0] = Some(m);
    }

    #[inline(always)]
    fn contains_move(&self, m: Move) -> bool {
        self.moves[0] == Some(m) || self.moves[1] == Some(m)
    }
}

pub const MAX_PSUEDO_MOVES: usize = 280;

// Late Move Reductions table
// 3D LMR table: [improving][depth][move_num]
static LMR_TABLE: OnceLock<[[[u8; MAX_PSUEDO_MOVES]; MAX_PLY]; 2]> = OnceLock::new();
fn get_lmr_table() -> &'static [[[u8; MAX_PSUEDO_MOVES]; MAX_PLY]; 2] {
    LMR_TABLE.get_or_init(|| {
        let mut table = [[[0; MAX_PSUEDO_MOVES]; MAX_PLY]; 2];

        let mut depth = 1;
        while depth < MAX_PLY {
            let mut move_num = 1;
            while move_num < MAX_PSUEDO_MOVES {
                let base = 0.50 + (depth as f64).ln() * (move_num as f64).ln() / 3.00;

                // reduce more when not improving
                table[0][depth][move_num] = (base + 0.5).round() as u8;

                // reduce less when improving
                table[1][depth][move_num] = (base - 0.5).round() as u8;

                move_num += 1;
            }
            depth += 1;
        }

        table
    })
}

#[derive(Clone, Copy, Default)]
pub struct ImprovingStackEntry {
    pub static_eval: Option<i32>,
    pub improving: bool,
}

// struct for shared data between every negamax call
pub struct Negamax {
    pub node_count: u64,
    pub repetition: RepetitionTable,

    /// Usage: `quiet_history[color][from][to]`
    quiet_history: [[[i32; 64]; 64]; 2],
    /// Usage: `cont_history[ply][color][previous_piece][previous_to][current_piece][current_to]`
    /// index
    cont_history: Box<[[[[[[i32; 64]; 6]; 64]; 6]; 2]; CONT_HISTORY_SIZE]>,
    move_stack: [Option<Move>; MAX_PLY],

    improving_table: [ImprovingStackEntry; MAX_PLY],
    killer_move_table: [KillerMoves; MAX_PLY],
    accumulators: Box<[[network::Accumulator; 2]; MAX_PLY]>,

    pub max_hard_nodes: Option<u64>, // maximum number of nodes to search
    pub hard_deadline: Option<minstant::Instant>, // deadline for the search to stop at
    pub stop: Arc<AtomicBool>,
    pub seldepth: usize,
}

impl Negamax {
    pub fn new() -> Self {
        Self {
            node_count: 0,
            repetition: RepetitionTable::new(),

            quiet_history: [[[0; 64]; 64]; 2],
            cont_history: unsafe { Box::new_zeroed().assume_init() },
            move_stack: [None; MAX_PLY],

            improving_table: [ImprovingStackEntry::default(); MAX_PLY],
            killer_move_table: [KillerMoves::default(); MAX_PLY],
            accumulators: Box::new([[network::Accumulator::default(); 2]; MAX_PLY]),

            max_hard_nodes: None,
            hard_deadline: None,
            stop: Arc::default(),
            seldepth: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn negamax(
        &mut self,
        position: &Chess,
        mut depth: usize,
        ply: usize,
        mut alpha: i32,
        beta: i32,
        pv: &mut PVariation,
        pv_node: bool,
    ) -> i32 {
        if ply == 0 {
            self.accumulators[0] = network::refresh(position);
        }

        let in_check = position.is_check();
        let is_root = ply == 0;

        // check extension
        if in_check {
            depth += 1;
        }

        if depth == 0 {
            return self.quiescence(position, &mut alpha, beta, ply, pv_node);
        }

        if depth > 1 && self.has_reached_limit() {
            return -alpha;
        }

        if ply >= MAX_PLY - 1 {
            return self.static_eval(position, ply);
        }

        if ply > self.seldepth {
            self.seldepth = ply;
        }

        let original_alpha = alpha;
        let hash = position.zobrist_hash(shakmaty::EnPassantMode::Legal);
        let halfmoves = position.halfmoves();

        if (self.repetition.is_repetition(ply, &hash, halfmoves) && ply > 0)
            || position.is_insufficient_material()
            || halfmoves >= 100
        {
            return 0;
        }

        // push current position hash before exploring children
        self.repetition.set_ply(ply, hash);

        let tt_entry = transposition_table::get(hash);
        let tt_hit = tt_entry.hash == hash;

        pv.length = 0; // ensure fresh pv
        // tt probing and cutoffs
        if tt_entry.depth >= depth && tt_hit {
            match tt_entry.bound {
                TTBound::Exact => {
                    pv.line[pv.length] = tt_entry.best_move;
                    pv.length += 1;
                    return tt_entry.score;
                }
                TTBound::Lower => {
                    if tt_entry.score >= beta {
                        pv.line[pv.length] = tt_entry.best_move;
                        pv.length += 1;
                        return tt_entry.score;
                    }
                }
                TTBound::Upper => {
                    if tt_entry.score < alpha {
                        pv.line[pv.length] = tt_entry.best_move;
                        pv.length += 1;
                        return tt_entry.score;
                    }
                }
            }
        }

        let mut static_eval = if in_check {
            self.improving_table[ply].improving = false;
            self.improving_table[ply].static_eval = None;
            None
        } else {
            let eval = self.static_eval(position, ply);
            self.improving_table[ply].static_eval = Some(eval);

            let past_eval = if ply >= 2 && self.improving_table[ply - 2].static_eval.is_some() {
                self.improving_table[ply - 2].static_eval
            } else if ply >= 4 {
                self.improving_table[ply - 4].static_eval
            } else {
                None
            };

            let improving = match past_eval {
                Some(past) => eval > past,
                None => false,
            };

            self.improving_table[ply].improving = improving;
            Some(eval)
        };

        let improving = self.improving_table[ply].improving;
        let turn = position.turn();

        // whole node pruning
        if !in_check && !is_root {
            let static_eval = *static_eval.get_or_insert(self.static_eval(position, ply));

            // razoring
            if depth <= 7 && !pv_node {
                let margin = 322 * (depth as i32);
                if static_eval + margin < alpha {
                    let razor_score = self.quiescence(position, &mut alpha, beta, ply,false);
                    if razor_score <= alpha {
                        return razor_score;
                    }
                }
            }

            // reverse futility pruning
            if depth <= 4 {
                let margin = if improving { 50 * depth } else { 80 * depth };
                if static_eval >= beta + margin as i32 {
                    return static_eval;
                }
            }

            // null move pruning
            if depth >= 3 {
                // Make sure it's not king and pawn endgame
                let board = position.board();
                let non_pawns = (board.by_color(turn) & (!board.pawns())) & (!board.kings());

                if non_pawns.any() && static_eval >= beta {
                    self.move_stack[ply] = None;

                    let null_position = position.clone().swap_turn().unwrap();
                    self.accumulators[ply + 1] = self.accumulators[ply];
                    // Ethereal tech to scale NMP reduction based on eval - beta.
                    let eval_reduction = ((static_eval - beta) / 191).min(3) as usize; // capped at 3 extra ply
                    let reduction = 3 + depth / 3 + eval_reduction;
                    let null_score = -self.negamax(
                        // search with zero window
                        &null_position,
                        depth.saturating_sub(reduction),
                        ply + 1,
                        -beta,
                        -beta + 1,
                        pv,
                        false,
                    );

                    // Make sure it's not a mate score
                    if null_score >= beta && null_score.abs() < MATE_THRESHOLD {
                        return null_score;
                    }
                }
            }
        }

        // Internal iterative reductions (IIR)
        // If the position has not been searched yet (i.e. no hash move), we try searching with reduced
        // depth to record a move that we can later re-use.
        if depth >= 6 && !in_check && pv_node && (!tt_hit || tt_entry.depth <= depth - 5) {
            depth -= 1;
        }

        let mut max = -MATE_SCORE;

        let moves = self.get_sorted_moves(position, ply, &tt_entry.best_move);
        if moves.is_empty() {
            if in_check {
                // checkmate
                return -MATE_SCORE + ply as i32;
            } else {
                // stalemate
                return 0;
            }
        }

        let mut best_move = None;
        let mut child_pv = PVariation::default();
        let mut quiets_tried: Vec<Move> = Vec::with_capacity(32);

        for (move_index, (_, mov, passed_0see)) in moves.into_iter().enumerate() {
            if depth > 1 && self.has_reached_limit() {
                return alpha;
            }

            let is_mate = max.abs() >= crate::search::MATE_THRESHOLD;
            let is_quiet = !mov.is_capture() && !mov.is_promotion();
            let is_killer = self.killer_move_table[ply].contains_move(mov);

            // Move loop pruning
            if !is_root && !pv_node && !in_check && !is_mate {
                // Futility pruning
                /*
                if depth <= self.fp_depth && move_index >= self.fp_moves {
                    let static_eval = static_eval.get_or_insert(self.static_eval(position));

                    // Discard moves with no potential to raise alpha
                    if *static_eval + (self.fp_base + self.fp_multiplier * depth) as i32 <= alpha {
                        continue;
                    }
                }
                */

                // Late move pruning (LMP)
                if is_quiet && !is_killer {
                    let mut lmp_threshold = 5 + 4 * depth * depth;

                    if !improving {
                        lmp_threshold -= 2 * depth;
                    } else {
                        lmp_threshold += 2 * depth;
                    }

                    if move_index >= lmp_threshold {
                        continue;
                    }
                }

                // SEE Pruning
                if move_index > 0 && depth <= 6 {
                    let depth = depth as i32;

                    let see_threshold = if is_quiet {
                        (-16 * depth * depth + 53 * depth + 27).min(0)
                    } else {
                        (-8 * depth * depth - 37 * depth + 10).min(0)
                    };

                    // minimum threshold is 0 anyway, if it already failed to pass it then no point continuing
                    if let Some(passed) = passed_0see
                        && !passed
                        && see_threshold == 0
                    {
                        continue;
                    }

                    if !super::see::see(position, &mov, see_threshold) {
                        continue;
                    }
                }
            }

            let mut position = position.clone();
            position.play_unchecked(mov);
            self.move_stack[ply] = Some(mov);

            self.accumulators[ply + 1] = self.accumulators[ply];
            network::apply_move(&mut self.accumulators[ply + 1], &mov, turn);

            self.node_count += 1;

            let mut score = -MATE_SCORE;
            child_pv.length = 0;

            // Principal variation search (PVS)
            // If we are in a non-PV node, OR we are in a PV-node examining moves after the 1st legal move
            if !pv_node || move_index > 0 {
                // Late move reductions (LMR)
                let lmr_conditions = !in_check
                    && depth >= 3
                    && move_index >= 3
                    && is_quiet
                    && !self.killer_move_table[ply].contains_move(mov);

                let lmr_reduction = if lmr_conditions {
                    get_lmr_table()[improving as usize][depth][move_index]
                } else {
                    0
                } as usize;

                // Perform zero-window search (ZWS) on non-PV nodes
                score = -self.negamax(
                    &position,
                    depth.saturating_sub(lmr_reduction + 1),
                    ply + 1,
                    -alpha - 1,
                    -alpha,
                    &mut child_pv,
                    false,
                );

                // if LMR fails high then do a full search instead
                if lmr_reduction > 0 && score > alpha {
                    score = -self.negamax(
                        &position,
                        depth - 1,
                        ply + 1,
                        -alpha - 1,
                        -alpha,
                        &mut child_pv,
                        false,
                    );
                }
            }

            // We are in a PV node and either it's the first legal move, OR the ZWS failed high
            if pv_node && (move_index == 0 || score > alpha) {
                score = -self.negamax(
                    &position,
                    depth - 1,
                    ply + 1,
                    -beta,
                    -alpha,
                    &mut child_pv,
                    true,
                );
            }

            if score > max {
                max = score;
                best_move = Some(mov);

                if score > alpha {
                    pv.length = 1 + child_pv.length;
                    pv.line[0] = Some(mov);
                    pv.line[1..pv.length + 1].copy_from_slice(&child_pv.line[0..pv.length]);

                    alpha = score;
                }
            }

            if score >= beta {
                // store quiet moves
                if is_quiet {
                    let bonus = (depth * depth) as i32;
                    let cont_bonus = (150 * depth - 125) as i32;

                    self.update_history(turn, &mov, bonus);
                    self.update_cont_history(turn, ply, &mov, cont_bonus);
                    self.store_killer_move(ply, mov);

                    // history maluses
                    for q in &quiets_tried {
                        self.update_history(turn, q, -bonus);
                        self.update_cont_history(turn, ply, q, -cont_bonus);
                    }
                }

                break;
            }

            if is_quiet {
                quiets_tried.push(mov);
            }
        }

        let bound = if max <= original_alpha {
            TTBound::Upper
        } else if max >= beta {
            TTBound::Lower
        } else {
            TTBound::Exact
        };

        transposition_table::put(
            hash,
            TTEntry {
                hash,
                depth,
                bound,
                best_move,
                score: max,
                generation: transposition_table::get_age(),
            },
        );

        max
    }

    #[inline(always)]
    fn gravity(entry: &mut i32, bonus: i32) {
        let clamped_bonus = bonus.clamp(-MAX_HISTORY, MAX_HISTORY);

        *entry += clamped_bonus - *entry * clamped_bonus.abs() / MAX_HISTORY;
    }

    #[inline(always)]
    fn update_history(&mut self, color: shakmaty::Color, mov: &Move, bonus: i32) {
        let from = mov.from().unwrap() as usize;
        let to = mov.to() as usize;

        let entry = &mut self.quiet_history[color as usize][from][to];
        Self::gravity(entry, bonus);
    }

    #[inline(always)]
    fn update_cont_history(&mut self, color: shakmaty::Color, ply: usize, mov: &Move, bonus: i32) {
        let cur_piece = mov.role() as usize - 1;
        let cur_to = mov.to() as usize;
        let color = color as usize;

        for (i, &offset) in CONT_HISTORY_PLIES.iter().enumerate() {
            if ply < offset {
                continue;
            }

            let Some(prev) = self.move_stack[ply - offset] else {
                continue;
            };

            let prev_piece = prev.role() as usize - 1;
            let prev_to = prev.to() as usize;

            let entry = &mut self.cont_history[i][color][prev_piece][prev_to][cur_piece][cur_to];
            Self::gravity(entry, bonus);
        }
    }

    #[inline(always)]
    fn history_score(&self, turn: usize, ply: usize, mov: &Move) -> i32 {
        let from = mov.from().unwrap() as usize;
        let to = mov.to() as usize;
        let cur_piece = mov.role() as usize - 1;

        let mut score = self.quiet_history[turn][from][to];

        for (i, &offset) in CONT_HISTORY_PLIES.iter().enumerate() {
            if ply < offset {
                continue;
            }

            let Some(prev) = self.move_stack[ply - offset] else {
                continue;
            };

            let prev_piece = prev.role() as usize - 1;
            let prev_to = prev.to() as usize;

            score += self.cont_history[i][turn][prev_piece][prev_to][cur_piece][to];
        }

        score
    }

    #[inline(always)]
    fn static_eval(&mut self, position: &Chess, ply: usize) -> i32 {
        // super::eval::evaluate(position)
        network::evaluate(&self.accumulators[ply], position.turn())
    }

    pub fn has_reached_limit(&self) -> bool {
        if self.stop.load(Ordering::Relaxed) {
            return true;
        }

        // Check node limit
        if let Some(hard_nodes) = self.max_hard_nodes
            && self.node_count >= hard_nodes
        {
            return true;
        }

        // Check deadline
        if let Some(hard_deadline) = self.hard_deadline
            && minstant::Instant::now() >= hard_deadline
        {
            return true;
        }

        false
    }

    fn quiescence(
        &mut self,
        position: &Chess,
        alpha: &mut i32,
        beta: i32,
        ply: usize,
        pv_node: bool,
    ) -> i32 {
        let static_eval = self.static_eval(position, ply);

        if self.has_reached_limit() {
            return static_eval;
        }

        if ply >= MAX_PLY - 1 {
            return self.static_eval(position, ply);
        }

        if ply > self.seldepth {
            self.seldepth = ply;
        }

        let original_alpha = *alpha;

        // TT probe
        let hash = position.zobrist_hash(shakmaty::EnPassantMode::Legal);
        let tt_entry = transposition_table::get(hash);
        let tt_hit = tt_entry.hash == hash;

        if tt_hit && !pv_node && tt_entry.score.abs() < MATE_THRESHOLD {
            match tt_entry.bound {
                TTBound::Exact => return tt_entry.score,
                TTBound::Lower if tt_entry.score >= beta => return tt_entry.score,
                TTBound::Upper if tt_entry.score <= *alpha => return tt_entry.score,
                _ => {}
            }
        }

        let mut best_value = static_eval;
        if best_value >= beta {
            return best_value;
        }
        if best_value > *alpha {
            *alpha = best_value;
        }

        let mut captures = position.capture_moves();
        self.sort_captures(&mut captures);

        let mut best_move = None;

        for capture in captures {
            // SEE Pruning
            if !super::see::see(position, &capture, 0) {
                continue;
            }

            let mut position = position.clone();
            self.accumulators[ply + 1] = self.accumulators[ply];
            network::apply_move(&mut self.accumulators[ply + 1], &capture, position.turn());
            position.play_unchecked(capture);

            let score = -self.quiescence(&position, &mut -beta, -*alpha, ply + 1, pv_node);
            self.node_count += 1;

            if score > best_value {
                best_value = score;
                best_move = Some(capture);
            }

            if score > *alpha {
                *alpha = score;
            }

            if score >= beta {
                break;
            }
        }

        // TT store (depth 0 = quiescence)
        let bound = if best_value <= original_alpha {
            TTBound::Upper
        } else if best_value >= beta {
            TTBound::Lower
        } else {
            TTBound::Exact
        };

        transposition_table::put(
            hash,
            TTEntry {
                hash,
                depth: 0,
                bound,
                best_move,
                score: best_value,
                generation: transposition_table::get_age(),
            },
        );

        best_value
    }

    fn get_sorted_moves(
        &self,
        position: &Chess,
        ply: usize,
        hash_move: &Option<Move>,
    ) -> Vec<(i32, Move, Option<bool>)> {
        let move_list = position.legal_moves();
        let turn = position.turn() as usize;
        let killers = self.killer_move_table[ply];

        let mut scored: Vec<(i32, Move, Option<bool>)> = Vec::with_capacity(move_list.len());
        for m in move_list {
            let mut above_0_see = None;
            let score = if Some(m) == *hash_move {
                2_000_000_000 // Hash move is highest priority
            } else if let Some(victim) = m.capture() {
                let mvv_lva = super::eval::MVV_LVA[victim as usize - 1][m.role() as usize - 1];
                if !super::see::see(position, &m, 0) {
                    above_0_see = Some(false);
                    -1_000_000_000 + mvv_lva // after quiets
                } else {
                    above_0_see = Some(true);
                    1_000_000_000 + mvv_lva
                }
            } else if killers.contains_move(m) {
                900_000_000 // Same value as MVV-LVA so it's above bad captures but below good ones
            } else {
                // History moves
                self.history_score(turn, ply, &m)
            };

            scored.push((score, m, above_0_see));
        }

        // sort the scores
        scored.sort_unstable_by_key(|&(s, _, _)| -s);
        scored
    }

    fn sort_captures(&self, moves: &mut MoveList) {
        moves.sort_unstable_by_key(|m| {
            if let Some(victim) = m.capture() {
                -super::eval::MVV_LVA[victim as usize - 1][m.role() as usize - 1]
            } else {
                100000 // aka show all normal moves last
            }
        });
    }

    fn store_killer_move(&mut self, ply: usize, m: Move) {
        if !self.killer_move_table[ply].contains_move(m) {
            self.killer_move_table[ply].insert_move(m);
        }
    }
}
