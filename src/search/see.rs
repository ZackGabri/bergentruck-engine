use shakmaty::Color::{self, Black, White};
use shakmaty::{Bitboard, Board, Chess, Move, Position, Rank, Role, Square, attacks};

// The default shakmaty attacks_to is based on color and it would be inefficient to call it twice
#[inline]
fn attackers_to(board: &Board, square: Square, occupancies: Bitboard) -> Bitboard {
    let white_board = board.by_color(White);
    let black_board = board.by_color(Black);

    let diagonals = board.bishops() | board.queens();
    let orthogonals = board.rooks() | board.queens();

    let white_pawn_attacks = attacks::pawn_attacks(Black, square) & (board.pawns() & white_board);
    let black_pawn_attacks = attacks::pawn_attacks(White, square) & (board.pawns() & black_board);
    let knight_attacks = attacks::knight_attacks(square) & board.knights();
    let diagonal_attacks = attacks::bishop_attacks(square, occupancies) & diagonals;
    let orthogonal_attacks = attacks::rook_attacks(square, occupancies) & orthogonals;
    let king_attacks = attacks::king_attacks(square) & board.kings();

    white_pawn_attacks
        | black_pawn_attacks
        | knight_attacks
        | diagonal_attacks
        | orthogonal_attacks
        | king_attacks
}

#[inline]
fn diagonal_xrays(board: &Board, occupied: Bitboard, to_sq: Square) -> Bitboard {
    attacks::bishop_attacks(to_sq, occupied) & (board.bishops() | board.queens()) & occupied
}

#[inline]
fn orthogonal_xrays(board: &Board, occupied: Bitboard, to_sq: Square) -> Bitboard {
    attacks::rook_attacks(to_sq, occupied) & (board.rooks() | board.queens()) & occupied
}

#[inline]
fn piece_value(role: Role) -> i32 {
    #[cfg(test)]
    return [100, 300, 300, 500, 900, 0][role as usize - 1];

    #[cfg(not(test))]
    return super::eval::PIECE_VALUES_MIDDLEGAME[role as usize - 1];
}

#[inline]
fn get_least_valuable_piece(
    board: &Board,
    attadef: Bitboard,
    color: Color,
) -> Option<(Square, Role)> {
    let attackers = attadef & board.by_color(color);
    if attackers.is_empty() {
        return None;
    }

    let (role, subset) = if !(attackers & board.pawns()).is_empty() {
        (Role::Pawn, board.pawns())
    } else if !(attackers & board.knights()).is_empty() {
        (Role::Knight, board.knights())
    } else if !(attackers & board.bishops()).is_empty() {
        (Role::Bishop, board.bishops())
    } else if !(attackers & board.rooks()).is_empty() {
        (Role::Rook, board.rooks())
    } else if !(attackers & board.queens()).is_empty() {
        (Role::Queen, board.queens())
    } else if !(attackers & board.kings()).is_empty() {
        (Role::King, board.kings())
    } else {
        return None;
    };

    let sq = (attackers & subset).first()?;
    Some((sq, role))
}

pub fn see(position: &Chess, mov: &Move, threshold: i32) -> bool {
    let board = position.board();
    let from_sq = mov.from().unwrap();
    let to_sq = mov.to();
    let us = position.turn();

    let promo_square = matches!(to_sq.rank(), Rank::First | Rank::Eighth);
    let promo_bonus = piece_value(Role::Queen) - piece_value(Role::Pawn);

    let promo_gain = |color: Color, candidates: Bitboard| -> i32 {
        if promo_square && (candidates & board.pawns() & board.by_color(color)).any() {
            promo_bonus
        } else {
            0
        }
    };

    let mut balance = mov.capture().map_or(0, piece_value)
        + mov.promotion().map_or(0, |promotion| {
            piece_value(promotion) - piece_value(Role::Pawn)
        })
        - threshold;

    if balance < 0 {
        return false;
    }

    let victim_value = piece_value(mov.promotion().unwrap_or(mov.role()))
        + promo_gain(us.other(), attacks::pawn_attacks(us, to_sq));
    balance -= victim_value;

    if balance >= 0 {
        return true;
    }

    let mut occupied = board.occupied() ^ Bitboard::from_square(from_sq);

    if mov.is_en_passant() {
        let ep_square = Square::from_coords(to_sq.file(), from_sq.rank());
        occupied ^= Bitboard::from_square(ep_square);
    }

    let mut attackers = attackers_to(board, to_sq, occupied) & occupied;
    let mut stm = us.other();

    while let Some((sq, attacker)) = get_least_valuable_piece(board, attackers, stm) {
        if attacker == Role::King && (attackers & board.by_color(stm.other())).any() {
            break;
        }

        // Make the capture
        occupied ^= Bitboard::from_square(sq);
        stm = stm.other();

        let exposed = if attacker == Role::Pawn && promo_square {
            piece_value(Role::Queen)
        } else {
            piece_value(attacker)
        } + promo_gain(stm, attackers);

        balance = -balance - 1 - exposed;
        if balance >= 0 {
            break;
        }

        if !matches!(attacker, Role::Knight | Role::King) {
            attackers |= if sq.file() == to_sq.file() || sq.rank() == to_sq.rank() {
                orthogonal_xrays(board, occupied, to_sq)
            } else {
                diagonal_xrays(board, occupied, to_sq)
            };
        }
        attackers &= occupied;
    }

    stm != us
}
