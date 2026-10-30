use anyhow::{Context, bail};
use sfbinpack::{
    CompressedTrainingDataEntryWriter, TrainingDataEntry,
    chess::{
        castling_rights::CastlingRights as SfCastlingRights, color::Color as SfColor,
        coords::Square as SfSquare, r#move::Move as SfMove, piece::Piece as SfPiece,
        piecetype::PieceType as SfPieceType, position::Position as SfPosition,
    },
};
use shakmaty::{CastlingSide, Chess, Color, EnPassantMode, File, Move, Position, Role};

use std::io::Write;

use super::GameRecord;

fn score_to_i16(score: i32) -> i16 {
    score.clamp(-31_000, 31_000) as i16
}

fn convert_square(sq: shakmaty::Square) -> SfSquare {
    SfSquare::new(sq as u32)
}

fn convert_color(color: Color) -> SfColor {
    match color {
        Color::White => SfColor::White,
        Color::Black => SfColor::Black,
    }
}

fn convert_role(role: Role) -> SfPieceType {
    match role {
        Role::Pawn => SfPieceType::Pawn,
        Role::Knight => SfPieceType::Knight,
        Role::Bishop => SfPieceType::Bishop,
        Role::Rook => SfPieceType::Rook,
        Role::Queen => SfPieceType::Queen,
        Role::King => SfPieceType::King,
    }
}

fn convert_move(mv: &Move, pos: &Chess) -> anyhow::Result<SfMove> {
    Ok(match mv {
        Move::Normal {
            from,
            to,
            promotion,
            ..
        } => {
            let from = convert_square(*from);
            let to = convert_square(*to);

            match promotion {
                Some(role) => SfMove::promotion(
                    from,
                    to,
                    SfPiece::new(convert_role(*role), convert_color(pos.turn())),
                ),
                None => SfMove::normal(from, to),
            }
        }

        Move::EnPassant { from, to } => {
            SfMove::en_passant(convert_square(*from), convert_square(*to))
        }

        Move::Castle { king, rook } => {
            // sfbinpack encodes castling as king captures rook
            SfMove::castle(convert_square(*king), convert_square(*rook))
        }

        Move::Put { .. } => bail!("Unsupported drop move"),
    })
}

fn convert_castling_rights(pos: &Chess) -> anyhow::Result<SfCastlingRights> {
    let castles = pos.castles();
    let mut rights = SfCastlingRights::NONE;

    for color in [Color::White, Color::Black] {
        for side in [CastlingSide::KingSide, CastlingSide::QueenSide] {
            // a rook square exists only when this castling right is present
            let Some(rook) = castles.rook(color, side) else {
                continue;
            };

            let king = pos
                .board()
                .king_of(color)
                .context("missing king while converting castling rights")?;

            let expected_rook_file = match side {
                CastlingSide::KingSide => File::H,
                CastlingSide::QueenSide => File::A,
            };

            if king.file() != File::E || rook.file() != expected_rook_file {
                bail!(
                    "cannot encode Chess960 castling rights: \
                     color={color:?}, king={king}, rook={rook}"
                );
            }

            rights |= match (color, side) {
                (Color::White, CastlingSide::KingSide) => SfCastlingRights::WHITE_KING_SIDE,
                (Color::White, CastlingSide::QueenSide) => SfCastlingRights::WHITE_QUEEN_SIDE,
                (Color::Black, CastlingSide::KingSide) => SfCastlingRights::BLACK_KING_SIDE,
                (Color::Black, CastlingSide::QueenSide) => SfCastlingRights::BLACK_QUEEN_SIDE,
            };
        }
    }

    Ok(rights)
}

fn convert_position(pos: &Chess) -> anyhow::Result<SfPosition> {
    let mut sf_pos = SfPosition::empty();

    // piece placement
    for (square, piece) in pos.board().iter() {
        sf_pos.place(
            SfPiece::new(convert_role(piece.role), convert_color(piece.color)),
            convert_square(square),
        );
    }

    sf_pos.set_side_to_move(convert_color(pos.turn()));

    if let Some(ep) = pos.ep_square(EnPassantMode::Legal) {
        sf_pos.set_ep_square_unchecked(convert_square(ep));
    }

    sf_pos.set_castling_rights(convert_castling_rights(pos)?);

    let ply = (pos.fullmoves().get() - 1)
        .saturating_mul(2)
        .saturating_add(u32::from(pos.turn() == Color::Black))
        .min(u16::MAX as u32) as u16;

    sf_pos.set_ply(ply);
    sf_pos.set_rule50_counter(pos.halfmoves().min(u8::MAX as u32) as u16);

    Ok(sf_pos)
}

/// Writes all positions from one game to an sfbinpack writer.
/// Rejects games containing missing scores before writing any entries.
pub fn write_game<W: Write>(
    writer: &mut CompressedTrainingDataEntryWriter<W>,
    game: &GameRecord,
) -> anyhow::Result<()> {
    for (pos, rm) in game.replay() {
        let score = rm.score.context("Bad game: missing move score")?;

        let score = score_to_i16(score);
        let stm_score = if pos.turn() == Color::White {
            score
        } else {
            -score
        };

        writer.write_entry(&TrainingDataEntry {
            pos: convert_position(&pos)?,
            mv: convert_move(&rm.mv, &pos)?,
            score: stm_score,
            result: game.result.for_side(pos.turn()),
            ply: ((pos.fullmoves().get() - 1) * 2 + u32::from(pos.turn() == Color::Black))
                .min(u16::MAX as u32) as u16,
        })?;
    }

    Ok(())
}
