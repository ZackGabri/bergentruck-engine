#![allow(unused)]

use std::error::Error;
use std::fs::File;
use std::ops::ControlFlow;
use std::path::Path;

use pgn_reader::{RawComment, RawTag, Reader, SanPlus, Skip, Visitor};
use shakmaty::{CastlingMode, Chess, Color, Move, Position, fen::Fen};

use crate::search::{MATE_SCORE, MATE_THRESHOLD};

/*
*
* 12am and i spent too long making this instead of the actual trainer
* It's made with being able to convert into viriformat/binpack with ease in mind
* But i haven't yet done those, so i will come back to it later as well as adding readers
*
*/

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameResult {
    WhiteWin,
    Draw,
    BlackWin,
}

impl GameResult {
    fn from_tag(s: &[u8]) -> Option<Self> {
        match s {
            b"1-0" => Some(Self::WhiteWin),
            b"0-1" => Some(Self::BlackWin),
            b"1/2-1/2" => Some(Self::Draw),
            _ => None, // "*" = unfinished, skip the game
        }
    }

    // viriformat WDL byte: 0 = Black win, 1 = draw, 2 = white win
    pub fn wdl_u8(self) -> u8 {
        match self {
            Self::BlackWin => 0,
            Self::Draw => 1,
            Self::WhiteWin => 2,
        }
    }

    // sf-binpack style result from the point of view of `stm`: -1 / 0 / 1
    pub fn for_side(self, stm: Color) -> i16 {
        match (self, stm) {
            (Self::Draw, _) => 0,
            (Self::WhiteWin, Color::White) | (Self::BlackWin, Color::Black) => 1,
            _ => -1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RecordedMove {
    pub mv: Move,
    // from white pov centipawns, None if the move had no parsable comment.
    pub score: Option<i32>,
}

#[derive(Clone, Debug)]
pub struct GameRecord {
    // starting position
    pub start: Chess,

    pub result: GameResult,
    pub moves: Vec<RecordedMove>,
}

// parses engine comments like `+11.41/9 0.009s, n=6911` or `-M2/12 0.005s`.
// returns the score from the point of view of the side that just moved.
pub fn parse_engine_comment(comment: &str) -> Option<i32> {
    let token = comment.split_whitespace().next()?; // "+11.41/9"
    let score = token.split('/').next()?; // "+11.41"

    let (sign, body) = match score.chars().next()? {
        '+' => (1i32, &score[1..]),
        '-' => (-1i32, &score[1..]),
        _ => (1i32, score),
    };

    if let Some(n) = body.strip_prefix('M') {
        let n: i32 = n.parse().ok()?;
        let magnitude = (MATE_SCORE - 2 * n).max(1);
        return Some(sign * magnitude);
    }

    let pawns: f64 = body.parse().ok()?;
    Some((pawns * 100.0).round() as i32 * sign)
}

// Working state while a game's movetext is being read.
pub struct Replay {
    current: Chess, // board after the last move
    start: Chess,
    result: GameResult,
    moves: Vec<RecordedMove>,
}

// Lowk i hate this pgn reader implementation
// it's too verbose and probably should've wrote my own parser at this point
// but oh well, i'm in too deep with it
pub struct PgnReader;

impl Visitor for PgnReader {
    type Tags = (Option<Chess>, Option<GameResult>);
    type Movetext = Replay;
    type Output = Result<GameRecord, Box<dyn Error>>;

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        ControlFlow::Continue((None, None))
    }

    fn tag(
        &mut self,
        tags: &mut Self::Tags,
        name: &[u8],
        value: RawTag<'_>,
    ) -> ControlFlow<Self::Output> {
        if name == b"FEN" {
            let fen = match Fen::from_ascii(value.as_bytes()) {
                Ok(fen) => fen,
                Err(err) => return ControlFlow::Break(Err(err.into())),
            };
            let pos = match fen.into_position(CastlingMode::Standard) {
                Ok(pos) => pos,
                Err(err) => return ControlFlow::Break(Err(err.into())),
            };
            tags.0 = Some(pos);
        } else if name == b"Result" {
            tags.1 = GameResult::from_tag(value.as_bytes());
        }
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, tags: Self::Tags) -> ControlFlow<Self::Output, Self::Movetext> {
        let Some(result) = tags.1 else {
            return ControlFlow::Break(Err("game has no usable Result tag".into()));
        };
        let start = tags.0.unwrap_or_default();
        ControlFlow::Continue(Replay {
            current: start.clone(),
            start,
            result,
            moves: Vec::new(),
        })
    }

    fn san(&mut self, st: &mut Self::Movetext, san_plus: SanPlus) -> ControlFlow<Self::Output> {
        match san_plus.san.to_move(&st.current) {
            Ok(mv) => {
                st.current.play_unchecked(mv);
                st.moves.push(RecordedMove { mv, score: None }); // filled in by the comment
                ControlFlow::Continue(())
            }
            Err(err) => ControlFlow::Break(Err(err.into())),
        }
    }

    fn comment(
        &mut self,
        st: &mut Self::Movetext,
        comment: RawComment<'_>,
    ) -> ControlFlow<Self::Output> {
        if let Ok(text) = std::str::from_utf8(comment.as_bytes())
            && let Some(mover_cp) = parse_engine_comment(text)
        {
            // After the move, if it's black's turn then white was the mover
            let white_cp = if st.current.turn() == Color::Black {
                mover_cp
            } else {
                -mover_cp
            };
            // A comment belongs to the move before it
            if let Some(last) = st.moves.last_mut() {
                last.score = Some(white_cp);
            }
        }
        ControlFlow::Continue(())
    }

    fn begin_variation(&mut self, _st: &mut Self::Movetext) -> ControlFlow<Self::Output, Skip> {
        ControlFlow::Continue(Skip(true)) // ignore side lines
    }

    fn end_game(&mut self, st: Self::Movetext) -> Self::Output {
        Ok(GameRecord {
            start: st.start,
            result: st.result,
            moves: st.moves,
        })
    }
}

/// Returns a list of games in the pgn file as a GameRecord list
pub fn read_pgn<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<GameRecord>> {
    let mut reader = Reader::new(File::open(path)?);
    let mut visitor = PgnReader;
    let mut all = Vec::new();
    while let Some(game) = reader.read_game(&mut visitor)? {
        match game {
            Ok(game) => all.push(game),
            Err(e) => eprintln!("skipping game: {e}"),
        }
    }
    Ok(all)
}
