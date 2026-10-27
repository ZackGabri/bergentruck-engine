use crate::pgn_reader::PgnReader;
use crate::pgn_reader::viriformat::*;
use crate::pgn_reader::*;
use crate::search::MATE_SCORE;
use pgn_reader::Reader;

#[test]
fn parses_engine_comments() {
    assert_eq!(
        parse_engine_comment("+11.41/9 0.009s, n=6911, sd=15"),
        Some(1141)
    );
    assert_eq!(parse_engine_comment("-0.25/9 0.009s"), Some(-25));
    assert_eq!(parse_engine_comment("+M3/11 0.008s"), Some(MATE_SCORE - 6));
    assert_eq!(
        parse_engine_comment("-M2/12 0.005s"),
        Some(-(MATE_SCORE - 4))
    );
    assert_eq!(parse_engine_comment("book"), None);
}

#[test]
fn reads_sample_game() {
    let pgn = br#"[Event "t"]
    [Result "1-0"]
    [SetUp "1"]
    [FEN "r1bqkbn1/Q3p1pr/1pp2pP1/3p3p/1n5P/2P2N2/PP1PPPBR/RNB1K3 w Qq - 0 11"]
    
    11. Qxa8 {+11.41/9 0.009s, n=6911, sd=15} Nc2+ {-13.09/9 0.009s, n=6460, sd=12} 1-0
    "#;
    let mut reader = Reader::new(&pgn[..]);
    let mut visitor = PgnReader;
    let game = reader
        .read_game(&mut visitor)
        .unwrap()
        .unwrap()
        .map_err(|e| e.to_string())
        .unwrap();

    assert_eq!(game.result, GameResult::WhiteWin);
    assert_eq!(game.moves.len(), 2);
    assert_eq!(game.moves[0].score, Some(1141));
    assert_eq!(game.moves[1].score, Some(1309));
}

#[test]
fn viriformat_startpos_board_layout() {
    let b = pack_board(&Chess::default(), 1);
    assert_eq!(
        u64::from_le_bytes(b[0..8].try_into().unwrap()),
        0xFFFF_0000_0000_FFFF
    );
    assert_eq!(b[24], 64); // white to move, no en passant
    assert_eq!(b[30], 1); // draw
    assert_eq!(u16::from_le_bytes(b[26..28].try_into().unwrap()), 1);
}

#[test]
fn viriformat_file_size_matches_spec() {
    // 32-byte board + 4 bytes per move + 4-byte terminator
    let pgn = br#"[Result "1-0"]

1. e4 {+0.30/9 0.01s} e5 {-0.20/9 0.01s} 1-0
"#;
    let mut reader = pgn_reader::Reader::new(&pgn[..]);
    let game = reader
        .read_game(&mut PgnReader)
        .unwrap()
        .unwrap()
        .map_err(|e| e.to_string())
        .unwrap();
    let mut buf = Vec::new();
    write_game(&mut buf, &game).unwrap();
    assert_eq!(buf.len(), 32 + 4 * 2 + 4);
}
