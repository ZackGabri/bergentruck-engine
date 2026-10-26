mod twm {
    use super::super::twm::*;
    use std::env;

    #[test]

    fn import_and_export() {
        let temp_path = env::temp_dir()
            .join("twmnet_test.bin")
            .into_string()
            .unwrap();

        // export test
        let twm_net = TWMNet::new();
        assert!(twm_net.export(&temp_path).is_ok());

        // import test
        let twm_net = TWMNet::import(&temp_path);
        assert!(twm_net.is_ok() && !twm_net.unwrap().as_ptr().is_null());
    }
}

mod pgn_reader {
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
}
