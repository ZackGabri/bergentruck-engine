// use std::io::{self, Write};
use shakmaty::{Board, Chess};

pub fn generate_fens(n: usize, seed: u64, book_path: Option<&str>, extra: Option<&str>) {
    let book_path_parsed = book_path.unwrap_or("NONE");
    let extra_parsed = extra.unwrap_or("NONE");
    println!("n: {n}, seed: {seed}, book_path: {book_path_parsed}, extra: {extra_parsed}");

    if book_path_parsed == "NONE" {
        println!("info string found 0 book lines")
    }
    
    return;
}