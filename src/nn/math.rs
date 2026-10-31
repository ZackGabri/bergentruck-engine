#![allow(unused)]

// Note: This is NOT win probability because the expected score includes draw probability * 0.5 as well.
pub fn get_expected_score(centipawns: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf(-centipawns / 400.0))
}

pub fn get_centipawns(probability: f64) -> f64 {
    -400.0 * (1.0 / probability - 1.0).log10()
}

pub fn get_target(wdl: f64, game_result: f64, score: f64) -> f64 {
    wdl * game_result + (1.0 - wdl) * get_expected_score(score)
}

// Root-mean-square error
pub fn get_rmse(predictions: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(predictions.len(), targets.len(), "Vectors must be of equal length");
    let batch_size = predictions.len();

    let mut loss = 0.0;
    for i in 0..batch_size {
        loss += (predictions[i] - targets[i]).powi(2);
    }
    (loss / batch_size as f64).sqrt()
}
