#![allow(unused)]

/*
    Chess evaluation
*/

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

/*
    Loss functions
*/

// Root-mean-square error
pub fn get_rmse(predictions: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(
        predictions.len(),
        targets.len(),
        "Vectors must be of equal length"
    );
    let batch_size = predictions.len();

    let mut loss = 0.0;
    for i in 0..batch_size {
        loss += (predictions[i] - targets[i]).powi(2);
    }
    (loss / batch_size as f64).sqrt()
}

// Binary cross-entropy loss
pub fn get_bce_loss(predictions: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(
        predictions.len(),
        targets.len(),
        "Vectors must be of equal length"
    );
    let batch_size = predictions.len();
    let eps = 1e-15; // Small epsilon to prevent log(0) NaN explosions

    let mut total_loss = 0.0;
    for i in 0..batch_size {
        let p = predictions[i].clamp(eps, 1.0 - eps); // Clamp predictions away from exact 0.0 and 1.0
        let t = targets[i];

        total_loss += -(t * p.ln() + (1.0 - t) * (1.0 - p).ln());
    }
    total_loss / batch_size as f64
}

/*
    Schedulers
*/

pub fn step_lr(start_lr: f64, gamma: f64, step: usize, current_step: usize) -> f64 {
    start_lr * gamma.powi((current_step / step) as i32)
}

pub fn linear_lr(start_lr: f64, end_lr: f64, total_steps: usize, current_step: usize) -> f64 {
    if current_step >= total_steps {
        return end_lr;
    }
    let lr_range = start_lr - end_lr;
    let step_fraction = current_step as f64 / total_steps as f64;
    start_lr - lr_range * step_fraction
}