#![allow(unused)]

use minstant::Instant;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use super::data::TrainingSample;
use super::twm::TWMNet;

pub struct TrainingSteps {
    pub batch_size: usize,
    pub batches_per_superbatch: usize,
    pub start_superbatch: usize,
    pub end_superbatch: usize,
}

impl Default for TrainingSteps {
    fn default() -> Self {
        Self {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 40,
        }
    }
}

pub struct LearningRate {
    pub start: f64,
    pub gamma: f64,
    pub step: usize,
}

impl Default for LearningRate {
    fn default() -> Self {
        Self {
            start: 0.001,
            gamma: 0.1,
            step: 18,
        }
    }
}

pub struct TrainerConfig {
    pub net_id: String,
    pub eval_scale: f64,
    pub training_steps: TrainingSteps,
    pub learning_rate: LearningRate,
    pub save_rate: usize,
    pub output_directory: String,
    pub test_set: Option<String>,
    pub seed: u64,
    pub log_every: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            net_id: "test".into(),
            eval_scale: 400.0,
            training_steps: TrainingSteps::default(),
            learning_rate: LearningRate::default(),
            save_rate: 10,
            output_directory: "checkpoints".into(),
            test_set: None,
            seed: 4, // this is a very random, very secure, and very unguessable number
            log_every: 1000,
        }
    }
}

pub struct NetworkTrainer {
    pub config: TrainerConfig,
    net: TWMNet,
}

impl NetworkTrainer {
    pub fn new() -> Self {
        Self {
            config: TrainerConfig::default(),
            net: TWMNet::new(),
        }
    }

    fn lr_at(&self, sb: usize) -> f64 {
        let lr = &self.config.learning_rate;
        lr.start * lr.gamma.powi((sb / lr.step) as i32)
    }

    fn superbatch_size(&self) -> usize {
        self.config.training_steps.batch_size * self.config.training_steps.batches_per_superbatch
    }

    pub fn train(&mut self, mut data: Vec<TrainingSample>) -> anyhow::Result<()> {
        println!("Started training with {} data samples", data.len());
        std::fs::create_dir_all(&self.config.output_directory)?;

        // suffle the data
        let mut rng = StdRng::seed_from_u64(self.config.seed);
        data.shuffle(&mut rng);
        let mut sample_index = 0;

        let training_steps = &self.config.training_steps;

        for sb in training_steps.start_superbatch..=training_steps.end_superbatch {
            let lr = self.lr_at(sb);
            self.net.update_learning_rate(lr);

            let mut loss = 0.0; // how do i calculate this?

            let start = Instant::now();
            for i in 0..self.superbatch_size() {
                if sample_index == data.len() {
                    data.shuffle(&mut rng);
                    sample_index = 0;
                }

                let s = &data[sample_index];
                sample_index += 1;

                self.net.reset();
                self.net.advance(s.features).expect("Advancing net failed");
                self.net.update([s.target]).expect("Updating net failed");

                if self.config.log_every > 0 && (i + 1) % self.config.log_every == 0 {
                    println!(
                        "Superbatch {sb} [{}/{}] {:.0} samples/s",
                        i + 1,
                        self.superbatch_size(),
                        (i + 1) as f64 / start.elapsed().as_secs_f64()
                    );
                }
            }

            println!(
                "superbatch {sb}/{} lr={lr:.6} loss={loss:.6} time={:.2?}",
                training_steps.end_superbatch,
                start.elapsed()
            );

            if sb % self.config.save_rate == 0 || sb == training_steps.end_superbatch {
                self.net.export(&format!(
                    "{}/{}-{sb}.net",
                    self.config.output_directory, self.config.net_id
                ))?;
            }
        }

        Ok(())
    }
}
