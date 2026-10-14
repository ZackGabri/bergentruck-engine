use burn::nn::{Linear, LinearConfig, Relu};
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct ChessEvalNet<B: Backend> {
    input_layer: Linear<B>,
    hidden_layer: Linear<B>,
    output_layer: Linear<B>,
    relu: Relu,
}

impl<B: Backend> ChessEvalNet<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            input_layer: LinearConfig::new(64, 64).init(device),
            hidden_layer: LinearConfig::new(64, 8).init(device),
            output_layer: LinearConfig::new(8, 1).init(device),
            relu: Relu::new(),
        }
    }

    pub fn forward(&self, board_features: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.input_layer.forward(board_features);
        let x = self.relu.forward(x);
        let x = self.hidden_layer.forward(x);
        let x = self.relu.forward(x);
        self.output_layer.forward(x) 
    }
}