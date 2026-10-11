use rand::{Rng, RngExt};

// ---------------------------------------------------------------- Traits

// A layer has to be able to perform forward and backward pass
pub trait Layer {
    fn forward(&mut self, input: &[f32], batch: usize) -> Vec<f32>;
    fn backward(&mut self, grad_out: &[f32], batch: usize) -> Vec<f32>;
    fn step(&mut self, _lr: f32) {}
}

// ---------------------------------------------------------------- Linear
pub struct Linear {

    pub in_dim: usize,
    pub out_dim: usize,

    // learnable parameters
    pub w: Vec<f32>,
    pub b: Vec<f32>,

    // gradients
    dw: Vec<f32>,
    db: Vec<f32>,

    input: Vec<f32>,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut impl Rng) -> Self {

        // PyTorch's default init: uniform(-1/sqrt(in), 1/sqrt(in))
        let bound = 1.0 / (in_dim as f32).sqrt();

        // parameters and biases are initialized randomly in the range [-bound, bound]
        let w = (0..out_dim * in_dim).map(|_| rng.random_range(-bound..bound)).collect();
        let b = (0..out_dim).map(|_| rng.random_range(-bound..bound)).collect();

        Self {
            in_dim,
            out_dim,
            w,
            b,
            dw: vec![0.0; out_dim * in_dim],
            db: vec![0.0; out_dim],
            input: Vec::new(),
        }
    }
}

impl Layer for Linear {
    fn forward(&mut self, input: &[f32], batch: usize) -> Vec<f32> {
        self.input = input.to_vec();

        // y[n][o] = b[o] + sum_i x[n][i] * w[o][i]

        // here we get the input from the previous layer, which is a flattened vector of size batch * in_dim
        // in the very first input of mnist this would be a vector of size 784 (28*28) for a single image,
        // or 784*batch for multiple images

        // self gives us our weights and biases, which are also flattened vectors

        self.input = input.to_vec();
        let mut out = vec![0.0; batch * self.out_dim];

        for n in 0..batch {
            for o in 0..self.out_dim {

                // bias of this single 'neuron'
                let mut sum = self.b[o];

                for i in 0..self.in_dim {
                    // weight multiplied with the input that this neuron got from prev layer
                    sum += input[n * self.in_dim + i] * self.w[o * self.in_dim + i];
                }

                out[n * self.out_dim + o] = sum;
            }
        }

        out
    }

    fn backward(&mut self, grad_out: &[f32], batch: usize) -> Vec<f32> {
        // dW[o][i] = sum_n grad_out[n][o] * x[n][i]
        // db[o]    = sum_n grad_out[n][o]
        // dx[n][i] = sum_o grad_out[n][o] * w[o][i]   (returned)
        todo!()
    }

    fn step(&mut self, lr: f32) {
        // w -= lr * dw, b -= lr * db  (momentum comes later)
        todo!()
    }
}

// ---------------------------------------------------------------- ReLU
pub struct ReLU {
    mask: Vec<bool>, // which inputs were > 0
}

impl ReLU {
    pub fn new() -> Self {
        Self { mask: Vec::new() }
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &[f32], _batch: usize) -> Vec<f32> {
        self.mask = input.iter().map(|&v| v > 0.0).collect();
        input.iter().map(|&v| v.max(0.0)).collect()
    }

    fn backward(&mut self, grad_out: &[f32], _batch: usize) -> Vec<f32> {
        // pass grad_out through where mask is true, 0 elsewhere
        todo!()
    }
}

// ---------------------------------------------------------------- Network

pub struct Network {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn forward(&mut self, x: &[f32], batch: usize) -> Vec<f32> {
        let mut out = x.to_vec();
        for layer in self.layers.iter_mut() {
            out = layer.forward(&out, batch);
        }
        out
    }

    pub fn backward(&mut self, grad: &[f32], batch: usize) {
        let mut g = grad.to_vec();
        for layer in self.layers.iter_mut().rev() {
            g = layer.backward(&g, batch);
        }
    }

    pub fn step(&mut self, lr: f32) {
        for layer in self.layers.iter_mut() {
            layer.step(lr);
        }
    }
}