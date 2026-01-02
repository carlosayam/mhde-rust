use burn::{
    module::{AutodiffModule, ModuleVisitor, ParamId},
    optim::{AdamConfig, GradientsParams, Optimizer, SgdConfig},
    prelude::{Backend, Float, Tensor, Int},
    tensor::{backend::AutodiffBackend, cast::ToElement}
};

use core::f64;
use std::{f64::consts::PI, iter::zip};

use rand::Rng;

use ball_tree::BallTree;
pub use ball_tree::Point;

use burn::tensor::ElementConversion;

/// A Burn Module must implement a `pdf` function to be able
/// to use this estimator
pub trait ModelTrait<B: AutodiffBackend>: AutodiffModule<B> {
    /// Calculates the PDF for the given $R^d$ data
    fn pdf(&self, data: &Tensor<B, 2>) -> Tensor<B, 1>;
}

pub struct HellingerOutput<B>
where
    B: Backend,
{
    pub loss: Tensor<B, 1>,
}

/// This `forward` function calculates the estimate for
/// squared Hellinger distance. At the moment, it assumes that
/// the volume balls do not depend on the parameters of the model.
pub fn forward<B: AutodiffBackend, M: ModelTrait<B>>(
    model: &M,
    data: &Tensor<B, 2>,
    balls: &Tensor<B, 1>
) -> Tensor<B, 1> {
    let pdf = model.pdf(data);
    // now calculate Hellinger Distance squared estimator, eqs (3) & (4) in paper
    let v = (pdf * balls.clone()).powf_scalar(0.5);
    let num = data.shape().dims[0];
    let factor = - 2.0 / ((num as f64) * PI).sqrt();
    v.sum() * factor + 1.0
}


/// wrapper type for Tensor<B,1> so we can implement Point interface
pub struct Vector<B: Backend>(Tensor<B ,1>);

impl<B: Backend> Vector<B> {
    fn dim(&self) -> usize {
        self.0.dims()[0]
    }
}

impl<B: Backend> Clone for Vector<B> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<B: Backend> PartialEq for Vector<B> {
    fn eq(&self, other: &Self) -> bool {
        for (e1, e2) in zip(self.0.clone().iter_dim(0), other.0.clone().iter_dim(0)) {
            if e1.into_scalar().to_f64() != e2.into_scalar().to_f64() {
                return false;
            }
        }
        true
    }
}

impl<B: Backend> Point for Vector<B> {
    fn distance(&self, other: &Self) -> f64 {
        (self.0.clone() - other.0.clone()).powi_scalar(2).sum().into_scalar().to_f64().sqrt()
    }

    fn move_towards(&self, other: &Self, d: f64) -> Self {
        Self(self.0.clone() + (self.0.clone() - other.0.clone()) * d)
    }
}

/// Calculate volume of n-ball of dimension `dim`
/// using the recursive formula
/// $$
/// V_n(R) = \begin{cases}
/// 1 &\text{if } n=0,\\[0.5ex]
/// 2R &\text{if } n=1,\\[0.5ex]
/// \dfrac{2\pi}{n}R^2 \times V_{n-2}(R) &\text{otherwise}.
/// \end{cases}
/// $$
fn volume_dim(dim: usize, radius: f64) -> f64 {
    match dim {
        0 => 1.0,
        1 => 2.0 * radius,
        _ => 2.0 * PI * radius * radius / (dim as f64) * volume_dim(dim - 2, radius)
    }
}

/// Minimum number of observations to be able to apply Ranneby's results
const MIN_OBSERVATIONS: usize = 16;

fn calculate_balls<B: Backend>(data: &Vec<Vector<B>>, split: bool, device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 1>) {

    // considered that the sample could be split to ensure i.i.d terms in the sum
    // but there were no apparent benefits; leaving this legacy in case need to investigate
    // again
    let num = data.len();
    assert!(num >= MIN_OBSERVATIONS, "not enough observations");
    let dim = data[0].dim();
    let data1 = if split { &data[0..(num / 2)] } else { &data[..] };  // slice used for calculate volume to nearest
    let data2 = if split { &data[(num / 2)..] } else { &data[..] };   // slice used to iterate points

    let algo = BallTree::new(data1.to_vec(), std::iter::repeat(()).take(data1.len()).collect());
    let mut query = algo.query();
    let nearest = if split { 1 } else { 2 };

    let radii: Vec<f64> = data2.iter()
        .map(|pt: &Vector<B>| query.nn(pt).take(nearest).last())
        .map(|nn_result| volume_dim(dim, nn_result.unwrap().1))
        .collect();

    (
        Tensor::stack(data2.iter().map(|pt: &Vector<B>| pt.0.clone()).collect(), 0),
        Tensor::from_data(radii.as_slice(), device)
    )
}


pub struct TrainingConfig {
    pub num_runs: usize,
    pub lr: f64,
    pub config_optimizer: SgdConfig,
}

struct GradientCheck<'a, B: AutodiffBackend> {
    epsilon: f64,
    is_less: bool,
    grads: &'a B::Gradients,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for GradientCheck<'_, B> {
    fn visit_float<const D: usize>(&mut self, _id: ParamId, tensor: &Tensor<B, D>) {
        if self.is_less {
            if let Some(grads) = tensor.grad(&self.grads) {
                for (_i, tensor) in grads.iter_dim(0).enumerate() {
                    let val: f64 = tensor.into_scalar().elem();
                    self.is_less = self.is_less && val.abs() < self.epsilon;
                }
            } else {
                // if we are missing a gradient, do not stop
                self.is_less = false;
            }
        }
    }
}

fn extract_batch<B: Backend>(tensor: Tensor<B, 2>, balls: Tensor<B, 1>, num_samples: usize) -> (Tensor<B, 2>, Tensor<B, 1>)
{
    let num_rows = tensor.dims()[0];

    // Generate random row indices
    let mut rng = rand::thread_rng();
    let mut random_indices_vec = Vec::new();
    for _ in 0..num_samples {
        random_indices_vec.push(rng.gen_range(0..num_rows) as i64);
    }

    // Convert the vector of indices into a Burn Int tensor
    let random_indices = Tensor::<B, 1, Int>::from_data(random_indices_vec.as_slice(), &tensor.device());

    // Select the rows using the random indices
    (
        tensor.select(0, random_indices.clone()),
        balls.select(0, random_indices.clone())
    )
}

pub fn run<B: AutodiffBackend, M: ModelTrait<B>>(
    mut model: M,
    sample: Vec<Tensor<B, 1>>,
    split: bool,
    device: B::Device,
) -> (usize, M) {

    let config = TrainingConfig {
        num_runs: 10000,
        lr: 0.004,
        config_optimizer: SgdConfig::new(),
    };

    let sample_wrapped: Vec<Vector<B>> = sample.iter().map(|tensor: &Tensor<B, 1>| Vector(tensor.clone())).collect();

    let (data, balls) = calculate_balls::<B>(&sample_wrapped, split, &device);

    let mut optimizer = config.config_optimizer.init::<B, M>();
    let epsilon: f64 = 0.000001;

    let orig_model = model.clone();

    let mut ix = 1;
    let num = sample.len();
    let batch_size = num / 20;
    let batch_in_sample = num / batch_size + 1;
    let max_iter = config.num_runs * batch_in_sample;
    let min_iter = (config.num_runs / 100) * batch_in_sample;
    let ix_iter = max_iter / 25 + 1;
    let mut lr_temp = 0.0;
    println!("Max runs: {}", max_iter);

    while ix < max_iter {

        let (data_batch, balls_batch) = extract_batch(data.clone(), balls.clone(), batch_size);

        let hd = forward(&model, &data_batch, &balls_batch);

        let grads = hd.backward();

        let is_less = {
            let mut grad_check = GradientCheck {
                epsilon,
                is_less: true,
                grads: &grads,
            };
            model.visit(&mut grad_check);
            grad_check.is_less
        };
        
        let grads_container = GradientsParams::from_grads(grads, &model);

        lr_temp = 1.0 - ((ix / 2) as f64) / (max_iter as f64);

        model = optimizer.step(config.lr * lr_temp, model, grads_container);

        // calculate HD for real
        let hd = forward(&model, &data, &balls);

        let hdhat_val: f64 = hd.into_scalar().elem::<f64>();

        if hdhat_val < -0.5 {
            model = orig_model.clone();
            // last_hd = 1.0;
            println!("Reseting (ix={})", ix);
            continue;
        }

        // if hdhat_val.abs() < last_hd * 0.9 {
        //     last_hd = hdhat_val.abs();
        // }

        if ix % ix_iter == 0 {
            println!("HD^2 Hat: {} (ix={}, lr={})", hdhat_val, ix, config.lr * lr_temp);
        }

        if ix > min_iter && is_less {
            println!("Final HD^2 Hat: {} ({})", hdhat_val, ix);
            break;
        }
        ix += 1;
    }
    (ix, model)
}


pub fn check_close<B: Backend, const D: usize>(a: Tensor<B, D, Float>, b: Tensor<B, D, Float>) {
    let resp = (a.clone() - b.clone()).max_abs();
    let msg = format!("TENSORS DIFFER\nGot >> \n{:?}\nExpected >> \n{:?}\n", a, b);
    assert!(resp.into_scalar().elem::<f64>() < 1E-5, "{}", msg);
}

#[cfg(test)]
mod tests {

    use super::*;

    use burn::{backend::{Autodiff, NdArray}, tensor::Shape};
    use rand::prelude::Distribution;
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;
    use statrs::distribution::Uniform;

    type AutoBE = Autodiff<NdArray<f64, i64>>;

    #[test]
    fn test_volume_dim() {
        for (dim, rx) in zip(1..5, 2..6) {
            let radius = rx as f64;
            let vol = volume_dim(dim, radius);
            let exp_vol: f64 = match dim {
                1 => 2.0 * radius,
                2 => PI * radius * radius,
                3 => PI * radius * radius * radius * 4.0 / 3.0,
                4 => PI * PI * radius * radius * radius * radius / 2.0,
                _ => panic!("wrong dimension")
            };
            assert!((vol - exp_vol).abs() < 1E-5, "Failed at dimension {:}: got {:}, expected {:}", dim, vol, exp_vol);
        }

    }

    #[test]
    fn test_calculate_balls() {
        let device: <AutoBE as Backend>::Device = Default::default();

        // Create a simple dataset to test the function
        // Important: if one changes the seed, expected values need to be computed
        let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(42);
        let dist = Uniform::new(0.0, 1.0).unwrap();
        let num = MIN_OBSERVATIONS;

        let data: Vec<_> = (0..num)
            .map(|_| Vector(Tensor::from_floats([dist.sample(&mut rng), dist.sample(&mut rng)], &device)))
            .collect();

        // Call the function with split = false
        let (data_tensor, balls) = calculate_balls::<AutoBE>(&data, false, &device);

        // Check if the output tensors have the correct shapes and values
        assert_eq!(data_tensor.shape(), Shape::new([num, 2]));
        assert_eq!(balls.shape(), Shape::new([num]));

        // this was verified using Mathematica:
        // data = ...;
        // nnf = Nearest[data];
        // (Pi Norm[# - nnf[#, 2][[2]]]^2) & /@ data
        let exp_balls: Tensor<AutoBE, 1> = Tensor::from_floats(
            [0.103625, 0.142726, 0.0929269, 0.142726, 0.0167641, 0.103625, 0.130203, 0.0167641,
            0.0469551, 0.0827687, 0.0531408, 0.00347566, 0.0531408, 0.1072, 0.1072, 0.00347566], &device);
        check_close(balls, exp_balls);

        // Call the function with split = true
        let (data_tensor_split, balls_split) = calculate_balls::<AutoBE>(&data, true, &device);

        // Check if the output tensors have the correct shapes and values when split
        assert_eq!(data_tensor_split.shape(), Shape::new([(num + 1) / 2, 2]));
        assert_eq!(balls_split.shape(), Shape::new([(num + 1) / 2]));

        // similarly, verified using Mathematica
        // nnf = Nearest[data[[1 ;; 8]]];
        // (Pi Norm[# - nnf[#, 2][[1]]]^2) & /@ data[[9 ;;]]
        let exp_balls_split: Tensor<AutoBE, 1> = Tensor::from_floats(
            [0.0741026,0.0929269,0.147737,0.0345391,0.235881,0.130203,0.197212,0.0290994], &device);
        check_close(balls_split, exp_balls_split);

    }

}
