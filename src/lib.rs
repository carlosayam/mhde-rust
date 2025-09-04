use burn::{
    module::{AutodiffModule, ModuleVisitor, ParamId}, optim::{AdamConfig, GradientsParams, Optimizer}, prelude::{Backend, Float, Tensor}, record::Record, tensor::{backend::AutodiffBackend, cast::ToElement}
};

use core::f64;
use std::{f64::consts::PI, iter::zip};

use ball_tree::BallTree;
pub use ball_tree::Point;

use burn::tensor::ElementConversion;

/// A Burn Module must implement a `pdf` on data function
pub trait ModelTrait<B: AutodiffBackend>: AutodiffModule<B> {
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

fn volume_dim(dim: usize, radius: f64) -> f64 {
    match dim {
        1 => 2.0 * radius,
        2 => PI * radius * radius,
        3 => PI * radius * radius * radius * 4.0 / 3.0,
        4 => PI * PI * radius * radius * radius * radius / 2.0,
        _ => panic!("wrong dimension")
    }
}

fn calculate_balls<B: Backend>(data: &Vec<Vector<B>>, split: bool, device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 1>) {

    // considered that the sample could be split to ensure i.i.d terms in the sum
    // but there were no apparent benefits; leaving this legacy in case need to investigate
    // again
    let num = data.len();
    let dim = data[0].dim();
    let data1 = if split { &data[0..(num / 2)] } else { &data[..] };  // slice used for calculate volume to nearest
    let data2 = if split { &data[(num / 2)..] } else { &data[..] };   // slice used to iterate points

    let algo = BallTree::new(data1.to_vec(), std::iter::repeat(()).take(num).collect());
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
    pub config_optimizer: AdamConfig,
}

struct GradientCheck<'a, B: AutodiffBackend> {
    epsilon: f64,
    is_less: bool,
    grads: &'a B::Gradients,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for GradientCheck<'_, B> {
    fn visit_float<const D: usize>(&mut self, _id: ParamId, tensor: &Tensor<B, D>) {
        if self.is_less {
            let grads = tensor.grad(&self.grads).unwrap();
            for (_i, tensor) in grads.iter_dim(0).enumerate() {
                let val: f64 = tensor.into_scalar().elem();
                self.is_less = self.is_less && val.abs() < self.epsilon;
            }
        }
    }
}


pub fn run<B: AutodiffBackend, M: ModelTrait<B>>(
    mut model: M,
    sample: Vec<Tensor<B, 1>>,
    split: bool,
    device: B::Device,
) -> (usize, M) {

    let config = TrainingConfig {
        num_runs: 1000,
        lr: 0.25,
        config_optimizer: AdamConfig::new(),
    };

    let sample_wrapped: Vec<Vector<B>> = sample.iter().map(|tensor: &Tensor<B, 1>| Vector(tensor.clone())).collect();

    let (data, balls) = calculate_balls::<B>(&sample_wrapped, split, &device);

    let mut optimizer = config.config_optimizer.init::<B, M>();
    let epsilon: f64 = 0.000001;

    let mut ix = 1;
    while ix <= config.num_runs {

        let hd = forward(&model, &data, &balls);

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

        model = optimizer.step(config.lr, model, grads_container);

        let bhat_val: f64 = hd.into_scalar().elem::<f64>();

        if ix % 10 == 0 {
            println!("HD^2 Hat: {} ({})", bhat_val, ix);
        }
        if is_less {
            break;
        }
        ix += 1;
    }
    (ix, model)
}
