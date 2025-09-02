use mhde::{run, ModelTrait};

use burn::{
    backend::{
        Autodiff,
        NdArray
    }, module::{
        Module,
        Param,
        ModuleVisitor,
        ParamId,
    }, prelude::{
        Backend, Tensor
    }, tensor::{backend::AutodiffBackend, TensorData, BroadcastArgs},
};
use argparse::{ArgumentParser, Store, StoreOption, StoreTrue};

use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;
use rand::distributions::Distribution;
use statrs::distribution::{Cauchy, Normal};
use std::f64::consts::PI;
use std::fs;
use ndarray::{array, Array1, Array2};


/// A Cauchy+Normal 3D distribution, defined using location in 3D and
/// the Cholesky decomposition for the covariance matrix
#[derive(Module, Debug)]
pub struct CauchyModel<B: Backend> {
    loc: Param<Tensor<B, 1>>,
    diagonal: Param<Tensor<B, 1>>,
    lower: Param<Tensor<B, 1>>,
}

// impl<B> ModelTrait<B> for CauchyModel<B>
// where B: AutodiffBackend
// {
//     fn pdf(&self, data: &Tensor<B, 2>) -> Tensor<B, 1> {
//         let diag = self.cholesky.slice()
//         assert!(elems.shape().dims[0] == 3);

//         let v = (self.loc.val() - data.clone()) / self.scale.val();
//         let v = v.powi_scalar(2);
//         let v = (v + 1.0) * self.scale.val() * PI;
//         v.powi_scalar(-1)
//     }
// }

struct Options {
    num: usize,
    seed: Option<u64>,
    split: bool,
}

struct GeneratorParams {
    loc: Array1<f64>,
    scale: Array1<f64>,
    matrix: Array2<f64>,
}

fn set_options(options: &mut Options) {
    let mut parser = ArgumentParser::new();

    parser.set_description("Minimum Helliger Distance Estimator for Cauchy+Normal 3D sample and model");

    parser.refer(&mut options.num)
        .add_argument("num", Store, "Number of observations (def 1000)");

    parser.refer(&mut options.seed)
        .add_option(&["-s", "--seed"], StoreOption, "Provide a seed for reproducibility, otherwise random sample");

    parser.refer(&mut options.split)
        .add_option(&["--split"], StoreTrue, "Use split sample variant");

    parser.parse_args_or_exit();
}

/// Generates distributed sample 
fn generate(options: &Options) -> Vec<Array1<f64>> {
    let mut rng: ChaCha8Rng = match options.seed {
        Some(val) => ChaCha8Rng::seed_from_u64(val),
        None => ChaCha8Rng::from_entropy(),
    };

    // Mathematica:
    // RotationMatrix[30 Degree, {1, 0, 0}] . RotationMatrix[45 Degree, {0, 1, 0}] // N
    // {{0.707107, 0., 0.707107}, {0.353553, 0.866025, -0.353553}, {-0.612372, 0.5, 0.612372}}
    let params = GeneratorParams {
        loc: array![0.0, -1.0, 1.0],
        scale: array![1.0, 2.0, 3.0],
        matrix: array![[0.707107, 0., 0.707107],
            [0.353553, 0.866025, -0.353553],
            [-0.612372, 0.5, 0.612372]],
    };

    // create random vector by creating 3 Cauchy distributions for each dimension
    // as per params loc and scale and then apply matrix rotation
    let dist: Vec<Cauchy> = (0..3).map(|ix| {
        Cauchy::new(params.loc[ix], params.scale[ix]).unwrap()
    }).collect();

    let vec = Vec::from_iter((0..options.num).map(
        |_| (0..3).map(|ix| dist[ix].sample(&mut rng)).collect::<Array1<f64>>()
    ));

    let vec = vec.iter().map(|row| row.dot(&params.matrix)).collect();

    vec
}

// fn min_median_max(numbers: &Vec<f64>) -> (f64, f64, f64) {

//     let mut to_sort = numbers.clone();
//     to_sort.sort_by(|a, b| a.partial_cmp(b).unwrap());

//     let mid = numbers.len() / 2;
//     let med = if numbers.len() % 2 == 0 {
//         (numbers[mid - 1] + numbers[mid]) / 2.0
//     } else {
//         numbers[mid]
//     };
//     (to_sort[0], med, to_sort[numbers.len()-1])
// }


// fn cauchy_model<B: Backend>(vec: &Vec<f64>, device: B::Device) -> CauchyModel<B> {
//     let (v_min, v_med, v_max) = min_median_max(&vec);

//     let loc: Tensor<B, 1> = Tensor::from_data([v_med], &device);
//     let scale: Tensor<B, 1> = Tensor::from_data([(v_max - v_min) / (vec.len() as f64)], &device);

//     CauchyModel {
//         loc: Param::from_tensor(loc),
//         scale: Param::from_tensor(scale),
//     }
// }

type AutoBE = Autodiff<NdArray<f64, i64>>;

fn main() {
    let device: <AutoBE as Backend>::Device = Default::default();
    let data: Tensor<AutoBE, 2> = Tensor::from_data([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, 2.0, 3.0]
    ], &device);
    let loc: Tensor<AutoBE, 1> = Tensor::from_data([-1.0, -3.0, 0.0], &device);

    let works = data.clone() + loc.clone().unsqueeze::<2>();

    println!("works = {:} +\n {:} = \n {:}", data, loc, works);

}

fn main1() {
    let device: <AutoBE as Backend>::Device = Default::default();
    let dim = 3;
    let options = Options {
        num: 10,
        seed: Some(42),
        split: false,
    };
    let raw_data = generate(&options);
}

fn main2() {
    let device: <AutoBE as Backend>::Device = Default::default();

    let dim = 3;

    // N points in D dimension
    let raw_data: Tensor<AutoBE, 2> = Tensor::from_data([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, 2.0, 3.0]
    ], &device);

    // the Cauchy location in 1 x D tensor
    let mut loc: Param<Tensor<AutoBE, 2>> = Param::from_tensor(Tensor::from_data([[0.0, 0.0, -1.0]], &device));

    // Cholesky decomposition of the covariance matrix
    // LowT = Lower Triangular matrix of size DxD/2, rows of increasing size
    let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(101);
    let dist = Normal::standard();

    // Code below demonstrates how to turn a diagonal + lower parametrization
    // of a lower triangular matrix into the Cholesky decomposition of a covariance
    // matrix and apply the operation to data

    // items on the diagonal: D (obviously)
    let data: Vec<f64> = Vec::from_iter((0..dim).map(|_| dist.sample(&mut rng)));
    let diagonal = Param::from_tensor(Tensor::<AutoBE, 1>::from_data(data.as_slice(), &device));

    println!("Diag Param >> {:} = {:}", diagonal.id, diagonal.val());

    // items below the diagonal: n * (n-1) / 2
    let dim2 = dim * (dim - 1) / 2;
    let data: Vec<f64> = Vec::from_iter((0..dim2).map(|_| dist.sample(&mut rng)));
    let lower = Param::from_tensor(Tensor::<AutoBE, 1>::from_data(data.as_slice(), &device));
    println!("Lower Param >> {:} = {:}", lower.id, lower.val());

    // given parameters diagonal and lower, below compose those into a lower triangular
    // matrix with positive diagonal entries
    let diagonal2 = diagonal.clone().val() * diagonal.val();

    // to simplify, the initial matrix is a flat tensor
    let mut flat = Tensor::<AutoBE, 1>::zeros([dim * dim], &device);
    let mut pos = 0;

    // fill diagonal
    for ix in 0..dim {
        flat = flat.slice_assign([pos..pos+1], diagonal2.clone().slice([ix..ix+1]));
        pos = pos + dim + 1;
    }
    // fill lower triangular part
    let mut len = 1;
    pos = 0;
    for ix in 1..dim {
        flat = flat.slice_assign([ix * dim .. ix * dim + len], lower.clone().val().slice([pos .. pos + len]));
        pos = pos + len;
        len = len + 1;
    }

    // reshape, transpose and get covariance matrix
    let matrix = flat.reshape([dim, dim]);
    // println!("{:}", matrix);
    let matrix_t = matrix.clone().transpose();
    // println!("{:}", matrix_t);
    let cov_matrix = matrix.matmul(matrix_t);
    // println!("{:}", cov_matrix);

    println!("Matrix >> {:}", cov_matrix);

    // then apply to data
    let data2 = raw_data.clone().matmul(cov_matrix);

    println!("Orig >>\n{:}", raw_data);

    println!("Transform >>\n{:}", data2);

    let grads = data2.backward();

    let diagonal_grad = diagonal.grad(&grads).unwrap();
    println!("Diag grad = {:}", diagonal_grad);

    let lower_grad = lower.grad(&grads).unwrap();
    println!("Lower grad = {:}", lower_grad);
}
