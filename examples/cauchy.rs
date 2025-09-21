//! Usage example using multivariate Cauchy-distributed vectors in R^d
//! 
//! Use like `cargo run --example cauchy -- -c examples/config.yaml` to use the provided `config.yaml` example.
//! 
use std::fs;
use std::iter::zip;

use yaml_rust::{Yaml, YamlLoader};
use ndarray::{Array, Array1, Array2, s};

use mhde::{run, ModelTrait};

use burn::{
    backend::{
        Autodiff,
        NdArray
    }, module::{
        Module,
        Param
    }, prelude::{
        Backend, Tensor, Shape, TensorData
    }, tensor::{backend::AutodiffBackend},
};
use argparse::{ArgumentParser, Store, Print};

use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

use rand::distributions::Distribution;
use statrs::distribution::Cauchy;
use std::{f64::consts::PI};

/// Parameters for Cauchy-distributed data in dim dimensions,
/// defined using location in R^{dim} and the lower triangular
/// inverse matrix with positive entries for the diagonal
/// and arbitrary values for the below-diagonal components
#[derive(Module, Debug)]
pub struct CauchyModel<B: Backend> {
    // space dimension
    dim: usize,
    // location for the multi-variate distribution
    loc: Param<Tensor<B, 1>>,
    // diagonal entries to be squared (so, their square roots)
    diagonal: Param<Tensor<B, 1>>,
    // below diagonal entries of length dim * (dim - 1) / 2.
    lower: Param<Tensor<B, 1>>,
}

impl<B: Backend> CauchyModel<B> {

    /// Calculates the matrix associated with the diagonal and lower parameters,
    /// and the factor to adjust the PDF (product of the diagonal entries)
    fn matrix_and_factor(&self) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let device = self.loc.device();
        let mut flat = Tensor::<B, 1>::zeros([self.dim * self.dim], &device);

        // diagonal squared, so these are positive entries
        let diagonal2 = self.diagonal.clone().val() * self.diagonal.val();
        // fill diagonal
        let mut pos = 0;
        for ix in 0..self.dim {
            flat = flat.slice_assign([pos..pos+1], diagonal2.clone().slice([ix..ix+1]));
            pos = pos + self.dim + 1;
        }

        let factor: Tensor<B, 1> = diagonal2.prod();

        // fill lower triangular part
        let mut len = 1;
        pos = 0;
        for ix in 1..self.dim {
            flat = flat.slice_assign([ix * self.dim .. ix * self.dim + len], self.lower.clone().val().slice([pos .. pos + len]));
            pos = pos + len;
            len = len + 1;
        }

        // finalise by turning into matrix
        let matrix = flat.reshape([self.dim, self.dim]);

        (matrix, factor)
    }
}

impl<B> ModelTrait<B> for CauchyModel<B>
where B: AutodiffBackend
{
    fn pdf(&self, data: &Tensor<B, 2>) -> Tensor<B, 1> {
        let (matrix, factor) = self.matrix_and_factor();
        let data = data.clone() - self.loc.val().clone().unsqueeze::<2>();
        let v = matrix.matmul(data.transpose()).transpose();
        let v = v.powi_scalar(2);
        let v = (v + 1.0) * PI;
        let v = v.powi_scalar(-1);
        let v = v.prod_dim(1).squeeze::<1>(1);
        v * factor
    }
}

#[derive(Debug)]
struct Options {
    dim: usize,
    loc: Vec<f64>,
    lower: Vec<Vec<f64>>,
    num: usize,
    seed: Option<u64>,
    split: Option<bool>,
}

/// Convert a parsed Yaml representation to Options for the experiment.
/// 
/// As the Yaml representation is very generic, using Vec and Hashmap, this function
/// expects the Yaml to have fields and values according to the following specification:
/// ```yaml
/// # number of dimensions for the experiment; must be between 1 and 4
/// dim: <int>
/// # location for Cauchy-distributed data; it must have `dim` number of items
/// loc: <array of f64>
/// # lower triangular matrix to transform the initial R^{dim} points; it is a series
/// # of arrays with increasing number of elements, starting with one f64 number and
/// # ending with an array of `dim` f64 numbers
/// lower:
///   - <array of f64, length 1>
///   - <array of f64, length 2>
///   - . . .
///   - <array of f64, length `dim`>
/// # number of observations to generate for the experiment; must be greater than 0
/// num: <int>
/// # seed to use for the experiment (optional, default local entropy)
/// seed: <int>
/// # whether or not to split the sample for nearest neighbours (optional, default false)
/// split: true
/// ```
/// 
/// # Panics
/// 
/// It will panic if the Yaml document doesn't have the right format.
fn yaml_to_options(doc: Yaml) -> Options {

    let dim: usize = doc["dim"].as_i64().expect("Must specify a `dim` (dimenstion) as integer") as usize;
    assert!(dim > 0 && dim <= 4, "dim (dimension) must be greater than 0 and less or equal to 4");

    let loc: Vec<f64> = (0..dim).map(|ix| doc["loc"][ix].as_f64().expect("Must specify `loc` as float array of length `dim`")).collect();

    let lower: Vec<Vec<f64>> = (0..dim)
    .map(|ix| (0..ix+1)
              .map(|jx| doc["lower"][ix][jx].as_f64().expect(format!("`lower` error at {},{}", ix, jx).as_str()))
              .collect())
    .collect();

    let _: Vec<_> = (0..dim).map(|ix| assert!(lower[ix][ix] > 0.0, "`lower` diagonal must have positive entries")).collect();

    let num: usize = doc["num"].as_i64().expect("Must specify a number of observations in the sample") as usize;
    assert!(num > 0, "`num` must be greater than zero");

    let seed: Option<u64> = match doc["seed"].as_i64() {
        Some(v) => Some(v as u64),
        None => None
    };

    let split: Option<bool> = doc["split"].as_bool();

    Options {
        dim,
        loc,
        lower,
        num,
        seed,
        split,
    }

}

/// Reads options from file; defaults to `config.yaml` but another name can be provided on
/// the command line.
/// # Panics
/// Panics if file is not found, if it can't be parsed as YAML or if it doesn't have the information
/// in the right format, see [`yaml_to_options`].
fn get_options() -> Options {
    let mut file_name = "config.yaml".to_string();

    {
        let mut ap = ArgumentParser::new();
        ap.set_description("Minimum Helliger Distance Estimator for Cauchy-distributed 1D sample and model");
        ap.add_option(&["-v", "--version"],
            Print(env!("CARGO_PKG_VERSION").to_string()), "Show version");
        ap.refer(&mut file_name)
            .add_option(&["-c", "--config"], Store, "Path to config.yaml");
        ap.parse_args_or_exit();
    }

    let content: String = fs::read_to_string(file_name).unwrap();
    let docs = YamlLoader::load_from_str(content.as_str()).unwrap();
    assert_eq!(docs.len(), 1, "Configuration must have one yaml dictionary");
    yaml_to_options(docs[0].clone())
}

/// Generates a multivariate Cauchy-distributed sample based on configuration given in Options
/// Generates (X_1, X_2, ... , X_d) as individual samples from a Cauchy distribution
/// and then transforms using matrix L location P into L X + P. 
fn generate_sample(options: &Options) -> Vec<Array1<f64>> {
    let mut rng: ChaCha8Rng = match options.seed {
        Some(val) => ChaCha8Rng::seed_from_u64(val),
        None => ChaCha8Rng::from_entropy(),
    };

    let lower_matrix: Array2<f64> = {
        let mut res = Array::zeros((options.dim, options.dim));
        for (mut dst, row_option) in zip(res.rows_mut().into_iter(), options.lower.iter()) {
            let arr = Array1::from_vec(row_option.to_vec());
            let len = row_option.len();
            dst.slice_mut(s![..len]).assign(&arr);
        }
        res
    };

    let loc = Array1::from_vec(options.loc.clone());
    let matrix = &lower_matrix.dot(&lower_matrix.t());

    let dist: Cauchy = Cauchy::new(0.0, 1.0).unwrap();

    // create sample from them
    let sample = Vec::from_iter((0..options.num).map(
        |_| (0..options.dim).map(|_| dist.sample(&mut rng)).collect::<Array1<f64>>()
    ));

    // now apply matrix and location transformation to sample
    let sample = sample.iter().map(|row| row.dot(matrix) + &loc).collect();

    sample
}

/// Calculate simple statistics from flat Vec, i.e. minimum, median and maximum
fn min_median_max(numbers: &Vec<f64>) -> (f64, f64, f64) {

    let mut to_sort = numbers.clone();
    to_sort.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mid = numbers.len() / 2;
    let med = if numbers.len() % 2 == 0 {
        (numbers[mid - 1] + numbers[mid]) / 2.0
    } else {
        numbers[mid]
    };
    (to_sort[0], med, to_sort[numbers.len()-1])
}

/// Given the sample, provide an initial CauchyModel with sensible initial location and diagonal
/// matrix as starting point for the inverse.
fn cauchy_model<B: Backend>(dim: usize, sample: &Vec<Array1<f64>>, device: B::Device) -> CauchyModel<B> {

    // flatten the data to obtain min, max and median
    let flat: Vec<f64> = sample.into_iter().flat_map(|arr| arr.clone().into_raw_vec_and_offset().0).collect();
    let (v_min, v_med, v_max) = min_median_max(&flat);

    // use median for location
    let loc = vec![v_med; dim];
    let loc: Tensor<B, 1> = Tensor::from_data(loc.as_slice(), &device);

    // use min and max to estimate a simple scale, and populate just the diagonal for the initial
    // Cholesky decomposition
    let lower_size = dim * (dim - 1) / 2;
    let scale = (v_max - v_min) / (sample.len() as f64);
    let lower: Tensor<B, 1> = {
        let mut res = Array::zeros((lower_size,));
        let mut px = 0;
        for ix in 0..dim-1 {
            res[px] = scale;
            px += ix + 2;
        }
        Tensor::from_data(res.as_slice().unwrap(), &device)
    };

    CauchyModel {
        dim: dim,
        loc: Param::from_tensor(loc),
        diagonal: Param::from_tensor(Tensor::ones(Shape::new([dim]), &device)),
        lower: Param::from_tensor(lower),
    }
}

type AutoBE = Autodiff<NdArray<f64, i64>>;

fn main() {
    let options = get_options();

    let device: <AutoBE as Backend>::Device = Default::default();
    let sample = generate_sample(&options);
    let model = cauchy_model::<AutoBE>(options.dim, &sample, device);
    let sample = sample.iter().map(|row| Tensor::from_data(TensorData::new(row.clone().into_raw_vec(), [model.dim]), &device)).collect();

    println!("Starting params");
    println!("Loc: {:?}", model.loc.val().clone());
    println!("Matrix: {:?}\n", model.matrix_and_factor());

    let split: bool = options.split.unwrap_or_default();

    let (iters, model) = run::<AutoBE, CauchyModel<AutoBE>>(
        model,
        sample,
        split,
        device,
    );

    println!("Final params (iters={})", iters);
    println!("Loc: {:?}", model.loc.val().clone());
    println!("Matrix: {:?}\n", model.matrix_and_factor());
}

#[cfg(test)]
mod tests {

    use burn::tensor::{ElementConversion, Float};

    use super::*;

    fn check_close<B: Backend, const D: usize>(a: Tensor<B, D, Float>, b: Tensor<B, D, Float>) {
        let resp = (a.clone() - b.clone()).max_abs();
        let msg = format!("TENSORS DIFFER\nGot >> \n{:?}\nExpected >> \n{:?}\n", a, b);
        assert!(resp.into_scalar().elem::<f64>() < 1E-5, "{}", msg);
    }

    #[test]
    fn test_matrix_2a() {
        let device: <AutoBE as Backend>::Device = Default::default();
        let model: CauchyModel<AutoBE> = CauchyModel {
            dim: 2,
            loc: Param::from_tensor(Tensor::from_floats([1.0, 2.0], &device)),
            diagonal: Param::from_tensor(Tensor::ones(Shape::new([2]), &device)),
            lower: Param::from_tensor(Tensor::from_floats([-1.0], &device)),
        };
        let (matrix, _factor) = model.matrix_and_factor();
        let exp_matrix: Tensor<AutoBE, 2> = Tensor::<AutoBE, 1>::from_floats([1.0, 0.0, -1.0, 1.0], &device).reshape([2, 2]);
        check_close(matrix, exp_matrix);
    }

    #[test]
    fn test_matrix_2b() {
        let device: <AutoBE as Backend>::Device = Default::default();
        let model: CauchyModel<AutoBE> = CauchyModel {
            dim: 2,
            loc: Param::from_tensor(Tensor::from_floats([1.0, 2.0], &device)),
            diagonal: Param::from_tensor(Tensor::from_floats([1.0, (2.0f64).sqrt()], &device)),
            lower: Param::from_tensor(Tensor::from_floats([0.0], &device)),
        };
        let (matrix, factor) = model.matrix_and_factor();
        let exp_matrix: Tensor<AutoBE, 2> = Tensor::<AutoBE, 1>::from_floats([1.0, 0.0, 0.0, 2.0], &device).reshape([2, 2]);
        check_close(matrix, exp_matrix);
        let exp_factor: Tensor<AutoBE, 1> = Tensor::<AutoBE, 1>::from_floats([2.0], &device);
        check_close(factor, exp_factor);
    }

    #[test]
    fn test_matrix_3() {
        let device: <AutoBE as Backend>::Device = Default::default();
        let model: CauchyModel<AutoBE> = CauchyModel {
            dim: 3,
            loc: Param::from_tensor(Tensor::from_floats([1.0, 2.0, 0.0], &device)),
            diagonal: Param::from_tensor(Tensor::from_floats([2.0, 0.3, 0.5], &device)),
            lower: Param::from_tensor(Tensor::from_floats([-1.0, -2.0, 4.0], &device)),
        };
        let (matrix, factor) = model.matrix_and_factor();
        let exp_matrix: Tensor<AutoBE, 2> = Tensor::<AutoBE, 1>::from_floats([4.0,0.0,0.0,-1.0,0.09,0.0,-2.0,4.0,0.25], &device).reshape([3, 3]);
        check_close(matrix, exp_matrix);
        let exp_factor: Tensor<AutoBE, 1> = Tensor::<AutoBE, 1>::from_floats([4.0 * 0.09 * 0.25], &device);
        check_close(factor, exp_factor);
    }

    #[test]
    fn test_cauchy_pdf_1() {
        let device: <AutoBE as Backend>::Device = Default::default();
        let model: CauchyModel<AutoBE> = CauchyModel {
            dim: 1,
            loc: Param::from_tensor(Tensor::from_floats([0.0], &device)),
            diagonal: Param::from_tensor(Tensor::from_floats([1.0], &device)),
            lower: Param::from_tensor(Tensor::from_floats([0.0; 0], &device)),
        };
        let data: Tensor<AutoBE, 2> = Tensor::from_data([[0.0], [1.0]], &device);
        let vals = model.pdf(&data);
        let exp_vals: Tensor<AutoBE, 1> = Tensor::from_data([0.31831, 0.159155], &device);
        check_close(vals, exp_vals);
    }

    #[test]
    fn test_cauchy_pdf_2() {
        let device: <AutoBE as Backend>::Device = Default::default();
        let model: CauchyModel<AutoBE> = CauchyModel {
            dim: 2,
            loc: Param::from_tensor(Tensor::from_floats([0.0, 1.0], &device)),
            diagonal: Param::from_tensor(Tensor::from_floats([1.0, (2.0f64).sqrt()], &device)),
            lower: Param::from_tensor(Tensor::from_floats([-1.0], &device)),
        };
        let data: Tensor<AutoBE, 2> = Tensor::from_data([[0.0, 0.0], [1.0, 1.5], [0.0, 1.0]], &device);
        let vals = model.pdf(&data);
        let exp_vals: Tensor<AutoBE, 1> = Tensor::from_data([0.0405285,0.101321,0.202642], &device);
        check_close(vals, exp_vals);
    }

}
