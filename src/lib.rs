use burn::{
    module::{AutodiffModule, ModuleVisitor, ParamId},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::{Backend, Tensor},
    tensor::backend::AutodiffBackend,
};

use core::f64;
use std::f64::consts::PI;

use linfa_nn::{BallTree, distance::L1Dist, NearestNeighbour};
use ndarray::{Array, array};

use burn::tensor::ElementConversion;

/// A Burn Module must implement a `pdf` on data function
pub trait ModelTrait<B: AutodiffBackend>: AutodiffModule<B> {
    fn pdf(&self, data: &Tensor<B, 1>) -> Tensor<B, 1>;
}

/// This `forward` function calculates the estimate for
/// squared Hellinger distance. At the moment, it assumes that
/// the volume balls do not depend on the parameters of the model.
pub fn forward<B: AutodiffBackend, M: ModelTrait<B>>(
    model: &M,
    data: &Tensor<B, 1>,
    balls: &Tensor<B, 1>
) -> Tensor<B, 1> {
    let pdf = model.pdf(data);
    // now calculate Hellinger Distance squared estimator, eqs (3) & (4) in paper
    let v = (pdf * balls.clone()).powf_scalar(0.5);
    let num = data.shape().dims[0];
    let factor = - 2.0 / ((num as f64) * PI).sqrt();
    v.sum() * factor + 1.0
}

/*
fn calculate_balls<B: Backend>(data: &Vec<f64>, split: bool, device: &B::Device) -> (Tensor<B, 1>, Tensor<B, 1>) {

    // considered that the sample could be split to ensure i.i.d terms in the sum
    // but there were no apparent benefits; leaving this legacy in case need to investigate
    // again
    let num = data.len();
    let data1 = if split { &data[0..(num / 2)] } else { &data[..] };  // slice used for calculate volume to nearest
    let data2 = if split { &data[(num / 2)..] } else { &data[..] };   // slice used to iterate points

    let algo = BallTree::new();
    let arr = Array::from_shape_vec([data1.len(), 1], data1.to_vec()).unwrap();
    let arr = arr.view();
    let nn_index = algo.from_batch(&arr, L1Dist).unwrap();
    let pos_nearest = 1;

    let radii: Vec<f64> = data2.iter()
        .map(|pt: &f64| (nn_index.k_nearest((array![*pt]).view(), pos_nearest + 1).unwrap(), pt))
        .map(|resp: (Vec<(ndarray::ArrayBase<ndarray::ViewRepr<&f64>, ndarray::Dim<[usize; 1]>>, usize)>, &f64)|
                    (resp.1 - resp.0[pos_nearest].0[0]).abs())  // distance to nearest neighbour
        .map(|v: f64| v * 2.0)                        // ball volume in dimension 1
        .collect();

    (
        Tensor::from_data(data2, device),
        Tensor::from_data(radii.as_slice(), device)
    )
}
*/

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
            let val: f64 = tensor.grad(&self.grads).unwrap().into_scalar().elem();
            self.is_less = val.abs() < self.epsilon;
        }
    }
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

/*
pub fn run<B: AutodiffBackend>(device: B::Device) {
    create_artifact_dir(ARTIFACT_DIR);
    // Config
    let config_optimizer = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5)));
    let config = MnistTrainingConfig::new(config_optimizer);
    B::seed(config.seed);

    let model = Model::<B>::new(&device);

    // Data
    let batcher = MnistBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());
    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    // Model
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new(
            &LossMetric::<B>::new(),
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, config.optimizer.init(), 1e-4);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    model_trained
        .save_file(
            format!("{ARTIFACT_DIR}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
}
*/

pub fn run<B: AutodiffBackend, M: ModelTrait<B>>(
    mut model: M,
    vec: Vec<f64>,
    split: bool,
    device: B::Device,
) -> (usize, M) {

    let config = TrainingConfig {
        num_runs: 1000,
        lr: 0.25,
        config_optimizer: AdamConfig::new(),
    };

    let balls = (Tensor::from_data([1.0, 2.0], &device), Tensor::from_data([1.0, 2.0], &device)); // calculate_balls::<B>(&vec, split, &device);

    let mut optimizer = config.config_optimizer.init::<B, M>();
    let epsilon: f64 = 0.000001;

    let mut ix = 1;
    while ix <= config.num_runs {

        let hd = forward(&model, &balls.0, &balls.1);

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
