use ndarray::{arr1, Array1, Array2};
use ndarray_linalg::Norm;
use super::activations::Activation;

pub trait Loss {
    /// Compute the squared error between `y` and `y_pred`.
    fn loss(&self, y_true: Array2<f32>, y_pred: Array2<f32>) -> f32;

    fn grad(
        &self,
        y_true: Array2<f32>,
        y_pred: Array2<f32>,
        parameters: &GradiantParameters,
    ) -> f64;
}

pub trait Grad {
    fn grad(&self, y_true: Array2<f32>, y_pred: Array2<f32>) -> Array1<f32>;
}

pub enum GradiantParameters {
    Activation(Box<dyn Activation>),
    T_LOG_VAR(Array2<f32>),
    // Y_REAL_WITH_INTERPOLATE((Option<Array1<f32>, Option<Array2<f32>>>))
    None,
}
/// The accumulated parameter gradients.

struct Gradients {}
/// The loss parameter values.

struct Parameters {}
/// The loss hyperparameter values.

struct HyperParameters {}
/// Useful intermediate values computed during the loss computation.

struct DerivedVariables {}

enum InitParamters {
    GlorotNormal,
    GlorotUniform,
    HeNormal,
    HeUniform
}

pub struct SquaredError {

}

impl SquaredError {
    pub fn new() -> Self {
        Self {}
    }
}

impl Loss for SquaredError {
    fn loss(&self, y_true: Array2<f32>, y_pred: Array2<f32>) -> f32 {
        todo!()
    }

    fn grad(
        &self,
        y_true: Array2<f32>,
        y_pred: Array2<f32>,
        parameters: &GradiantParameters,
    ) -> f64 {
        match parameters {
            GradiantParameters::Activation(x) => todo!(),
            GradiantParameters::T_LOG_VAR(_) => todo!(),
            GradiantParameters::None => todo!(),
        }
    }
}

pub struct CrossEntropy;

pub struct VAELoss;

pub struct WGAN_GPLoss;

pub struct NCELoss;

pub enum Losses {
    SquaredError(SquaredError),
    CrossEntropy(CrossEntropy),
    VAELoss(VAELoss),
    WGAN_GPLoss(WGAN_GPLoss),
    NCELoss(NCELoss),
}

impl Loss for Losses {
    fn loss(&self, y_true: Array2<f32>, y_pred: Array2<f32>) -> f32 {
        todo!()
    }

    fn grad(
        &self,
        y_true: Array2<f32>,
        y_pred: Array2<f32>,
        parameters: &GradiantParameters,
    ) -> f64 {
        todo!()
    }
}
