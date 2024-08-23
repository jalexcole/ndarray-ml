//! In mathematical optimization and decision theory, a loss function or cost function (sometimes also called an error function)[1] is a function that maps an event or values of one or more variables onto a real number intuitively representing some "cost" associated with the event. An optimization problem seeks to minimize a loss function. An objective function is either a loss function or its opposite (in specific domains, variously called a reward function, a profit function, a utility function, a fitness function, etc.), in which case it is to be maximized. The loss function could include terms from several levels of the hierarchy.

//! In statistics, typically a loss function is used for parameter estimation, and the event in question is some function of the difference between estimated and true values for an instance of data. The concept, as old as Laplace, was reintroduced in statistics by Abraham Wald in the middle of the 20th century.[2] In the context of economics, for example, this is usually economic cost or regret. In classification, it is the penalty for an incorrect classification of an example. In actuarial science, it is used in an insurance context to model benefits paid over premiums, particularly since the works of Harald Cramér in the 1920s.[3] In optimal control, the loss is the penalty for failing to achieve a desired value. In financial risk management, the function is mapped to a monetary loss.

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
