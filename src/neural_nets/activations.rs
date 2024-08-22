use bon::builder;
use ndarray::{ArrayBase, ArrayD, IxDyn};
use core::f64;
use std::fmt::{self, Debug, Formatter};
/// Trait representing the base functionality of an activation function.
/// This is equivalent to the `ActivationBase` class in Python.
pub trait ActivationBase: Debug {
    /// Apply the activation function to an input.
    fn call(&self, z: &ArrayD<f64>) -> ArrayD<f64> {
        let reshaped_z: ArrayD<f64> = if z.ndim() == 1 {
            z.clone().into_shape_with_order(IxDyn(&[1, z.len()])).unwrap()
        } else {
            z.clone()
        };
        self.fn_impl(&z)
    }

    /// The core activation function to be implemented by specific activations.
    fn fn_impl(&self, z: &ArrayD<f64>) -> ArrayD<f64>;

    /// Compute the gradient of the activation function with respect to the input.
    fn grad(
        &self,
        x: &ArrayD<f64>,
        kwargs: Option<&std::collections::HashMap<String, f64>>,
    ) -> ArrayD<f64>;
}

pub trait Activation {
    fn function(&self, z: ndarray::ArrayD<f64>) -> f64;

    fn grad(&self, x: ArrayD<f64>);
}

#[derive(Default, Debug, Copy, Clone)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn function(&self, z: ndarray::ArrayD<f64>) -> f64 {
        todo!();

        let result = 1.0 / (1.0 + (-z).exp());
        
    }

    fn grad(&self, x: ArrayD<f64>) {
        let fn_x = self.function(x);

        let result = fn_x * (1.0 - fn_x);
        todo!()
    }
}
/// A rectified linear activation function.
/// Notes
/// -----
/// "ReLU units can be fragile during training and can "die". For example, a
/// large gradient flowing through a ReLU neuron could cause the weights to
/// update in such a way that the neuron will never activate on any datapoint
/// again. If this happens, then the gradient flowing through the unit will
/// forever be zero from that point on. That is, the ReLU units can
/// irreversibly die during training since they can get knocked off the data
/// manifold.
///
///For example, you may find that as much as 40% of your network can be "dead"
///(i.e. neurons that never activate across the entire training dataset) if
///the learning rate is set too high. With a proper setting of the learning
///rate this is less frequently an issue." [*]_
///
///References
///----------
///.. [*] Karpathy, A. "CS231n: Convolutional neural networks for visual recognition."
#[derive(Default, Debug)]
pub struct ReLU;

impl ActivationBase for ReLU {
    fn fn_impl(&self, z: &ArrayD<f64>) -> ArrayD<f64> {
       z.clamp(0.0, f64::INFINITY)
    }

    fn grad(
        &self,
        x: &ArrayD<f64>,
        kwargs: Option<&std::collections::HashMap<String, f64>>,
    ) -> ArrayD<f64> {
        todo!()
    }
}

#[derive(Default)]
pub struct LeakyReLU;

#[derive(Default)]
pub struct GELU;

#[derive(Default)]
pub struct Tanh;

#[derive(Default, Debug)]
pub struct Affine {
    slope: f64,
    intercept: f64,
}

impl Affine {
    /// Create a new Affine activation function with the specified slope and intercept.
    pub fn new(slope: f64, intercept: f64) -> Self {
        Affine { slope, intercept }
    }
}

impl ActivationBase for Affine {
    fn fn_impl(&self, z: &ArrayD<f64>) -> ArrayD<f64> {
        z.mapv(|x| self.slope * x + self.intercept)
    }

    fn call(&self, z: &ArrayD<f64>) -> ArrayD<f64> {
        let reshaped_z: ArrayD<f64> = if z.ndim() == 1 {
            z.clone().into_shape(IxDyn(&[1, z.len()])).unwrap()
        } else {
            z.clone()
        };
        self.fn_impl(&z)
    }

    fn grad(
        &self,
        x: &ArrayD<f64>,
        kwargs: Option<&std::collections::HashMap<String, f64>>,
    ) -> ArrayD<f64> {
        todo!()
    }
}

impl std::fmt::Display for Affine {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Affine(slope={}, intercept={})",
            self.slope, self.intercept
        )
    }
}

#[derive(Default, Debug, Clone)]
pub struct Identity;

impl ActivationBase for Identity {
    fn fn_impl(&self, z: &ArrayD<f64>) -> ArrayD<f64> {
        todo!()
    }

    fn grad(
        &self,
        x: &ArrayD<f64>,
        kwargs: Option<&std::collections::HashMap<String, f64>>,
    ) -> ArrayD<f64> {
        todo!()
    }
}

#[derive(Default)]
pub struct ELU;

#[derive(Default)]
pub struct Exponential;

#[derive(Default)]
pub struct SELU;

#[derive(Default)]
pub struct HardSigmoid;

#[derive(Default)]
pub struct SoftPlus;
