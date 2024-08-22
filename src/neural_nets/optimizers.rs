use std::{
    fmt::{Debug, Display},
    num,
};

use ndarray::Array2;

pub enum Optimizers {
    SGD(SGD),
    AdaGrad(AdaGrad),
    RMSProp(RMSProp),
    Adam(Adam),
}

pub trait Optimizer: Display + Copy + Debug {
    fn step(&mut self);

    fn reset_step(&mut self);

    fn update(
        &mut self,
        param: &Array2<f64>,
        param_grad: &Array2<f64>,
        param_name: &str,
        cur_loss: Option<f64>,
    ) -> Array2<f64>;
}

pub struct SGD;

pub struct AdaGrad;

pub struct RMSProp;

pub struct Adam;

struct HyperParameters {
    id: String,
    /// Learning rate for update. Default is 0.001.
    lr: f64,
    /// Constant term to avoid divide-by-zero errors during the update calc.
    /// Default is 1e-7.
    eps: f64,
    /// The fraction of the previous update to add to the current update.
    /// If 0, no momentum is applied. Default is 0.
    momentum: f64,
    /// Rate of decay for the moving average. Typical values are [0.9, 0.99,
    /// 0.999]. Default is 0.9.
    decay: f64,
    /// The rate of decay to use for in running estimate of the second
    /// moment (variance) of the gradient. Default is 0.999.
    decay2: f64,
    /// If not None, all param gradients are scaled to have maximum l2 norm of
    /// `clip_norm` before computing update. Default is None.
    clip_norm: Option<f64>,
    /// The learning rate scheduler. If None, use a constant learning
    ///rate equal to `lr`. Default is None.
    lr_scheduler: String,
}

impl Default for HyperParameters {
    fn default() -> Self {
        Self {
            id: Default::default(),
            lr: 0.001,
            eps: 1.0e-7,
            momentum: 0.0,
            decay: 0.9,
            decay2: 0.999,
            clip_norm: None,
            lr_scheduler: String::from("lr"),
        }
    }
}
