use std::{
    fmt::{Debug, Display},
    num,
};

use ndarray::Array2;

use super::schedulers::Schedulers;

#[derive(Debug, Clone)]
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
#[derive(Debug, Copy, Clone)]
struct Cache {
    
}

impl Default for Cache {
    fn default() -> Self {
        Self {}
    }
}
#[derive(Debug, Clone)]
pub enum Optimizers {
    SGD(SGD),
    AdaGrad(AdaGrad),
    RMSProp(RMSProp),
    Adam(Adam),
}

impl Optimizer for Optimizers {
    fn step(&mut self) {
        todo!()
    }

    fn reset_step(&mut self) {
        todo!()
    }

    fn update(
        &mut self,
        param: &Array2<f64>,
        param_grad: &Array2<f64>,
        param_name: &str,
        cur_loss: Option<f64>,
    ) -> Array2<f64> {
        todo!()
    }
}

impl Display for Optimizers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Optimizers::SGD(_) => todo!(),
            Optimizers::AdaGrad(_) => todo!(),
            Optimizers::RMSProp(_) => todo!(),
            Optimizers::Adam(_) => todo!(),
        }
    }
}

pub trait Optimizer: Display + Clone + Debug {
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
#[derive(Debug, Clone)]
pub struct SGD {
    hyperparams: HyperParameters,
    cache: Cache,
    curl_step: f32,
    lr_scheduler: Schedulers,
}

impl SGD {

}

impl Default for SGD {
    fn default() -> Self {
        Self { hyperparams: Default::default(), cache: Default::default(), curl_step: Default::default(), lr_scheduler: Default::default() }
    }
}
#[derive(Debug, Clone)]
pub struct AdaGrad {
    hyperparams: HyperParameters,
    cache: Cache,
    curl_step: f32,
    lr_scheduler: Schedulers,
}

impl AdaGrad {

}
#[derive(Debug, Clone)]
pub struct RMSProp {
    hyperparams: HyperParameters,
    cache: Cache,
    curl_step: f32,
    lr_scheduler: Schedulers,
}

impl RMSProp {

}
#[derive(Debug, Clone)]
pub struct Adam {
    hyperparams: HyperParameters,
    cache: Cache,
    curl_step: f32,
    lr_scheduler: Schedulers,
}


