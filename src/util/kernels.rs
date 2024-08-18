use std::fmt::{Debug, Display};

use ndarray::ArrayD;



pub trait KernelBase: Display + Debug {

    fn kernel(&self, x: &ArrayD<f64>, y: &ArrayD<f64>) -> f64;
}
#[derive(Debug)]
pub struct KernelParameters {
    id: String,
    d: isize,
    gamma: Option<f64>,
    c0: f64,
}

#[derive(Debug)]
pub struct KernelInitializer {
    param: KernelParameters,
}