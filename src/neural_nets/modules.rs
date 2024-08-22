use ndarray::{Array4, ArrayD};

use super::{
    layers::Conv1D,
    optimizers::{Optimizer, Optimizers},
};

pub trait Module {
    fn init_params(&self);
    fn forward(&self, x_main: Array4<f32>, x_skip: Option<Array4<f32>>) -> (Array4<f32>, Array4<f32>);
    fn backward(&self, out: ArrayD<f64>);
    fn components(&self) -> Compoments;
}

pub struct Compoments {}

struct Init {}

pub struct paramters {

}

pub struct hyperparameters {

}

pub struct DerivedVariables {

}

struct Conc1D<'a> {
    Stride: usize,
    pad: String,
    init: Init,
    kernel_wdidth: usize,
    dilation: &'a usize,
    out_ch: &'a usize,
    optimizer: &'a Optimizers,
}

pub struct WavenetResidualModule {
    init: Init,
    dilation: usize,
    optimizer: Optimizers,
    ch_residual: usize,
    ch_dilation: usize,
    kernel_width: usize,
    conv_dilation: Option<Conv1D>,
}

pub struct SkipConnectionIdentityModule {
    init: Init,
    dilation: usize,
    optimizer: Optimizers,
    ch_residual: usize,
    ch_dilation: usize,
    kernel_width: usize,
    conv_dilation: Option<Conv1D>,
    conv_1x1: Option<Conv1D>,

}

pub struct SkipConnectionConvModule {

}

pub struct BidirectionalLSTM {

}

pub struct MultiHeadedAttentionModule {
    
}