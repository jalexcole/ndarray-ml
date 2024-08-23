use ndarray::{Array4, ArrayD};

use super::{
    layers::Conv1D,
    optimizers::{Optimizer, Optimizers},
};

pub trait Module {
    fn init_params(&self);
    fn forward(
        &self,
        x_main: Array4<f32>,
        x_skip: Option<Array4<f32>>,
    ) -> (Array4<f32>, Array4<f32>);
    fn backward(&self, out: ArrayD<f64>);
    fn components(&self) -> Compoments;
}

pub struct Compoments {}

impl Default for Compoments {
    fn default() -> Self {
        Self {}
    }
}

struct Init {}

pub struct Parameters {}

impl Default for Parameters {
    fn default() -> Self {
        Self {}
    }
}

pub struct hyperparameters {}

pub struct DerivedVariables {}

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

impl Module for WavenetResidualModule {
    fn init_params(&self) {
        todo!()
    }

    fn forward(
        &self,
        x_main: Array4<f32>,
        x_skip: Option<Array4<f32>>,
    ) -> (Array4<f32>, Array4<f32>) {
        todo!()
    }

    fn backward(&self, out: ArrayD<f64>) {
        todo!()
    }

    fn components(&self) -> Compoments {
        todo!()
    }
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

impl Module for SkipConnectionIdentityModule {
    fn init_params(&self) {
        todo!()
    }

    fn forward(
        &self,
        x_main: Array4<f32>,
        x_skip: Option<Array4<f32>>,
    ) -> (Array4<f32>, Array4<f32>) {
        todo!()
    }

    fn backward(&self, out: ArrayD<f64>) {
        todo!()
    }

    fn components(&self) -> Compoments {
        todo!()
    }
}

pub struct SkipConnectionConvModule {}
impl Module for SkipConnectionConvModule {
    fn init_params(&self) {
        todo!()
    }

    fn forward(
        &self,
        x_main: Array4<f32>,
        x_skip: Option<Array4<f32>>,
    ) -> (Array4<f32>, Array4<f32>) {
        todo!()
    }

    fn backward(&self, out: ArrayD<f64>) {
        todo!()
    }

    fn components(&self) -> Compoments {
        todo!()
    }
}
pub struct BidirectionalLSTM {}

impl Module for BidirectionalLSTM {
    fn init_params(&self) {
        todo!()
    }

    fn forward(
        &self,
        x_main: Array4<f32>,
        x_skip: Option<Array4<f32>>,
    ) -> (Array4<f32>, Array4<f32>) {
        todo!()
    }

    fn backward(&self, out: ArrayD<f64>) {
        todo!()
    }

    fn components(&self) -> Compoments {
        todo!()
    }
}

pub struct MultiHeadedAttentionModule {}

impl Module for MultiHeadedAttentionModule {
    fn init_params(&self) {
        todo!()
    }

    fn forward(
        &self,
        x_main: Array4<f32>,
        x_skip: Option<Array4<f32>>,
    ) -> (Array4<f32>, Array4<f32>) {
        todo!()
    }

    fn backward(&self, out: ArrayD<f64>) {
        todo!()
    }

    fn components(&self) -> Compoments {
        todo!()
    }
}

pub enum Modules {
    WavenetResidualModule(WavenetResidualModule),
    SkipConnectionIdentityModule(SkipConnectionIdentityModule),
    SkipConnectionConvModule(SkipConnectionConvModule),
    BidirectionalLSTM(BidirectionalLSTM),
    MultiHeadedAttentionModule(MultiHeadedAttentionModule),
}

impl Module for Modules {
    fn init_params(&self) {
        match self {
            Modules::WavenetResidualModule(m) => m.init_params(),
            Modules::SkipConnectionIdentityModule(m) => m.init_params(),
            Modules::SkipConnectionConvModule(m) => m.init_params(),
            Modules::BidirectionalLSTM(m) => m.init_params(),
            Modules::MultiHeadedAttentionModule(m) => m.init_params(),
        }
    }

    fn forward(
        &self,
        x_main: Array4<f32>,
        x_skip: Option<Array4<f32>>,
    ) -> (Array4<f32>, Array4<f32>) {
        match self {
            Modules::WavenetResidualModule(m) => m.forward(x_main, x_skip),
            Modules::SkipConnectionIdentityModule(m) => m.forward(x_main, x_skip),
            Modules::SkipConnectionConvModule(m) => m.forward(x_main, x_skip),
            Modules::BidirectionalLSTM(m) => m.forward(x_main, x_skip),
            Modules::MultiHeadedAttentionModule(m) => m.forward(x_main, x_skip),
        }
    }

    fn backward(&self, out: ArrayD<f64>) {
        match self {
            Modules::WavenetResidualModule(m) => m.backward(out),
            Modules::SkipConnectionIdentityModule(m) => m.backward(out),
            Modules::SkipConnectionConvModule(m) => m.backward(out),
            Modules::BidirectionalLSTM(m) => m.backward(out),
            Modules::MultiHeadedAttentionModule(m) => m.backward(out),
        }
    }

    fn components(&self) -> Compoments {
        match self {
            Modules::WavenetResidualModule(m) => m.components(),
            Modules::SkipConnectionIdentityModule(m) => m.components(),
            Modules::SkipConnectionConvModule(m) => m.components(),
            Modules::BidirectionalLSTM(m) => m.components(),
            Modules::MultiHeadedAttentionModule(m) => m.components(),
        }
    }
}
