use std::{default, ops::Index};

use ndarray::Array2;

pub struct LayerData;
pub struct LayerBase;

pub struct LayerSummary {
    layer: usize,
    parameters:usize,
    hyper_parameters: usize,
} 

pub trait Layer {
    /// Unfreeze the layer parameters so they can be updated.
    fn freeze(&self);
    /// Erase all the layer's derived variables and gradients.
    fn flush_gradients(&self);

    fn set_params(&self, summary_dict: LayerParameters);

    fn unfreeze(&self);

    fn update(&self);

    fn summary(&self) -> LayerSummary;

}

struct LayerParameters;

pub struct DotProductAttention;

impl Layer for DotProductAttention {
    fn freeze(&self) {
        todo!()
    }

    fn flush_gradients(&self) {
        todo!()
    }

    fn set_params(&self, summary_dict: LayerParameters) {
        todo!()
    }

    fn unfreeze(&self) {
        todo!()
    }

    fn update(&self) {
        todo!()
    }

    fn summary(&self) -> LayerSummary {
        todo!()
    }
}

pub struct RBM;

pub struct Add;

pub struct Multiply;

pub struct Flatten;

pub struct BatchNorm2d;

pub struct BatchNorm1d;

pub struct LayerNorm2D;

pub struct Embedding;

pub struct FullyConnected {
    is_initialized: bool,
    n_in: Array2<f64>,
}

impl FullyConnected {
    pub fn forward(&mut self, X: Array2<f64>, retain_derive: bool) -> Array2<f64> {
        if (!self.is_initialized) {
            // self.n_in = X.shape().index(1)
        }

        todo!()
    }

    fn _fwd(&self, X: Array2<f64>) {}
}

pub struct Softmax;

pub struct SparseEvolution;

pub struct Conv1D;

pub struct Conv2D;

pub struct Pool2D;

pub struct Deconv2D;

pub struct RNNCell;

pub struct LSTMCell;

///  A single vanilla (Elman)-RNN layer.
///
///         # Parameters
///         ----------
///         n_out : int
///             The dimension of a single hidden state / output on a given
///             timestep.
///         act_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
///             The activation function for computing ``A[t]``. Default is
///             `'Tanh'`.
///         init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
///             The weight initialization strategy. Default is `'glorot_uniform'`.
///         optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
///             The optimization strategy to use when performing gradient updates
///             within the :meth:`update` method.  If None, use the :class:`SGD
///             <numpy_ml.neural_nets.optimizers.SGD>` optimizer with default
///             parameters. Default is None.
pub struct RNN {
    n_out: usize,
    n_timesteps: Option<usize>,
    is_initialized: bool,
}

impl RNN {
    fn new(n_out: usize) -> Self {
        Self {
            n_out: n_out,
            n_timesteps: None,
            is_initialized: false,
        }
    }
}

impl Layer for RNN {
    fn freeze(&self) {
        todo!()
    }

    fn flush_gradients(&self) {
        todo!()
    }

    fn set_params(&self, summary_dict: LayerParameters) {
        todo!()
    }
    
    fn unfreeze(&self) {
        todo!()
    }
    
    fn update(&self) {
        todo!()
    }
    
    fn summary(&self) -> LayerSummary {
        todo!()
    }
}

pub struct LSTM;

pub enum Layers {
    FullyConnected(FullyConnected),
    Softmax(Softmax),
    SparseEvolution(SparseEvolution),
    Conv1D(Conv1D),
    Conv2D(Conv2D),
    Pool2D(Pool2D),
    Deconv2D(Deconv2D),
    RNN(RNN),
    LSTM(LSTM),
}

impl Layer for Layers {
    fn freeze(&self) {
        match self {
            Layers::FullyConnected(_) => todo!(),
            Layers::Softmax(_) => todo!(),
            Layers::SparseEvolution(_) => todo!(),
            Layers::Conv1D(_) => todo!(),
            Layers::Conv2D(_) => todo!(),
            Layers::Pool2D(_) => todo!(),
            Layers::Deconv2D(_) => todo!(),
            Layers::RNN(_) => todo!(),
            Layers::LSTM(_) => todo!(),
        }
    }

    fn flush_gradients(&self) {
        match self {
            Layers::FullyConnected(_) => todo!(),
            Layers::Softmax(_) => todo!(),
            Layers::SparseEvolution(_) => todo!(),
            Layers::Conv1D(_) => todo!(),
            Layers::Conv2D(_) => todo!(),
            Layers::Pool2D(_) => todo!(),
            Layers::Deconv2D(_) => todo!(),
            Layers::RNN(_) => todo!(),
            Layers::LSTM(_) => todo!(),
        }
    }

    fn set_params(&self, summary_dict: LayerParameters) {
        match self {
            Layers::FullyConnected(_) => todo!(),
            Layers::Softmax(_) => todo!(),
            Layers::SparseEvolution(_) => todo!(),
            Layers::Conv1D(_) => todo!(),
            Layers::Conv2D(_) => todo!(),
            Layers::Pool2D(_) => todo!(),
            Layers::Deconv2D(_) => todo!(),
            Layers::RNN(_) => todo!(),
            Layers::LSTM(_) => todo!(),
        }
    }
    
    fn unfreeze(&self) {
        match self {
            Layers::FullyConnected(_) => todo!(),
            Layers::Softmax(_) => todo!(),
            Layers::SparseEvolution(_) => todo!(),
            Layers::Conv1D(_) => todo!(),
            Layers::Conv2D(_) => todo!(),
            Layers::Pool2D(_) => todo!(),
            Layers::Deconv2D(_) => todo!(),
            Layers::RNN(_) => todo!(),
            Layers::LSTM(_) => todo!(),
        }
    }
    
    fn update(&self) {
        match self {
            Layers::FullyConnected(_) => todo!(),
            Layers::Softmax(_) => todo!(),
            Layers::SparseEvolution(_) => todo!(),
            Layers::Conv1D(_) => todo!(),
            Layers::Conv2D(_) => todo!(),
            Layers::Pool2D(_) => todo!(),
            Layers::Deconv2D(_) => todo!(),
            Layers::RNN(_) => todo!(),
            Layers::LSTM(_) => todo!(),
        }
    }
    
    fn summary(&self) -> LayerSummary {
        match self {
            Layers::FullyConnected(_) => todo!(),
            Layers::Softmax(_) => todo!(),
            Layers::SparseEvolution(_) => todo!(),
            Layers::Conv1D(_) => todo!(),
            Layers::Conv2D(_) => todo!(),
            Layers::Pool2D(_) => todo!(),
            Layers::Deconv2D(_) => todo!(),
            Layers::RNN(_) => todo!(),
            Layers::LSTM(_) => todo!(),
        }
    }
}
