use std::default;

pub struct LayerData;
pub struct LayerBase;

pub trait Layer {

    /// Unfreeze the layer parameters so they can be updated.
    fn freeze(&self);
    /// Erase all the layer's derived variables and gradients.
    fn flush_gradients(&self);

    fn set_params(&self, summary_dict: LayerParameters);
}

struct LayerParameters;

pub struct DotProductAttention;

pub struct RBM;

pub struct Add;

pub struct Multiply;

pub struct Flatten;

pub struct BatchNorm2d;

pub struct BatchNorm1d;

pub struct LayerNorm2D;

pub struct Embedding;

pub struct FullyConnected;

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
///         Parameters
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
    is_initialized: bool
}

impl RNN {
    fn new(n_out: usize) -> Self {
        Self { n_out: n_out, n_timesteps: None, is_initialized: false }
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
}

pub struct LSTM;
