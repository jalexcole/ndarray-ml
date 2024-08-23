use ndarray::{Array2, Array4, ArrayD};

use super::{activations::Activations, optimizers::Optimizers};




struct HyperParameters {
    layer: String,
    T : usize,
    init: String,
    loss: String, 
    optimizer: Optimizers,
    latent_dim: usize,
    env_conv1: usize,
}

struct DerivedVariables {

}

struct Parameters {

}

pub struct BernoulliVAE;

impl BernoulliVAE {
    fn _init_params(&self) {

    }

    fn _build_encoder(&self) {

    }

    fn _build_decoder(&self) {

    }

    pub fn parameters(&self) -> Parameters {
        todo!()
    }

    fn _sample(&self, t_mean: Array2<f32>, t_log_var: Array2<f32>) -> Array2<f32> {
        todo!()
    }

    pub fn forward(&self, X_train: ArrayD<f64>) {
        
    }

    pub fn backward(&self) {

    }

    pub fn update(&self, cur_loss: f32) {

    }

    fn flush_gradients(&self) {

    }

    fn fit(&self, X_train: Array4<f32>, epochs: usize, batchsize: usize, verbos: bool) {
        
    }
}

/// A variational autoencoder (VAE) with 2D convolutional encoder and Bernoulli
/// input and output units.
pub struct BernoulliVAEBuilder {
    /// The dimension of the variational parameter `t`. Default is 5.
    T: usize,
    /// The dimension of the output for the first FC layer of the encoder.
    /// Default is 256.
    latent_dim: usize,
    ///
    enc_conv1_pad: usize,
    ///
    enc_conv2_pad: usize,
    ///
    enc_conv1_out_ch: usize,
    ///
    enc_conv2_out_ch: usize,
    ///
    enc_conv1_stride: usize,
    ///
    enc_pool1_stride: usize,
    ///
    enc_conv2_stride: usize,
    ///
    enc_pool2_stride: usize,
    ///
    enc_conv1_kernel_shape: [usize; 2],
    ///
    enc_pool1_kernel_shape: [usize; 2],
    ///
    enc_conv2_kernel_shape: [usize; 2],
    ///
    enc_pool2_kernel_shape: (usize, usize),
    /// The optimization strategy to use when performing gradient updates.
    /// If None, use the :class:`~numpy_ml.neural_nets.optimizers.SGD`
    /// optimizer with default parameters. Default is "RMSProp(lr=0.0001)".
    optimizer: String,
    /// The weight initialization strategy. Valid entries are
    /// {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform',
    /// 'std_normal', 'trunc_normal'}. Default is 'glorot_uniform'.
    init: String,
}



impl BernoulliVAEBuilder {
    fn new() -> Self {
        Self {
            T: 5,
            latent_dim: 256,
            enc_conv1_pad: 0,
            enc_conv2_pad: 0,
            enc_conv1_out_ch: 0,
            enc_conv2_out_ch: 32,
            enc_conv1_stride: 64,
            enc_pool1_stride: 1,
            enc_conv2_stride: 2,
            enc_pool2_stride: 1,
            enc_conv1_kernel_shape: [5, 5],
            enc_pool1_kernel_shape: [2, 2],
            enc_conv2_kernel_shape: [5, 5],

            enc_pool2_kernel_shape: (2, 2),
            optimizer: "RMSProp(lr=0.0001)".to_owned(),
            init: "glorot_uniform".to_owned(),
        }
    }

    
}
/// A word2vec model supporting both continuous bag of words (CBOW) and
/// skip-gram architectures, with training via noise contrastive
/// estimation.
pub struct Word2Vec;

pub struct WGAN_GP;

pub trait Model {
    fn hyperparameter(&self) -> HyperParameters;
    fn derived_variables(&self) -> DerivedVariables;
    fn gradients(&self) -> Parameters;
}

pub enum Models {
    Bernoulli(BernoulliVAE),
    Word2Vec(Word2Vec),
    WGAN_GP(WGAN_GP)
   
}

impl Model for Models {
    fn hyperparameter(&self) -> HyperParameters {
        todo!()
    }

    fn derived_variables(&self) -> DerivedVariables {
        todo!()
    }

    fn gradients(&self) -> Parameters {
        todo!()
    }
}


// Decoders //

trait Decoder {

}

struct FullyConnected<'a> {
    act_fn: Activations,
    n_out: usize,
    optimzer: &'a Optimizers
}

// Embedding

struct Embedding {
    
}
