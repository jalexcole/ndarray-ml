pub struct BernoulliVAE;

pub struct BernoulliVAEBuilder {
    T: usize,
    latent_dim: usize,
    enc_conv1_pad: usize,
    enc_conv2_pad: usize,
    enc_conv1_out_ch: usize,
    enc_conv2_out_ch: usize,
    enc_conv1_stride: usize,
    enc_pool1_stride: usize,
    enc_conv2_stride: usize,
    enc_pool2_stride: usize,
    enc_conv1_kernel_shape: [usize; 2],
    enc_pool1_kernel_shape: [usize; 2],
    enc_conv2_kernel_shape: [usize; 2],
    enc_pool2_kernel_shape: (usize, usize),
    optimizer: String,
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

    fn build(&self) -> BernoulliVAE {
        todo!()
    }
}

pub struct Word2Vec;

pub struct WGAN_GP;

pub trait Model {}

pub enum Models {}

impl Model for Models {}
