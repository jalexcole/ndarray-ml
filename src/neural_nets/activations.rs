use ndarray::ArrayBase;



pub trait Activation {

    fn function(&self, z: ndarray::ArrayD<f64>) -> f64;

    fn grad(&self);
}

pub struct Sigmoid;

pub struct ReLU;

pub struct LeakyReLU;

pub struct GELU;

pub struct Tanh;

pub struct Affine;

pub struct Identity;

pub struct ELU;

pub struct Exponential;

pub struct SELU;

pub struct HardSigmoid;

pub struct SoftPlus;