use super::activations::{ActivationBase, Activations, Identity, ReLU};

struct Parameters {

}

struct HyperParameters {

}

pub trait Initializer {

}

#[derive(Debug)]
pub struct ActivationInitializer {
    param: Option<String>,
}

impl ActivationInitializer {
    /// Create a new ActivationInitializer. If `param` is `None`, it defaults to the identity function.
    pub fn new(param: Option<String>) -> Self {
        ActivationInitializer { param }
    }

    /// Initialize the activation function based on the parameter.
    pub fn init(&self) -> Box<dyn ActivationBase> {
        match &self.param {
            None => Box::new(Identity),
            Some(param) => {
                if let Ok(act_fn) = self.init_from_str(param) {
                    act_fn
                } else {
                    panic!("Unknown activation: {}", param)
                }
            }
        }
    }

    /// Initialize activation function from the `param` string.
    fn init_from_str(&self, act_str: &str) -> Result<Box<dyn ActivationBase>, String> {
        match act_str.to_lowercase().as_str() {
            "relu" => Ok(Box::new(ReLU)),
            "identity" => Ok(Box::new(Identity)),
            // Additional cases for other activation functions...
            _ => Err(format!("Unknown activation: {}", act_str)),
        }
    }
}



pub struct SchedulerInitializer {
    param: Parameters,
    lr: usize
}

pub struct OptimizerInitializer {
    param: Parameters,

}

pub struct WeightInitializer {
    act_fn: Activations,

}

impl WeightInitializer {
    fn new(act_fn_str: String, mode: Mode) -> Self {
        

        match mode {
            Mode::he_normal => todo!(),
            Mode::he_uniform => todo!(),
            Mode::glorot_normal => todo!(),
            Mode::glorot_uniform => todo!(),
            Mode::std_normal => todo!(),
            Mode::trunc_normal => todo!(),
        }

        Self { act_fn: todo!() }
    }

    fn calc_glorot_gain(&self) -> f32 {
        let mut gain = 1.0;

        match self.act_fn {
            Activations::Tanh(_) => gain = 5.0 / 3.0,
            Activations::ReLu(_) => gain = f32::sqrt(2.0),
            Activations::LEAKY_RELU(_) => todo!(),
            _ => gain = 1.0,
        }

        gain
    }
}


#[derive(Debug)]
enum Activation {
    ReLu,
    Tanh,
    SELU,
    Sigmoid,
    Identity,
    HardSigmoid,
    SoftPlus,
    Exponential,
    ADDINE,
    LEAKY_RELU,
    GELU,
    ELU,
}

enum Mode {
    he_normal,
    he_uniform,
    glorot_normal,
    glorot_uniform,
    std_normal,
    trunc_normal,
}

impl Default for Mode {
    fn default() -> Self {
        Mode::glorot_uniform
    }
}

struct InitializerParameters {}

pub enum Initializers {
    Activation(ActivationInitializer),
    Scheduler(SchedulerInitializer),
    Optimizer(OptimizerInitializer),
    Weight(WeightInitializer),
}
impl Initializer for Initializers {
    
}
