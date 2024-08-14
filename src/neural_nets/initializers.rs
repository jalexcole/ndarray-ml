use super::activations::{ActivationBase, Identity, ReLU};


#[derive(Debug)]
pub struct ActivationInitializer{
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
            "relu" => {
                Ok(Box::new(ReLU))
            },
            "identity" => Ok(Box::new(Identity)),
            // Additional cases for other activation functions...
            _ => Err(format!("Unknown activation: {}", act_str)),
        }
    }
}

pub struct SchedulerInitializer;

pub struct OptimizerINitializer;

pub struct WeightInitializer;

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
    ADDINE, LEAKY_RELU, GELU, ELU
}
