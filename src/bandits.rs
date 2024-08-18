use core::fmt::{Debug, Display};

use ndarray::{Array1, Array2};

pub trait Bandit: Display + Debug {
    type Step;
    type NArms;

    fn hyperparameter(&self) -> Hyperparameters;
    fn oracle_payoff(&self, context: Option<Array1<f64>>) -> (f64, usize);
    fn pull(&mut self, arm_id: usize, context: Option<Array1<f64>>) -> f64;
    fn reset(&mut self);
}

// Struct to hold the hyperparameters of the bandit
#[derive(Debug, Clone, PartialEq)]
pub struct Hyperparameters {
    id: String,
    payoffs: Option<Array2<f64>>,
    payoff_probs: Option<Array2<f64>>,
    K: Option<usize>,
    D: Option<usize>,
    payoff_variance: Option<Vec<f64>>,
    thetas: Option<Array2<f64>>,
}

pub struct MultinomialBandit {
    payoffs: Array2<f64>,
    payoff_probs: Array2<f64>,
    arm_evs: Vec<f64>,
    best_ev: f64,
    best_arm: usize,
    step: usize,
    n_arms: usize,
    hyperparameters: Hyperparameters,
}

impl MultinomialBandit {}

pub struct BernoulliBandit {
    payoffs: Array2<f64>,
    arm_evs: Vec<f64>,
    best_ev: f64,
    best_arm: usize,
    step: usize,
    n_arms: usize,
    hyperparameters: Hyperparameters,
}

pub struct GaussianBandit {
    means: Vec<f64>,
    variances: Vec<f64>,
    best_ev: f64,
    best_arm: usize,
    step: usize,
    n_arms: usize,
    hyperparameters: Hyperparameters,
}

pub struct ShortestPathBandit {}

pub struct ContextualBernoulliBandit {}

pub struct ContextualLinearBandit {}
