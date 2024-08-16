use core::fmt::{Debug, Display};



pub trait Bandit: Display + Debug {
    type Step;
    type NArms;

    fn hyperparameter(&self);
    fn oracle_payoff(&self);

    fn pull(&self);

    fn reset(&self);

    fn __pull(&self);
}

pub struct MultinomialBandit {

}

impl MultinomialBandit {

}

pub struct BernoulliBandit {

}

pub struct GaussianBandit {}

pub struct ShortestPathBandit {}

pub struct ContextualBernoulliBandit {}

pub struct ContextualLinearBandit {}