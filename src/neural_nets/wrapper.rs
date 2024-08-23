use super::layers::{Layer, Layers};




pub struct WrapperSummary {

}

pub struct SummaryDict {

}

pub trait Wrapper {
    fn forward(&self);
    fn backward(&self);
    fn trainable(&self);
    fn parameters(&self);
    fn hyper_parameters(&self);
    fn derived_variables(&self);
    fn gradiants(&self);
    fn act_fn(&self);
    fn X(&self);
    fn freeze(&self);
    fn unfreeze(&self);
    fn flush_gradients(&self);
    fn update(&self);
    fn set_params(&self, summary_dict: SummaryDict);
    /// Return a dict of the layer parameters, hyperparameters, and ID.
    fn summary(&self) -> WrapperSummary;
}

trait WrapperBase: Wrapper {
    fn _wrapper_params(&self);
    fn _set_wrapper_params(&self);
}

pub struct Dropout {
    base_layer: Layers,
    p: f32,
    trainable: f64,

}

impl Dropout {
    fn new(wrapped_layer: &impl Layer, p: f32) -> Self {
        todo!()
    }
}

impl Wrapper for Dropout {
    fn forward(&self) {
        todo!()
    }

    fn backward(&self) {
        todo!()
    }

    fn trainable(&self) {
        todo!()
    }

    fn parameters(&self) {
        todo!()
    }

    fn hyper_parameters(&self) {
        todo!()
    }

    fn derived_variables(&self) {
        todo!()
    }

    fn gradiants(&self) {
        todo!()
    }

    fn act_fn(&self) {
        todo!()
    }

    fn X(&self) {
        todo!()
    }

    fn freeze(&self) {
        todo!()
    }

    fn unfreeze(&self) {
        todo!()
    }

    fn flush_gradients(&self) {
        todo!()
    }

    fn update(&self) {
        todo!()
    }

    fn set_params(&self, summary_dict: SummaryDict) {
        todo!()
    }

    fn summary(&self) -> WrapperSummary {
        todo!()
    }
}

impl WrapperBase for Dropout {
    fn _wrapper_params(&self) {
        todo!()
    }

    fn _set_wrapper_params(&self) {
        todo!()
    }
}

fn init_wrappers(layer: &impl Layer, wrapper_list: &Vec<impl Wrapper>) {
    wrapper_list.iter().for_each(|wr| {
        
    })
}

pub enum Wrappers {
    Dropout(Dropout),
}

impl Wrapper for Wrappers {
    fn forward(&self) {
        todo!()
    }

    fn backward(&self) {
        todo!()
    }

    fn trainable(&self) {
        todo!()
    }

    fn parameters(&self) {
        todo!()
    }

    fn hyper_parameters(&self) {
        todo!()
    }

    fn derived_variables(&self) {
        todo!()
    }

    fn gradiants(&self) {
        todo!()
    }

    fn act_fn(&self) {
        todo!()
    }

    fn X(&self) {
        todo!()
    }

    fn freeze(&self) {
        todo!()
    }

    fn unfreeze(&self) {
        todo!()
    }

    fn flush_gradients(&self) {
        todo!()
    }

    fn update(&self) {
        todo!()
    }

    fn set_params(&self, summary_dict: SummaryDict) {
        todo!()
    }

    fn summary(&self) -> WrapperSummary {
        todo!()
    }
}
