/// Hidden Markov model module
use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use std::f64::EPSILON;

pub struct MultinomialHMM {
    /// The transition matrix between latent states in the HMM. Index `i`,
    /// `j` gives the probability of transitioning from latent state `i` to
    /// latent state `j`.
    A: Array2<f64>,
    /// The emission matrix. Entry `i`, `j` gives the probability of latent
    /// state `i` emitting an observation of type `j`
    B: Array2<f64>,
    /// The number of unique latent states;
    N: usize,
    /// The number of unique observation types
    V: usize,
    /// The collection of observed training sequences.
    O: Array2<usize>,
    /// The number of sequence in `O`.
    I: usize,
    /// The number of observations in each sequence in `O`.
    T: usize,
}

impl MultinomialHMM {
    fn new(A: Array2<f32>, B: Array2<f32>, pi: Array1<f32>, eps: Option<f32>) -> Self {
        todo!()
    }
    pub fn generate(
        &self,
        n_steps: usize,
        latent_state_types: Array1<f32>,
        obs_types: Array1<f32>,
    ) -> (Array1<f32>, Array1<f32>) {
        todo!()
    }
    /// Given the HMM parameterized by :math:`(A`, B, \pi)` and an observation
    ///    sequence `O`, compute the marginal likelihood of `O`,
    ///    :math:`P(O \mid A,B,\pi)`, by marginalizing over latent states.
    pub fn log_likelihood(&self, O: &Array2<f32>) -> f32 {
        todo!();
        if O.ndim() == 1 {
            // FIXME: O =  O.into_shape_with_order(shape![1, -1]).unwrap();
        }

        let shape = O.shape();

        let i = shape[0];
        let t = shape[1];

        // let forward = self._forward(O.index_axis(axis!(1), 0));
        // let log_likelihood = logsumexp(forward.slice(s![..., T - 1]);
        // return log_likelihood;
    }

    fn decode(&self, O: &Array1<f32>) -> (Vec<f32>, f32) {
        todo!();
    }

    fn _forward(&self, Obs: Array1<f32>) -> Array2<f32> {
        todo!()
    }

    fn _backwards(&self, Obs: Array1<f32>) -> Array2<f32> {
        todo!();
    }

    fn _initialize_paramters(&self) {}
    /// Given an observation sequence `O` and the set of possible latent states,
    /// learn the MLE HMM parameters `A` and `B`.
    pub fn fit(
        &self,
        O: Array2<f32>,
        latent_state_type: usize,
        observation_types: Vec<f32>,
        pi: Array1<f32>,
        tol: f32,
        verbose: bool,
    ) -> (Array2<f32>, Array2<f32>, Array1<f32>) {
        todo!()
    }
    /// Run a single E-step update for the Baum-Welch/Forward-Backward
    /// algorithm. This step estimates ``xi`` and ``gamma``, the excepted
    /// state-state transition counts and the expected state-occupancy counts,
    /// respectively.
    fn _e_step(&self, O: Array2<f32>) -> (Array3<f32>, Array4<f32>, Array2<f32>) {
        todo!()
    }

    /// Run a single M-step update for the Baum-Welch/Forward-Backward
    fn _m_step(
        &self,
        O: Array2<f32>,
        gamma: Array3<f32>,
        xi: Array4<f32>,
        phi: Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>, Array1<f32>) {
        todo!()
    }
}

mod gpt {
    use ndarray::{Array1, Array2};

    /// A simple hidden Markov model with multinomial emission distribution.
    pub struct MultinomialHMM {
        pub parameters: Parameters,
        pub hyperparameters: Hyperparameters,
        pub derived_variables: DerivedVariables,
    }

    pub struct Parameters {
        pub a: Option<Array2<f64>>,  // transition matrix
        pub b: Option<Array2<f64>>,  // emission matrix
        pub pi: Option<Array1<f64>>, // prior probability of each latent state
    }

    pub struct Hyperparameters {
        pub eps: f64, // epsilon
    }

    pub struct DerivedVariables {
        pub n: Option<usize>, // number of latent state types
        pub v: Option<usize>, // number of observation types
    }
}
