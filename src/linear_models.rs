use ndarray::{Array1, Array2};




pub struct BayesianLinearRegressionUnkownVariance;

pub struct BayesianLinearRegressionKnownVariance;

pub struct GeneralizedLinearModel;

///
/// ```math
/// y_i = \beta^\top \mathbf{x}_i + \epsilon_i
/// ```
pub struct LinearRegression {
    fit_intercept: bool,
    beta: Option<Array2<f32>>,
    sigma_inv: Option<Array2<f32>>
}

impl LinearRegression {

    pub fn new(fit_intercept: bool) -> Self {
        Self {fit_intercept, beta: None, sigma_inv: None}
    }

    pub fn update(&self, X: Array2<f32>, y: &Array2<f32>, weights: &Option<Array1<f32>>) {
        
    }

    fn update1D(&self) {

    }

    fn updae2d(&self) {

    }

    pub fn fit(&self) {

    }

    pub fn predict(&self, X: &Array2<f32>) {
        let x = X.clone();
        if self.fit_intercept {
            
        }
    }

    
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new(true)
    }
}

pub struct LogisticRegression;

pub struct GaussianNBClassifier;

pub struct RidgeRegression;