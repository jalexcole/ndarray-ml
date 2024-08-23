use ndarray::{Array2, Array3};

use crate::util::Either;

pub struct GPRegression {}


impl GPRegression {
    fn fit(&self) {}

    fn predict(&self, X: &Array2<f32>, conf_interval: f32, return_cov: bool) -> Either<(Array2<f32>, Array2<f32>), (Array2<f32>, Array2<f32>, Array2<f32>)>{
        todo!();

        let X_star = X.clone();
        

        
    }

    fn marginal_log_likelihood(&self) -> f64{


        return 0.0
    }

    fn sample(&self) -> Array3<f32>{
        todo!()
    }
}
pub struct KernelRegression {}

impl KernelRegression {

}

pub struct KNN {}


impl KNN {

}