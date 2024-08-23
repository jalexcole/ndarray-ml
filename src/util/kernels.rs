//! In machine learning, kernel machines are a class of algorithms for pattern 
/// analysis, whose best known member is the support-vector machine (SVM). 
/// These methods involve using linear classifiers to solve nonlinear problems.
/// [1] The general task of pattern analysis is to find and study general types 
/// of relations (for example clusters, rankings, principal components, 
/// correlations, classifications) in datasets. For many algorithms that solve 
/// these tasks, the data in raw representation have to be explicitly 
/// transformed into feature vector representations via a user-specified 
/// feature map: in contrast, kernel methods require only a user-specified 
/// kernel, i.e., a similarity function over all pairs of data points computed 
/// using inner products. The feature map in kernel machines is infinite 
/// dimensional but only requires a finite dimensional matrix from user-input 
/// according to the Representer theorem. Kernel machines are slow to compute 
/// for datasets larger than a couple of thousand examples without parallel 
/// processing.
/// Kernel methods owe their name to the use of kernel functions, which enable 
/// them to operate in a high-dimensional, implicit feature space without ever 
/// computing the coordinates of the data in that space, but rather by simply 
/// computing the inner products between the images of all pairs of data in the 
/// feature space. This operation is often computationally cheaper than the 
/// explicit computation of the coordinates. This approach is called the 
/// "kernel trick".[2] Kernel functions have been introduced for sequence data, 
/// graphs, text, images, as well as vectors.
/// Algorithms capable of operating with kernels include the kernel perceptron, 
/// support-vector machines (SVM), Gaussian processes, principal components 
/// analysis (PCA), canonical correlation analysis, ridge regression, spectral 
/// clustering, linear adaptive filters and many others.
/// Most kernel algorithms are based on convex optimization or eigenproblems 
/// and are statistically well-founded. Typically, their statistical properties 
/// are analyzed using statistical learning theory (for example, using 
/// Rademacher complexity).
/// *This content is adapted from 
/// [Wikipedia](https://en.wikipedia.org/wiki/Kernel_method) 
/// under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).*
use std::fmt::{Debug, Display};

use ndarray::{Array2, ArrayD};

#[derive(Debug)]
pub struct KernelParameters {
    id: String,
    d: isize,
    gamma: Option<f64>,
    c0: f64,
}

pub struct KernalHyperParameters {}

#[derive(Debug)]
pub struct KernelInitializer {
    param: KernelParameters,
}

pub trait Kernel: Display + Debug {
    fn kernel(&self, x: &ArrayD<f64>, y: &ArrayD<f64>) -> f64;
}

pub struct LinearKernel {}

impl LinearKernel {
    fn _kernel(&self, x: Array2<f32>, y: Option<Array2<f32>>) -> Array2<f32> {
        let y_ = match y {
            Some(y) => y,
            None => x.clone(),
        };
        let y_ = y_.t().to_owned();
        x * (y_)
    }
}

pub struct PolynomaicKernel {}
/// Radial basis function (RBF) / squared exponential kernel.
pub struct RBFKernel {}

pub enum Kernals {
    Linear(LinearKernel),
    Polynomaic(PolynomaicKernel),
    RBF(RBFKernel),
}
