use ndarray::{arr1, s, Array, Array1, Array2, Array3, Array4, ArrayD, Axis, IxDyn, Zip};
use ndarray_rand::{rand::seq::SliceRandom, rand_distr::Normal, RandomExt};
use rand::thread_rng;

use crate::util::Either;
/// Compute the minibatch indices for a training dataset.
///
/// # Parameters
/// - `X`: The dataset to divide into minibatches. Assumes the first dimension
///   represents the number of training examples.
/// - `batchsize`: The desired size of each minibatch. Note, however, that if `X.len() % batchsize > 0`
///   then the final batch will contain fewer than `batchsize` entries. Default is 256.
/// - `shuffle`: Whether to shuffle the entries in the dataset before dividing into minibatches.
///   Default is `true`.
///
/// # Returns
/// A tuple containing:
/// - An `Array2<usize>` where each row contains the indices into the dataset for each batch.
/// - The number of batches.
/// # Authors
/// - [ChatGPT] for conversion from Python to Rust
pub fn minibatch(X: &ArrayD<f64>, batchsize: usize, shuffle: bool) -> (Array2<usize>, usize) {
    let N = X.shape()[0]; // First dimension represents the number of training examples
    let mut ix: Vec<usize> = (0..N).collect();
    let n_batches = (N as f64 / batchsize as f64).ceil() as usize;

    if shuffle {
        let mut rng = thread_rng();
        ix.as_mut_slice().shuffle(&mut rng);
    }

    // Create an Array2 to hold the batch indices
    let mut batches = Array2::<usize>::zeros((n_batches, batchsize));

    for (i, mut row) in batches.axis_iter_mut(Axis(0)).enumerate() {
        let start = i * batchsize;
        let end = usize::min(start + batchsize, N);
        let batch_indices = &ix[start..end];
        for (j, &index) in batch_indices.iter().enumerate() {
            row[j] = index;
        }
    }

    (batches, n_batches)
}

#[test]
fn test_minibatch() {
    // Create a dummy dataset with 1024 examples, each with 10 features
    let X = Array2::from_shape_vec((1024, 10), (0..10240).map(|x| x as f64).collect())
        .unwrap()
        .into_dyn();
    let batchsize = 256; // Batch size
    let shuffle = false; // Set shuffle to false to simplify validation

    let (batches, n_batches) = minibatch(&X, batchsize, shuffle);

    // Check the number of batches
    assert_eq!(n_batches, 4);

    // Check the shape of the batches array
    assert_eq!(batches.shape(), &[4, 256]);

    // Validate that the indices are in the expected range
    for i in 0..n_batches {
        for j in 0..batchsize {
            let idx = batches[(i, j)];
            if i == n_batches - 1 && j >= X.shape()[0] % batchsize {
                // The last batch might have fewer elements; zeros are expected in unused slots
                assert_eq!(idx, 0, "Expected 0, but found {}", idx);
            } else {
                // Validate that the index is within the valid range of 0..1024
                assert!(idx < 1024, "Index out of range: idx = {}", idx);
            }
        }
    }

    // Additional validation to ensure that when shuffle is false, the indices are in order
    let expected_indices: Vec<usize> = (0..1024).collect();
    let flat_indices: Vec<usize> = batches.iter().cloned().filter(|&x| x != 0).collect();
    assert_eq!(flat_indices, expected_indices);
}
pub fn calc_pad_dims_2D() {
    todo!()
}

/// Computes the padding necessary to ensure that convolving `X` with a 2D kernel
/// of shape `kernel_shape` and stride `stride` produces outputs with dimensions `out_dim`.
///
/// # Parameters
/// - `x_shape`: A tuple of `(n_ex, in_rows, in_cols, in_ch)` representing the dimensions of the input volume.
///              Padding is applied to `in_rows` and `in_cols`.
/// - `out_dim`: A tuple of `(out_rows, out_cols)` representing the desired dimensions of an output example after applying the convolution.
/// - `kernel_shape`: A tuple representing the dimensions of the 2D convolution kernel.
/// - `stride`: An integer representing the stride for the convolution kernel.
/// - `dilation`: An integer representing the number of pixels inserted between kernel elements. Default is 0.
///
/// # Returns
/// - A tuple of `(left, right, up, down)` representing the padding dimensions for `X`.
///
/// # Errors
/// - Returns an error if any of the input parameters have invalid types or values.
fn calc_pad_dims_2d(
    x_shape: (usize, usize, usize, usize),
    out_dim: (usize, usize),
    kernel_shape: (usize, usize),
    stride: usize,
    dilation: usize,
) -> Result<(usize, usize, usize, usize), String> {
    let (n_ex, in_rows, in_cols, in_ch) = x_shape;
    let (out_rows, out_cols) = out_dim;
    let (fr, fc) = kernel_shape;

    let d = dilation;
    let (_fr, _fc) = (fr * (d + 1) - d, fc * (d + 1) - d);

    let pr = ((stride as isize * (out_rows as isize - 1) + _fr as isize - in_rows as isize) / 2)
        as usize;
    let pc = ((stride as isize * (out_cols as isize - 1) + _fc as isize - in_cols as isize) / 2)
        as usize;

    let out_rows1 =
        (1 + (in_rows as isize + 2 * pr as isize - _fr as isize) / stride as isize) as usize;
    let out_cols1 =
        (1 + (in_cols as isize + 2 * pc as isize - _fc as isize) / stride as isize) as usize;

    let (mut pr1, mut pr2) = (pr, pr);
    if out_rows1 == out_rows - 1 {
        pr2 += 1;
    } else if out_rows1 != out_rows {
        return Err("Output rows do not match expected output dimension.".to_string());
    }

    let (mut pc1, mut pc2) = (pc, pc);
    if out_cols1 == out_cols - 1 {
        pc2 += 1;
    } else if out_cols1 != out_cols {
        return Err("Output columns do not match expected output dimension.".to_string());
    }

    // if [pr1, pr2, pc1, pc2].iter().any(|&x| x < 0) {
    //     return Err(format!("Padding cannot be less than 0. Got: {:?}", (pr1, pr2, pc1, pc2)));
    // }

    Ok((pr1, pr2, pc1, pc2))
}

pub fn calc_pad_dims_1D() {
    todo!()
}

/// Computes the padding necessary to ensure that convolving `X` with a 1D kernel
/// of shape `kernel_width` and stride `stride` produces outputs with length `l_out`.
///
/// # Parameters
/// - `X_shape`: A tuple of `(n_ex, l_in, in_ch)` representing the dimensions of the input volume. Padding is applied on either side of `l_in`.
/// - `l_out`: The desired length of an output example after applying the convolution.
/// - `kernel_width`: The width of the 1D convolution kernel.
/// - `stride`: The stride for the convolution kernel.
/// - `dilation`: The number of pixels inserted between kernel elements. Default is `0`.
/// - `causal`: Whether to compute the padding dims for a regular or causal convolution. If `causal`, padding is added only to the left side of the sequence. Default is `false`.
///
/// # Returns
/// A tuple of `(left, right)` representing the padding dimensions for `X`.
///
/// # Panics
/// Panics if `X_shape` is not a tuple, or if `l_out`, `kernel_width`, or `stride` are not integers.
///
pub fn calc_pad_dims_1d(
    X_shape: (usize, usize, usize),
    l_out: usize,
    kernel_width: usize,
    stride: usize,
    dilation: usize,
    causal: bool,
) -> (usize, usize) {
    let (n_ex, l_in, in_ch) = X_shape;
    let d = dilation;
    let fw = kernel_width;

    // Update effective filter shape based on dilation factor
    let _fw = fw * (d + 1) - d;
    let total_pad = (stride * (l_out - 1) + _fw - l_in) as isize;

    if causal {
        let pw1 = total_pad as usize;
        let l_out1 = 1 + (l_in as isize + total_pad - _fw as isize) / stride as isize;
        assert_eq!(l_out1, l_out as isize);
        return (pw1, 0);
    } else {
        let pw = (total_pad / 2) as usize;
        let l_out1 = 1 + ((l_in + 2 * pw - _fw) / stride);

        // Add asymmetric padding pixels to right / bottom
        let (pw1, pw2) = if l_out1 == l_out - 1 {
            (pw, pw + 1)
        } else if l_out1 != l_out {
            panic!("Padding mismatch");
        } else {
            (pw, pw)
        };

        // if pw1 < 0 || pw2 < 0 {
        //     panic!("Padding cannot be less than 0. Got: ({}, {})", pw1, pw2);
        // }
        (pw1, pw2)
    }
}

// pub fn pad1D() -> (ArrayD<f32>, (usize, usize)) {
//     todo!()
// }

/// Zero-pads a 3D input volume `X` along the second dimension.
///
/// # Parameters
/// - `X`: Input volume of shape `(n_ex, l_in, in_ch)`. Padding is applied to `l_in`.
/// - `pad`: The padding amount. If `same`, add padding to ensure that the output length of a 1D convolution with a kernel of `kernel_shape` and stride `stride` is the same as the input length. If `causal`, compute padding such that the output both has the same length as the input AND `output[t]` does not depend on `input[t + 1:]`. If 2-tuple, specifies the number of padding columns to add on each side of the sequence.
/// - `kernel_width`: The dimension of the 2D convolution kernel. Only relevant if `pad` is `same` or `causal`. Default is `None`.
/// - `stride`: The stride for the convolution kernel. Only relevant if `pad` is `same` or `causal`. Default is `None`.
/// - `dilation`: The dilation of the convolution kernel. Only relevant if `pad` is `same` or `causal`. Default is `0`.
///
/// # Returns
/// - `X_pad`: The padded output volume of shape `(n_ex, padded_seq, in_channels)`.
/// - `p`: A tuple representing the number of 0-padded columns added to the (left, right) of the sequences in `X`.
///
/// # Example
/// ```rust
/// let X = Array::from_shape_vec((1, 5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
/// let (X_pad, p) = pad_1d(X, (1, 1), None, None, 0);
/// ```
pub fn pad_1d(
    X: Array3<f64>,
    pad: (usize, usize),
    kernel_width: Option<usize>,
    stride: Option<usize>,
    dilation: usize,
) -> (Array3<f64>, (usize, usize)) {
    let mut p = pad;
    let (n_ex, l_in, in_ch) = X.dim();

    if p.0 == p.1 {
        let mut X_pad = Array3::<f64>::zeros((n_ex, l_in + p.0 + p.1, in_ch));
        Zip::from(X_pad.slice_mut(s![.., p.0..l_in + p.0, ..]))
            .and(X.view())
            .for_each(|out, &input| *out = input);
        (X_pad, p)
    } else if kernel_width.is_some() && stride.is_some() {
        let causal = match pad {
            (0, _) => true,
            _ => false,
        };
        p = calc_pad_dims_1d(
            X.dim(),
            l_in,
            kernel_width.unwrap(),
            stride.unwrap(),
            dilation,
            causal,
        );
        pad_1d(X, p, None, None, 0)
    } else {
        (X, p)
    }
}

pub enum PadType {
    Same,
    Causal,
    Fixed((usize, usize)),
}

// Example of the helper function `calc_pad_dims_1d`, which you'll need to define:
// fn calc_pad_dims_1d(
//     shape: &[usize],
//     l_in: usize,
//     kernel_width: usize,
//     stride: usize,
//     causal: bool,
//     dilation: usize,
// ) -> (usize, usize) {
//     let effective_kernel_width = (kernel_width - 1) * dilation + 1;
//     let output_length = (l_in + stride - 1) / stride;

//     if causal {
//         (effective_kernel_width - 1, 0)
//     } else {
//         let total_pad = ((output_length - 1) * stride + effective_kernel_width).saturating_sub(l_in);
//         let left_pad = total_pad / 2;
//         let right_pad = total_pad - left_pad;
//         (left_pad, right_pad)
//     }
// }

pub fn pad2d(X: Array4<f32>, pad: &[usize], kernel_shape: (usize, usize), stride: isize, dilation: isize) -> (Array4<f64>, [usize; 4]) {
    todo!()
}

// pub fn dilate() {}

/// Dilate a 4D volume `X` by `d`.
///
/// # Notes
/// For a visual depiction of a dilated convolution, see [Dumoulin & Visin (2016)](https://arxiv.org/pdf/1603.07285v1.pdf).
///
/// # Parameters
/// - `X`: A 4D array with shape `(n_ex, in_rows, in_cols, in_ch)`. This represents the input volume.
/// - `d`: The number of 0-rows to insert between each adjacent row + column in `X`.
///
/// # Returns
/// A 4D array with shape `(n_ex, out_rows, out_cols, in_ch)` where:
/// - `out_rows = in_rows + d * (in_rows - 1)`
/// - `out_cols = in_cols + d * (in_cols - 1)`
#[inline]
fn dilate(X: &Array4<f64>, d: usize) -> Array4<f64> {
    let (n_ex, in_rows, in_cols, n_in) = X.dim();

    // Calculate the dimensions of the output array
    let out_rows = in_rows + d * (in_rows - 1);
    let out_cols = in_cols + d * (in_cols - 1);

    // Create the output array filled with zeros
    let mut Xd = Array4::<f64>::zeros((n_ex, out_rows, out_cols, n_in));

    // Fill the output array with the values from X, leaving d rows and columns of zeros between them
    for n in 0..n_ex {
        for r in 0..in_rows {
            for c in 0..in_cols {
                for ch in 0..n_in {
                    Xd[(n, r * (d + 1), c * (d + 1), ch)] = X[(n, r, c, ch)];
                }
            }
        }
    }

    Xd
}
/// Compute the fan-in and fan-out for a weight matrix/volume.
///
/// # Parameters
/// - `weight_shape`: A slice representing the dimensions of the weight matrix/volume.
///   The final 2 entries must be `in_ch`, `out_ch`.
///
/// # Returns
/// A tuple `(fan_in, fan_out)` where:
/// - `fan_in`: The number of input units in the weight tensor.
/// - `fan_out`: The number of output units in the weight tensor.
///
/// # Errors
/// Returns a `Result` with an error message if the `weight_shape` does not have the expected dimensions.
pub fn calc_fan(weight_shape: &[usize]) -> Result<(usize, usize), String> {
    match weight_shape.len() {
        2 => Ok((weight_shape[0], weight_shape[1])),
        3 | 4 => {
            let (in_ch, out_ch) = (
                weight_shape[weight_shape.len() - 2],
                weight_shape[weight_shape.len() - 1],
            );
            let kernel_size: usize = weight_shape[..weight_shape.len() - 2].iter().product();
            Ok((in_ch * kernel_size, out_ch * kernel_size))
        }
        _ => Err(format!("Unrecognized weight dimension: {:?}", weight_shape)),
    }
}

pub fn calc_conv_out_dims(
    X_shape: &[usize],
    W_shape: &[usize],
    stride: isize,
    dilation: isize,
) -> Either<[usize; 3], [usize; 4]> {
    todo!()
}
#[inline]
fn im2col_indices() {}

pub fn im2col() {}

pub fn col2im() {}

pub fn conv2d(X: &Array4<f64>, W: &Array4<f64>, b: &[usize], dialtion: Option<usize>) -> Array4<f64> {
    todo!();
    
}

pub fn conv1D() {}

pub fn deconv2d_naive() {}

pub fn conv2D_naive() {}

pub fn he_uniform() {}

pub fn he_normal() {}

pub fn glorot_uniform() {}

pub fn glorot_normal() {}

/// Generate draws from a truncated normal distribution via rejection sampling.
///
/// # Parameters
/// - `mean`: The mean/center of the distribution. Can be a scalar or an array.
/// - `std`: Standard deviation (spread or "width") of the distribution. Can be a scalar or an array.
/// - `out_shape`: Output shape as an array of `usize`.
///
/// # Returns
/// An `ArrayD<f64>` containing samples from the truncated normal distribution parameterized by `mean` and `std`.
pub fn truncated_normal(out_shape: &[usize], mean: f64, std: f64, attempts: usize) -> ArrayD<f64> {
    let out_shape_dyn = IxDyn(out_shape);

    // Generate random samples from a normal distribution
    let mut samples = Array::random(out_shape_dyn.clone(), Normal::new(mean, std).unwrap());

    for _ in 0..attempts {
        // Apply the truncation condition
        let reject = samples.mapv(|x| (x >= mean + 2.0 * std) || (x <= mean - 2.0 * std));

        if reject.mapv(|x| x as usize).sum() == 0 {
            // Your code here
            break;
        }

        // Resample the rejected values
        let resamples = Array::random(reject.shape(), Normal::new(mean, std).unwrap());

        samples.zip_mut_with(&reject, |s, &r| {
            if r {
                *s = resamples[[0]];
            }
        });
    }

    samples
}

#[test]
fn test_truncated_normal() {
    let mean = arr1(&[0.0, 1.0, 2.0]).into_dyn();
    let std = 1.0;
    let out_shape = [3, 1000]; // 3 distributions, each with 1000 samples

    let samples = truncated_normal(&out_shape, mean.mean().unwrap(), std, 1);

    // Check that no value is more than 2 standard deviations away from the mean
    for (i, sample) in samples.axis_iter(ndarray::Axis(1)).enumerate() {
        let m = mean[i];
        let s = std;
        assert!(sample.iter().all(|&x| x >= m - 2.0 * s && x <= m + 2.0 * s));
    }
}
