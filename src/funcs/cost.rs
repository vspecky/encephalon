use crate::utils::{Tensor, EncephalonError};

pub enum CostFns {
    MeanAbsolute,
    MeanSquared,
    RootMeanSquared,
}

type EError = EncephalonError;

// Mean Squared Error Function
pub fn mse(predicted: &Tensor, actual: &Tensor) -> Result<f32, EError> {
    if predicted.cols() != actual.cols() || predicted.rows() != actual.rows() {
        return Err(EError::new("Predicted and Label Tensor dimensions do not match"));
    }

    let cols = predicted.cols() as f32;

    let mut err = 0_f32;

    for (pred, act) in predicted.mat.iter().zip(&actual.mat) {
        let mut instance_err = 0_f32;

        for (p, y) in pred.iter().zip(act) {
            instance_err += (p - y).powi(2);
        }

        err += instance_err / cols;
    }

    Ok(err / predicted.rows() as f32)
}

// Mean Absolute Error Function
pub fn m_abs(predicted: &Tensor, actual: &Tensor) -> Result<f32, EError> {
    if predicted.cols() != actual.cols() || predicted.rows() != actual.rows() {
        return Err(EError::new("Predicted and Label Tensor dimensions do not match"));
    }

    let cols = predicted.cols() as f32;

    let mut err = 0_f32;

    for (pred, act) in predicted.mat.iter().zip(&actual.mat) {
        let mut instance_err = 0_f32;

        for (p, y) in pred.iter().zip(act) {
            instance_err += (p - y).abs();
        }

        err += instance_err / cols;
    }

    Ok(err / predicted.rows() as f32)
}

// Root Mean Squared Error Function
pub fn rmse(predicted: &Tensor, actual: &Tensor) -> Result<f32, EError> {
    if predicted.cols() != actual.cols() || predicted.rows() != actual.rows() {
        return Err(EError::new("Predicted and Label Tensor dimensions do not match"));
    }

    let cols = predicted.cols() as f32;

    let mut err = 0_f32;

    for (pred, act) in predicted.mat.iter().zip(&actual.mat) {
        let mut instance_err = 0_f32;

        for (p, y) in pred.iter().zip(act) {
            instance_err += (p - y).powi(2);
        }

        err += instance_err / cols;
    }

    Ok((err / predicted.rows() as f32).powf(0.5))
}

// Derivative of Mean Squared Error Function (Might need refactoring when implementing backpropagation)
pub fn d_mse_gradient(input: &Tensor, predicted: Tensor, actual: &Tensor) -> Result<Tensor, EError> {
    if predicted.cols() != actual.cols() || predicted.rows() != actual.rows() {
        return Err(EError::new("Predicted and Label Tensor dimensions do not match"));
    }

    let m = input.rows() as f32;

    Ok(input.transpose().mult(&predicted.sub(&actual)?)?.mult_scalar(2. / m).transpose())
}