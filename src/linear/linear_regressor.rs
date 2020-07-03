use std::{
    vec::Vec,
}

use crate::{
    utils::{Tensor, EncephalonError, TrainParams,},
    funcs::{
        cost,
        gd::GradientDescent
    }
}

use rand::prelude::*;

type TensorType = Vec<Vec<f32>>;
type EError = EncephalonError;

pub struct LinearRegressor {
    input_size: usize,
    weights: Tensor,
}

impl LinearRegressor {
    pub fn new(input_size: usize) -> Result<Self, EError> {
        if input_size == 0 {
            return Err(EError::new("Linear Regression needs at least one input"));
        }

        let weight_mat = Vec::<Vec<f32>>::with_capacity(1);

        let row = Vec::<f32>::with_capacity(input_size + 1);
        let rng = rand::thread_rng();
        for _ in 0..input_size {
            row.push(rng.gen<f32>()); 
        }

        weight_mat.push(row);

        let weights = Tensor::new(weight_mat)?;

        Ok(Self { input_size, weights })
    }

    pub fn predict(&self, data: &Tensor) -> Result<Tensor, EError> {
        if data.cols() != self.input_size {
            return Err(EError::new("Data size does not match input size"));
        }

        let prepared_data = data.insert_cols_front(vec![1.]);

        let res = prepared_data.mult(&self.weights.transpose());

        Ok(res)
    }

    fn fit(&mut self, inp: &Tensor, out: &Tensor, params: &TrainParams) -> Result<(), EError> { 
        let x = inp.insert_cols_front(vec![1.]);

        let predicted = self.predict(inp)?;

        let w_gradients = cost::d_mse_gradient(&x, predicted, outputs)?;

        self.weights = self.weights.sub(&w_gradients.mult_scalar(params.lr))?;

        Ok(())
    }

    pub fn train(&mut self, params: TrainParams) -> Result<(), EError> {
        for _ in 0..params.epochs {
            match params.gd {
                GradientDescent::Batch => {
                    self.fit(&params.inputs, &params.outputs, &params)?;
                }
                
                GradientDescent::Stochastic => {
                    for (inp, out) in params.inputs.mat.iter().zip(&params.outputs.mat) {
                        let inp_tensor = Tensor::new(vec![inp.clone()])?;
                        let out_tensor = Tensor::new(vec![out.clone()])?;

                        self.fit(&inp_tensor, &out_tensor, &params)?;
                    }
                }

                GradientDescent::MiniBatch(batch_size) => {
                    if params.inputs.mat.len() / batch_size as usize == 0 {
                        return Err(EError::new("Cannot split inputs into that many batches"));
                    }
                    
                    let batches = (params.inputs.mat.len() as f32 / batch_size as f32).ceil();
                    
                    for batch in 0..batches as u32 {
                        let mut inp = Vec::<Vec<f32>>::with_capacity(batch_size as usize);
                        let mut out = Vec::<Vec<f32>>::with_capacity(batch_size as usize);

                        for (i, o) in params.inputs.mat.iter().skip(batch as usize * batch_size as usize)
                            .zip(params.outputs.mat.iter().skip(batch as usize * batch_size as usize)) {
                            inp.push(i.clone());
                            out.push(o.clone());
                        }

                        let inp_tens = Tensor::new(inp)?;
                        let out_tens = Tensor::new(out)?;

                        self.fit(&inp_tens, &out_tens, &params)?;
                    }
                }
            }
        }

        Ok(())
    }
}

