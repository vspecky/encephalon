use std::{
    vec::Vec,
};

use crate::{
    utils::{
        EncephalonError,
        Tensor,
        TrainParams,
    },
    funcs::{
        cost::{self, CostFns},
        gd::GradientDescent,
    },
};

use rand::prelude::*;

type EError = EncephalonError;

pub struct Perceptron {
    inputs: u32,
    outputs: u32,
    weights: Tensor,
    cost: CostFns
}

impl Perceptron {
    pub fn new(inp: u32, out: u32, cost_fn: CostFns) -> Result<Self, EError> {
        if inp == 0 || out == 0 {
            return Err(EError::new("A perceptron cannot have zero amount of inputs or outputs"));
        }

        let mut weights = Vec::<Vec<f32>>::with_capacity((out + 1) as usize);

        let mut rng = rand::thread_rng();
        for _ in 0..out {
            let mut w  = Vec::<f32>::with_capacity((inp + 1) as usize);

            for _ in 0..=inp {
                w.push(rng.gen::<f32>());
            }

            weights.push(w);
        }

        let weight_tensor = Tensor::new(weights)?;

        Ok(Self {
            inputs: inp,
            outputs: out,
            weights: weight_tensor,
            cost: cost_fn,
        })
    }

    pub fn predict(&self, data: &Tensor) -> Result<Tensor, EError> {
        if data.cols() != self.inputs as usize {
            return Err(EError::new("Size of data provided does not match input size"));
        }

        let prepared_data = data.insert_cols_front(vec![1.]);

        let res = prepared_data.mult(&self.weights.transpose())?;

        Ok(res)
    }

    fn fit(&mut self, inputs: &Tensor, outputs: &Tensor, params: &TrainParams) -> Result<(), EError> {
        let x = inputs.insert_cols_front(vec![1.]);

        let predicted = self.predict(inputs)?;

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
