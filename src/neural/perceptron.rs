use std::{
    vec::Vec,
};

use crate::{
    utils::{
        EncephalonError,
        Tensor,
    },
    funcs::{
        cost::{self, CostFns},
        gd::GradientDescent,
    },
};

use rand::prelude::*;

type EError = EncephalonError;
type TensorType = Vec<Vec<f32>>;

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

        let mut weights: TensorType = Vec::new();

        let mut rng = rand::thread_rng();
        for _ in 0..out {
            let mut w: Vec<f32> = Vec::new();

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

        let res = data.mult(&self.weights.transpose())?;

        Ok(res)
    }

    pub fn fit(&mut self, input: &Tensor, output: &Tensor, lr: f32) -> Result<(), EError> {
        let x = input.insert_cols_front(vec![1.]);

        let predicted = self.predict(&input)?;

        let w_gradients = match self.cost {
            _ => cost::d_mse_gradient(&x, predicted, output)?
        };

        self.weights = self.weights.sub(&w_gradients.mult_scalar(lr))?;

        Ok(())
    }

    pub fn train(&self, x: &Tensor, y: &Tensor, lr: f32, gd: GradientDescent) -> Result<(), EError> {
        unimplemented!();
    }
}