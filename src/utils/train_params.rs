use super::Tensor;
use crate::funcs::{
    cost::CostFns,
    gd::GradientDescent,
};

pub struct TrainParams<'a> {
    pub inputs: &'a Tensor,
    pub outputs: &'a Tensor,
    pub lr: f32,
    pub cost: CostFns,
    pub gd: GradientDescent,
    pub epochs: u32
}

impl<'a> TrainParams<'a> {
    pub fn new(inputs: &'a Tensor, outputs: &'a Tensor, epochs: u32,
               lr: f32, cost: CostFns, gd: GradientDescent) -> Self {

        Self {
            inputs, outputs, lr,
            cost, gd, epochs
        }
    }
}
