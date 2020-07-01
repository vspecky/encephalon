pub enum GradientDescent {
    Batch,
    Stochastic,
    MiniBatch(u32)
}

pub use GradientDescent::*;