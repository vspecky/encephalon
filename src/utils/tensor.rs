use std::{
    vec::Vec,
    fmt,
};
use super::error::EncephalonError;

type EError = EncephalonError;
type TensorType = Vec<Vec<f32>>;

#[derive(Debug)]
pub struct Tensor {
    pub mat: TensorType,
    m: usize,
    n: usize,
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.mat == other.mat
    }
}

impl PartialEq<TensorType> for Tensor {
    fn eq(&self, other: &TensorType) -> bool {
        self.mat == *other
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut a = String::from(&format!("Tensor({}x{}) [ ", self.m, self.n));

        for row in self.mat.iter() {
            let mut row_str = String::from("[ ");

            for elem in row.iter() {
                row_str.push_str(&format!("{} ", elem));
            }

            row_str.push_str("] ");
            a.push_str(&row_str);
        }

        a.push_str("]");

        write!(f, "{}", a)
    }
}

impl Tensor {
    pub fn new(mat: TensorType) -> Result<Self, EError> {
        if mat.len() == 0 {
            return Err(EError::new("Tensor should have at least one row"));
        }

        let n = mat[0].len();

        if n == 0 {
            return Err(EError::new("Tensor should have at least one element in each row"));
        }

        if !mat.iter().all(|row| row.len() == n) {
            return Err(EError::new("Tensor rows don't all have the same length"));
        }

        let m = mat.len();

        Ok(Self {
            mat: mat,
            m: m,
            n: n,
        })
    }

    pub fn zeros(m: usize, n: usize) -> Result<Self, EError> {
        if m == 0 || n == 0 {
            return Err(EError::new("Cannot create tensor with zero dimension(s)"));
        }

        let mat: TensorType = vec![vec![0.0; n]; m];

        Ok(Self { mat, m, n })
    }

    pub fn rows(&self) -> usize {
        self.n
    }

    pub fn cols(&self) -> usize {
        self.m
    }

    pub fn transpose(&self) -> Self {
        let mut t_tensor = Vec::<Vec<f32>>::with_capacity(self.n);

        for x in 0..self.n {
            let mut row = Vec::<f32>::with_capacity(self.m);
            for y in 0..self.m {
                row.push(self.mat[y][x]);
            }
            t_tensor.push(row);
        }

        Self {
            mat: t_tensor,
            m: self.n,
            n: self.m
        }
    }

    pub fn add(&self, other: &Self) -> Result<Self, EError> {
        if self.m != other.m || self.n != other.n {
            return Err(EError::new("Tried to add matrices of unidentical dimensions"));
        }

        let mut res_matrix = Vec::<Vec<f32>>::with_capacity(self.m);

        for (row1, row2) in self.mat.iter().zip(&other.mat) {
            let mut res_row = Vec::<f32>::with_capacity(self.n);

            for (elem_1, elem_2) in row1.iter().zip(row2) {
                res_row.push(elem_1 + elem_2);
            }

            res_matrix.push(res_row);
        }

        let res = Tensor::new(res_matrix)?;

        Ok(res)
    }

    pub fn sub(&self, other: &Self) -> Result<Self, EError> {
        if self.m != other.m || self.n != other.n {
            return Err(EError::new("Tried to subtract matrices of unidentical dimensions"));
        }

        let mut res_matrix = Vec::<Vec<f32>>::with_capacity(self.m);

        for (row1, row2) in self.mat.iter().zip(&other.mat) {
            let mut res_row = Vec::<f32>::with_capacity(self.n);

            for (elem_1, elem_2) in row1.iter().zip(row2) {
                res_row.push(elem_1 - elem_2);
            }

            res_matrix.push(res_row);
        }

        let res = Self::new(res_matrix)?;

        Ok(res)
    }

    pub fn mult(&self, other: &Self) -> Result<Self, EError> {
        if self.n != other.m {
            return Err(EError::new("Tried to multiply incompatible matrices"));
        }

        let mut res_matrix = Vec::<Vec<f32>>::with_capacity(self.m);

        for i in 0..self.m as usize {
            let mut res_row = Vec::<f32>::with_capacity(other.n);

            for j in 0..other.n as usize {
                let mut val: f32 = 0.;
                for k in 0..other.m as usize{
                    val += self.mat[i][k] * other.mat[k][j];
                }
                res_row.push(val);
            }

            res_matrix.push(res_row);
        }

        let res = Self::new(res_matrix)?;

        Ok(res)
    }

    pub fn mult_elemwise(&self, other: &Self) -> Result<Self, EError> {
        if self.m != other.rows() || self.n != other.cols() {
            return Err(EError::new("Tensors should have identical dimensions for elemwise multiplication"));
        }

        let mut res_mat = Vec::<Vec<f32>>::with_capacity(self.m);

        for (mine, his) in self.mat.iter().zip(&other.mat) {
            let mut row = Vec::<f32>::with_capacity(self.n);

            for (x, y) in mine.iter().zip(his) {
                row.push(x * y);
            }

            res_mat.push(row);
        }

        let res = Self::new(res_mat)?;

        Ok(res)
    }

    pub fn mult_scalar(&self, val: f32) -> Self {
        let mut res_matrix = Vec::<Vec<f32>>::with_capacity(self.m);

        for row in self.mat.iter() {
            let mut row_matrix = Vec::<f32>::with_capacity(self.n);

            for elem in row.iter() {
                row_matrix.push(val * *elem);
            }

            res_matrix.push(row_matrix);
        }

        Self {
            mat: res_matrix,
            m: self.m,
            n: self.n
        }
    }

    pub fn clone(&self) -> Self {
        let mut mat = Vec::<Vec<f32>>::with_capacity(self.m);

        for row in self.mat.iter() {
            mat.push(row.clone());
        }

        Self {
            mat: mat,
            m: self.m,
            n: self.n
        }
    }

    pub fn insert_cols_front(&self, elems: Vec<f32>) -> Self {
        let mut mat = Vec::<Vec<f32>>::with_capacity(self.m);

        for row in self.mat.iter() {
            let mut row_mat = Vec::<f32>::with_capacity(self.n + elems.len());

            for elem in elems.iter() {
                row_mat.push(*elem);
            }

            for elem in row.iter() {
                row_mat.push(*elem);
            }

            mat.push(row_mat);
        }

        Self {
            mat,
            m: self.m,
            n: self.n + elems.len()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_constructor() {
        let my_tensor: TensorType = vec![vec![1., 2., 3.]];
        
        let mat_res = Tensor::new(my_tensor);

        assert!(mat_res.is_ok(), "Problem in tensor constructor.");
    }

    #[test]
    fn tensor_transpose() {
        let my_tensor: TensorType = vec![vec![1., 2., 3.],
                                         vec![4., 5., 6.],
                                         vec![7., 8., 9.]];

        let mat = Tensor::new(my_tensor).unwrap();

        let transpose: TensorType = vec![vec![1., 4., 7.],
                                         vec![2., 5., 8.],
                                         vec![3., 6., 9.]];
                                         
        assert_eq!(mat.transpose().mat, transpose);
    }

    #[test]
    fn tensor_display() {
        let my_tensor: TensorType = vec![vec![1., 2., 3.],
                                         vec![4., 5., 6.]];

        let mat = Tensor::new(my_tensor).unwrap();

        let mat_str = String::from("Tensor(2x3) [ [ 1 2 3 ] [ 4 5 6 ] ]");

        assert_eq!(mat_str, format!("{}", mat));
    }

    #[test]
    fn tensor_add() {
        let mat_1 = Tensor::new(vec![vec![1., 2., 3.]]).unwrap();
        let mat_2 = Tensor::new(vec![vec![4., 5., 6.]]).unwrap();

        assert_eq!(mat_1.add(&mat_2).unwrap(), vec![vec![5., 7., 9.]]);
    }

    #[test]
    fn tensor_sub() {
        let mat_1 = Tensor::new(vec![vec![1., 2., 3.]]).unwrap();
        let mat_2 = Tensor::new(vec![vec![4., 5., 6.]]).unwrap();

        assert_eq!(mat_1.sub(&mat_2).unwrap(), vec![vec![-3., -3., -3.]]);
    }

    #[test]
    fn tensor_mult() {
        let mat_1 = Tensor::new(vec![vec![1., 2., 3.], vec![1., 2., 3.]]).unwrap();
        let mat_2 = Tensor::new(vec![vec![1., 2., 3.], vec![1., 2., 3.], vec![1., 2., 3.]]).unwrap();
        
        assert_eq!(mat_1.mult(&mat_2).unwrap(), vec![vec![6., 12., 18.], vec![6., 12., 18.]]);
    }
}
