use std::{
    vec::Vec,
    fmt,
};
use super::error::EncephalonError;

type EError = EncephalonError;
type TensorType = Vec<Vec<i32>>;

#[derive(Debug)]
pub struct Tensor {
    pub mat: TensorType,
    m: u32,
    n: u32,
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
            m: m as u32,
            n: n as u32,
        })
    }

    pub fn zeros(m: u32, n: u32) -> Result<Self, EError> {
        if m == 0 || n == 0 {
            return Err(EError::new("Cannot create tensor with zero dimension(s)"));
        }

        let mat: TensorType = vec![vec![0; n as usize]; m as usize];

        Ok(Self { mat, m, n })
    }

    pub fn transpose(&self) -> Self {
        let mut t_tensor = Vec::<Vec<i32>>::new();

        for x in 0..self.n as usize {
            let mut row = Vec::<i32>::new();
            for y in 0..self.m as usize {
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

        let mut res = Self::zeros(self.m, self.n).unwrap();

        for y in 0..self.m as usize {
            for x in 0..self.n as usize {
                res.mat[y][x] = self.mat[y][x] + other.mat[y][x];
            }
        }

        Ok(res)
    }

    pub fn sub(&self, other: &Self) -> Result<Self, EError> {
        if self.m != other.m || self.n != other.n {
            return Err(EError::new("Tried to subtract matrices of unidentical dimensions"));
        }

        let mut res = Self::zeros(self.m, self.n).unwrap();

        for y in 0..self.m as usize {
            for x in 0..self.n as usize {
                res.mat[y][x] = self.mat[y][x] - other.mat[y][x];
            }
        }

        Ok(res)
    }

    pub fn mult(&self, other: &Self) -> Result<Self, EError> {
        if self.n != other.m {
            return Err(EError::new("Tried to multiply incompatible matrices"));
        }

        let mut res = Self::zeros(self.m, other.n).unwrap();

        for i in 0..self.m as usize {
            for j in 0..other.n as usize {
                for k in 0..other.m as usize{
                    res.mat[i][j] += self.mat[i][k] * other.mat[k][j];
                }
            }
        }

        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_constructor() {
        let my_tensor: TensorType = vec![vec![1, 2, 3]];
        
        let mat_res = Tensor::new(my_tensor);

        assert!(mat_res.is_ok(), "Problem in tensor constructor.");
    }

    #[test]
    fn tensor_transpose() {
        let my_tensor: TensorType = vec![vec![1, 2, 3],
                                         vec![4, 5, 6],
                                         vec![7, 8, 9]];

        let mat = Tensor::new(my_tensor).unwrap();

        let transpose: TensorType = vec![vec![1, 4, 7],
                                         vec![2, 5, 8],
                                         vec![3, 6, 9]];
                                         
        assert_eq!(mat.transpose().mat, transpose);
    }

    #[test]
    fn tensor_display() {
        let my_tensor: TensorType = vec![vec![1, 2, 3],
                                         vec![4, 5, 6]];

        let mat = Tensor::new(my_tensor).unwrap();

        let mat_str = String::from("Tensor(2x3) [ [ 1 2 3 ] [ 4 5 6 ] ]");

        assert_eq!(mat_str, format!("{}", mat));
    }

    #[test]
    fn tensor_add() {
        let mat_1 = Tensor::new(vec![vec![1, 2, 3]]).unwrap();
        let mat_2 = Tensor::new(vec![vec![4, 5, 6]]).unwrap();

        assert_eq!(mat_1.add(&mat_2).unwrap(), vec![vec![5, 7, 9]]);
    }

    #[test]
    fn tensor_sub() {
        let mat_1 = Tensor::new(vec![vec![1, 2, 3]]).unwrap();
        let mat_2 = Tensor::new(vec![vec![4, 5, 6]]).unwrap();

        assert_eq!(mat_1.sub(&mat_2).unwrap(), vec![vec![-3, -3, -3]]);
    }

    #[test]
    fn tensor_mult() {
        let mat_1 = Tensor::new(vec![vec![1, 2, 3]]).unwrap();
        let mat_2 = Tensor::new(vec![vec![1], vec![2], vec![3]]).unwrap();
        
        assert_eq!(mat_1.mult(&mat_2).unwrap(), vec![vec![14]]);
    }
}