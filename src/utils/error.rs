use std::fmt;

#[derive(Debug)]
pub struct EncephalonError {
    msg: String
}

impl fmt::Display for EncephalonError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} (in '{}' at {})", self.msg, file!(), line!())
    }
}

impl EncephalonError {
    pub fn new(m: &'static str) -> Self {
        Self {
            msg: String::from(m)
        }
    }
}