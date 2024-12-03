use super::{BinaryField, Complex, IntegerRing};

#[derive(Debug, Copy, Clone)]
pub enum DistanceMetric {
    Euclidean,
    Minkowski,
    Chebyshev,
    InverseCorrelation,
    TanimotoDisimilarity,
    Levenshtein,
    Hamming,
    CrossEntropy,
    KLDivergence,
}

pub enum TypeStructs {
    RealField,
    BinaryField,
    Bits,
    Complex,
    IntegerRing,
    StringGroup 
}

#[derive(Debug, Clone, Copy)]
pub enum Bits {
    Zero,
    One,
}

pub const BINARY_FIELD_LENGTH: usize = 16;
