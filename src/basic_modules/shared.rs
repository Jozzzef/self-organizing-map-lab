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

#[derive(Debug, Clone, Copy)]
pub enum Bits {
    Zero,
    One,
}

pub const BINARY_FIELD_LENGTH: usize = 16;
