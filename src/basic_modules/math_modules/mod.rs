//this is for non-learning based functions, la
//use nalgebra::DVector;

pub mod binary_field;
pub use binary_field::*;
pub mod bits_field;
pub use bits_field::*;
pub mod complex_field;
pub use complex_field::*;
pub mod integer_ring;
pub use integer_ring::*;
pub mod real_field;
pub use real_field::*;
pub mod string_group;
pub use string_group::*;

mod shared_math;