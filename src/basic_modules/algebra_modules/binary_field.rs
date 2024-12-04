//Dependencies
use nalgebra::{DVector, DMatrix};

use rand::distributions::{Bernoulli, Distribution};

use std::fmt::Debug;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Sub;

//for text functions
use std::fs;
use std::path::Path;

// Import the Error trait
use std::env;
use std::error::Error;


//global variables import
use crate::basic_modules::shared::Bits;
use crate::basic_modules::shared::BINARY_FIELD_LENGTH;
use crate::basic_modules::shared::DistanceMetric;
use crate::basic_modules::math_modules::*;

//module import
//use crate::basic_modules::math_modules::*;

//define the type as a struct
/// Binary Field
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinaryField ([Bits; BINARY_FIELD_LENGTH]);

// Binary Field Well Known Operations
impl Add for BinaryField {
    type Output = BinaryField;

    fn add(self, other: BinaryField) -> BinaryField {
        let mut arr = [Bits::Zero; BINARY_FIELD_LENGTH];
        for i in 0..(BINARY_FIELD_LENGTH - 1) {
            arr[i] = self.0[i] + other.0[i];
        }
        return Self(arr);
    }
}

impl Sub for BinaryField {
    type Output = BinaryField;

    fn sub(self, other: BinaryField) -> BinaryField {
        let mut arr = [Bits::Zero; BINARY_FIELD_LENGTH];
        for i in 0..(BINARY_FIELD_LENGTH - 1) {
            arr[i] = self.0[i] - other.0[i];
        }
        return Self(arr);
    }
}

impl Mul for BinaryField {
    // add is equivalent to AND
    type Output = BinaryField;

    fn mul(self, other: BinaryField) -> BinaryField {
        let mut arr = [Bits::Zero; BINARY_FIELD_LENGTH];
        for i in 0..(BINARY_FIELD_LENGTH - 1) {
            arr[i] = self.0[i] * other.0[i];
        }
        return Self(arr);
    }
}

impl Div for BinaryField {
    // add is equivalent to NAND
    type Output = BinaryField;

    fn div(self, other: BinaryField) -> BinaryField {
        let mut arr = [Bits::Zero; BINARY_FIELD_LENGTH];
        for i in 0..(BINARY_FIELD_LENGTH - 1) {
            arr[i] = self.0[i] / other.0[i];
        }
        return Self(arr);
    }
}

//Binary Field SOM-Usage-Specific Operations
impl BinaryField {
    /// case #1 (DistanceMetric::TanimotoDisimilarity): Calculates Tanimoto similarity coefficient between two arrays
    /// Tanimoto = (size of intersection) / (size of union) [i.e. Returns value between 0.0 (no similarity) and 1.0 (identical)]
    /// T(A,B) = |A ∩ B| / |A ∪ B|
    /// T_Disimilariy = 1.0 - (|A ∩ B| / |A ∪ B|) i.e. the compliment of it
    /// Useful for treating the Binary number as a whole number, as it checks if the entire array is equal, not the arrays individual elements
    /// 
    /// case #2 (DistanceMetric::Hamming): essentially an XOR of the nested arrays
    /// useful if want to take the inner difference of the arrays into account, as the difference metric is done at the array level (not at the higher DVector level)
    pub fn vector_distance(metric: DistanceMetric, v: &DVector<Self>, w: &DVector<Self>) -> f64 {
        match metric {
            DistanceMetric::TanimotoDisimilarity => {
                //interaction & union are specifically the magnitude of both
                let mut intersection: f64 = 0.0;
                let mut union: f64 = 0.0;
                for i in 0..v.len() {
                    let mut is_arr_eq = true;
                    //.0 is accessing the element defined for the struct BinaryField
                    for j in 0..v[i].0.len(){
                        is_arr_eq = v[i].0[j] == w[i].0[j];
                    }
                    if is_arr_eq {
                        intersection += 1.0;
                    }
                    union += 1.0;
                }
                return 1.0 - (intersection / union) 
            }
            DistanceMetric::Hamming => {
                let mut difference: f64 = 0.0;
                for i in 0..v.len() {
                    for j in 0..v[i].0.len() {
                        if v[i].0[j] != w[i].0[j] {
                            difference += 1.0;
                        }
                    }
                }
                return difference;
            }
            _ => {
                println!("using an unsupported metric: {:?}", metric);
                return 0.0;
            }
        }
    }

    ///The generalized definition of a median of a set of objects is "a new object which has the smallest sum of distances to all objects in that set".
    ///An optional requirement is that the median has to already be a member of the existing set. It is enforced here for efficiency purposes.
    ///A generalized mean should be used for the case where it is not necessarily in the batch vector set but in the algebraic set.
    ///gen_med = argmin_m{\sum_{i \in S} distance(i,m)}
    ///handles abstract types automatically because of distance_calc func usage
    pub fn vector_median(
        metric: DistanceMetric,
        batch_vectors: &mut Vec<DVector<Self>>,
    ) -> DVector<Self> {
        let mut min_distance_index: usize = 0; // the index of that minimal distance vector
        let mut min_sum_distance: f64 = 0.0; // the distance of that minimal distance vector
        let mut has_first_loop_occured: bool = false;

        for i in 0..(batch_vectors.len()) {
            let mut curr_dist_sum: f64 = 0.0;

            for j in 0..(batch_vectors.len()) {
                let v = &batch_vectors[i];
                let w = &batch_vectors[j];
                curr_dist_sum = curr_dist_sum + Self::vector_distance(metric, v, w);
            }

            if has_first_loop_occured {
                if curr_dist_sum < min_sum_distance {
                    min_sum_distance = curr_dist_sum;
                    min_distance_index = i;
                }
            } else {
                min_sum_distance = curr_dist_sum;
                // min_distance_index already set to zero
                has_first_loop_occured = true;
            }
        }
        return batch_vectors[min_distance_index].clone();
    }

    /// Set a random array of bits
    pub fn random_value() -> Self {
        let mut arr = [Bits::Zero; BINARY_FIELD_LENGTH];
        for i in 0..(BINARY_FIELD_LENGTH - 1) {
            let d = Bernoulli::new(0.5).unwrap();
            let v: bool = d.sample(&mut rand::thread_rng());
            arr[i] = Bits::bool_as_bits(v);
        }
        let return_value = BinaryField(arr);
        return return_value;
    }

    /// Best Matching Unit is the closest vector in the map to the current input vector (or median of batch input vectors)
    /// returns (bmu vector value, it's index in matrix, it's distance from y)
    pub fn get_best_matching_unit(
        y: &DVector<BinaryField>,
        som_map: &DMatrix<DVector<BinaryField>>,
        metric: DistanceMetric,
    ) -> (DVector<BinaryField>, Vec<usize>, f64) {
        let mut min_distance: f64 = 0.0;
        let mut min_distance_index: (usize, usize) = (0, 0);
        let mut is_first_loop: bool = true;
        // Apply a transformation to each element in place
        for i in 0..som_map.nrows() {
            for j in 0..som_map.ncols() {
                // Access each element by mutable reference
                let curr_dist = BinaryField::vector_distance(metric, y, &som_map[(i, j)]);
                if !is_first_loop && curr_dist < min_distance {
                    min_distance = curr_dist;
                    min_distance_index = (i, j);
                } else if is_first_loop {
                    min_distance = curr_dist;
                    is_first_loop = false;
                }
            }
        }
        let min_vector = som_map[min_distance_index].clone();
        return (
            min_vector,
            vec![min_distance_index.0, min_distance_index.1],
            min_distance,
        ); 
    }

}