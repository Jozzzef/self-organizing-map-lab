use nalgebra::DVector;
use std::ops::Sub;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Div;
use std::fmt::Debug;
use rand::distributions::{Bernoulli, Distribution};

use crate::basic_modules::shared::DistanceMetric;
use crate::basic_modules::shared::Bits;
use crate::basic_modules::shared::BINARY_FIELD_LENGTH;


#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinaryField {values: [Bits; BINARY_FIELD_LENGTH]}


impl BinaryField {

    pub fn vector_distance(metric: DistanceMetric, v: &DVector<Self>, w: &DVector<Self>) -> f64 {
        match metric {
            DistanceMetric::TanimotoDisimilarity => {
                println!("StringGroup + Levenshtein");
                return 0.0;
            }
            DistanceMetric::Hamming => {
                println!("StringGroup + Hamming");
                return 0.0;
            }
            _ => {
                println!("StringGroup with unsupported metric: {:?}", metric);
                return 0.0;
            }
        }
    }

    pub fn vector_median(metric: DistanceMetric, batch_vectors: &mut Vec<DVector<Self>>) -> DVector<Self> {
        //the generalized definition of a median of a set of objects is "a new object which has the smallest sum of distances to all objects in that set".
        //an optional requirement is that the median has to already be a member of the existing set. 
            //it is enforced here for efficiency purposes. 
            //A generalized mean should be used for the case where it is not necessarily in the batch vector set but in the algebraic set.
        //gen_med = argmin_m{\sum_{i \in S} distance(i,m)}
        //handles abstract types automatically because of distance_calc func usage
        let mut min_distance_index: usize = 0; // the index of that minimal distance vector
        let mut min_sum_distance: f64 = 0.0; // the distance of that minimal distance vector
        let mut has_first_loop_occured : bool = false;
        
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
            }
            else {
                min_sum_distance = curr_dist_sum;
                // min_distance_index already set to zero
                has_first_loop_occured = true;
            }
        }
        return batch_vectors[min_distance_index].clone();
    }

    pub fn random_value() -> Self {
        let mut arr = [Bits::Zero; BINARY_FIELD_LENGTH];
        for i in 0..(BINARY_FIELD_LENGTH-1) {
            let d = Bernoulli::new(0.5).unwrap(); 
            let v: bool = d.sample(&mut rand::thread_rng());
            arr[i] = Bits::bool_as_bits(v);
        }
        let return_value = BinaryField{ values: arr };
        return return_value
    }

}

impl Add for BinaryField {
    type Output = BinaryField;

    fn add(self, other: BinaryField) -> BinaryField {
        let mut arr = [Bits::Zero; BINARY_FIELD_LENGTH];
        for i in 0..(BINARY_FIELD_LENGTH-1) {
           arr[i] = self.values[i] + other.values[i]; 
        }
        return Self { values: arr } 
    }
}

impl Sub for BinaryField {
    type Output = BinaryField;
  
    fn sub(self, other: BinaryField) -> BinaryField {
        let mut arr = [Bits::Zero; BINARY_FIELD_LENGTH];
        for i in 0..(BINARY_FIELD_LENGTH-1) {
           arr[i] = self.values[i] - other.values[i]; 
        }
        return Self { values: arr }
    }
}

impl Mul for BinaryField { // add is equivalent to AND
    type Output = BinaryField;
  
    fn mul(self, other: BinaryField) -> BinaryField {
        let mut arr = [Bits::Zero; BINARY_FIELD_LENGTH];
        for i in 0..(BINARY_FIELD_LENGTH-1) {
           arr[i] = self.values[i] * other.values[i]; 
        }
        return Self { values: arr }
    }
}

impl Div for BinaryField { // add is equivalent to NAND
    type Output = BinaryField;
  
    fn div(self, other: BinaryField) -> BinaryField {
        let mut arr = [Bits::Zero; BINARY_FIELD_LENGTH];
        for i in 0..(BINARY_FIELD_LENGTH-1) {
           arr[i] = self.values[i] / other.values[i]; 
        }
        return Self { values: arr }
    }
}
