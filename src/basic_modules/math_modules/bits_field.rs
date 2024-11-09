use nalgebra::DVector;
use std::ops::Sub;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Div;
use rand::distributions::{Bernoulli, Distribution};

use crate::basic_modules::shared::DistanceMetric;
use crate::basic_modules::shared::Bits;

impl Bits {
    // Method to get the integer value
    pub fn as_u8(&self) -> u8 {
        match self {
            Bits::Zero => 0,
            Bits::One => 1,
        }
    }
    //Assoc func to get from u8 to Bit
    pub fn u8_as_bits(u8_num: u8) -> Bits {
        match u8_num {
            0 => Bits::Zero,
            1 => Bits::One,
            _ => Bits::Zero
        }
    }
    pub fn bool_as_bits(bool_v: bool) -> Bits {
        match bool_v {
            true => Bits::One,
            false => Bits::Zero,
        }
    }
    pub fn bits_as_bool(bit_v: Bits) -> bool {
        match bit_v {
            Bits::One => true,
            Bits::Zero => false,
        }
    }
    // Toggle method to switch between Zero and One
    pub fn toggle(&mut self) {
        *self = match *self {
            Bits::Zero => Bits::One,
            Bits::One => Bits::Zero,
        }
    }

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
        let d = Bernoulli::new(0.5).unwrap(); 
        let v: bool = d.sample(&mut rand::thread_rng());
        let return_value = Bits::bool_as_bits(v);
        return return_value
    }

}

impl Add for Bits { // add is equivalent to OR
    type Output = Bits;
  
    fn add(self, other: Bits) -> Bits {
       if self == Bits::One || other == Bits::One {
            return Bits::One 
       } else{ return Bits::Zero }
    }
}

impl Sub for Bits { // add is equivalent to NOR
    type Output = Bits;
  
    fn sub(self, other: Bits) -> Bits {
       if !(self == Bits::One) || !(other == Bits::One) {
            return Bits::One 
       } else{ return Bits::Zero }
    }
}

impl Mul for Bits { // add is equivalent to AND
    type Output = Bits;
  
    fn mul(self, other: Bits) -> Bits {
       if self == Bits::One && other == Bits::One {
            return Bits::One 
       } else{ return Bits::Zero }
    }
}

impl Div for Bits { // add is equivalent to NAND
    type Output = Bits;
  
    fn div(self, other: Bits) -> Bits {
       if !(self == Bits::One) && !(other == Bits::One) {
            return Bits::One 
       } else{ return Bits::Zero }
    }
}

impl PartialEq for Bits {
    fn eq(&self, other: &Self) -> bool {
       self.as_u8() == other.as_u8() 
    }
}
