use nalgebra::DVector;
use std::ops::Sub;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Div;
use std::fmt::Debug;
use rand::random;

use crate::basic_modules::shared::DistanceMetric;

#[derive(Debug, Clone, Copy)]
pub struct Complex {
    a: f64,
    b: f64,
}

impl Complex {
    
    pub fn as_string(&self) -> String {
        let a_string = self.a.round().to_string();
        let b_string = &self.b.round().to_string();
        return a_string + b_string + "i";
    }

    pub fn vector_distance(metric: DistanceMetric, v: &DVector<Self>, w: &DVector<Self>) -> f64 {
            match metric {
                DistanceMetric::Euclidean => {
                    println!("ComplexField + Euclidean");
                    return 0.0;
                }
                DistanceMetric::Minkowski => {
                    println!("ComplexField + Minkowski");
                    return 0.0;
                }
                DistanceMetric::Chebyshev => {
                    println!("ComplexField + Chebyshev");
                    return 0.0;
                }
                _ => {
                    println!("ComplexField with unsupported metric: {:?}", metric);
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
        let real_random_a = random::<f64>();
        let real_random_b = random::<f64>();
        let return_value = Self{a: real_random_a, b: real_random_b};
        return return_value
    }

}

impl Add for Complex {
    type Output = Complex;
  
    fn add(self, other: Complex) -> Complex {
       return Complex {a: self.a + other.a, b: self.b + other.b}
    }
}

impl Sub for Complex {
    type Output = Complex;
  
    fn sub(self, other: Complex) -> Complex {
        return Complex {a: self.a - other.a, b: self.b - other.b}
    }
}

impl Mul for Complex { // add is equivalent to AND
    type Output = Complex;
  
    fn mul(self, other: Complex) -> Complex {
        return Complex {
            a: (self.a * other.a) - (other.b * other.b),
            b: (self.a * other.b) - (other.b * other.a)}
    }
}

impl Div for Complex { // add is equivalent to NAND
    type Output = Complex;
  
    fn div(self, other: Complex) -> Complex {
        return Complex {
            a: ((self.a * other.a) + (self.b * other.b))/(other.a.powf(2.0) + other.b.powf(2.0)),
            b: ((self.b * other.a) - (self.a * other.b))/(other.a.powf(2.0) + other.b.powf(2.0))}
    }
}

impl PartialEq for Complex {
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b
    }
}
