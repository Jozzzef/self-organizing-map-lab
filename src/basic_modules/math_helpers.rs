//this is for non-learning based functions, la
use nalgebra::DVector;
use std::collections::HashSet;
use std::ops::Sub;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Div;

// 🦂🦂🦂🦂🦂🦂🦂🦂🦂🦂🦂
//Set Theoretic Operations

pub fn cartesian_product(vectors: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    // Start with an initial product containing one empty vector
    let mut result: Vec<Vec<usize>> = vec![vec![]];

    for vec in vectors {
        // For each vector, create the new combinations
        result = result.into_iter()
                       .flat_map(|prev| {
                           vec.iter().map(move |&x| {
                               let mut new_combination = prev.clone();
                               new_combination.push(x);
                               new_combination
                           }).collect::<Vec<Vec<usize>>>()
                       })
                       .collect();
    }

    result
}


pub fn set_difference_for_nested_vectors<T>(v: &mut Vec<Vec<T>>, w: &Vec<Vec<T>>) -> Option<T> 
where
    T: PartialEq,
{
    // Retain only those sub-vectors in `v` that are NOT present in `w`
    v.retain(|vec_v| {
        !w.iter().any(|vec_w| vec_v == vec_w)  // Remove if `vec_v` exists in `w`
    });

    // Since the mutation is done in-place, return None to indicate no value is returned
    None
}



pub fn set_intersection_of_nested_vectors<T>(v: &mut Vec<Vec<T>>, w: &Vec<Vec<T>>) -> Option<()> 
where
    T: PartialEq + Eq + Clone + std::hash::Hash,
{
    // Convert `w` to a HashSet for faster lookups
    // the underscore (_) in HashSet<_> is a type placeholder that tells the compiler to infer the type automatically. 
    let w_set: HashSet<_> = w.iter().collect();

    // Retain only those elements in `v` that are present in `w`
    v.retain(|vec_v| w_set.contains(vec_v));

    // Since we're modifying `v` in place, return None
    None
}




// 🗻🗻🗻🗻🗻🗻🗻🗻🗻🗻🗻🗻🗻🗻🗻🗻🗻🗻
// Algebra Definitions 

// Create any sets and operations that are defined on the set that is not already heandled by rust by default 
// BINARY
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Bits {
    Zero,
    One,
}
impl Bits {
    // Method to get the integer value
    fn as_u8(&self) -> u8 {
        match self {
            Bits::Zero => 0,
            Bits::One => 1,
        }
    }
    //Assoc func to get from u8 to Bit
    fn as_bits(u8_num: u8) -> Bits {
        match u8_num {
            0 => Bits::Zero,
            1 => Bits::One,
            _ => Bits::Zero
        }
    }
    // Toggle method to switch between Zero and One
    fn toggle(&mut self) {
        *self = match *self {
            Bits::Zero => Bits::One,
            Bits::One => Bits::Zero,
        }
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


// COMPLEX 
#[derive(Debug, PartialEq, Clone, Copy)]
struct Complex {
    a: f64,
    b: f64,
}
impl Complex {
    // new constructor
    fn new(a: f64, b: f64) -> Self {
        Self {a, b}
    }
    
    // Method to get the integer value
    fn as_string(&self) -> String {
        let a_string = &self.a.round().to_string();
        let b_string = &self.b.round().to_string();
        return a_string.to_owned() + b_string + "i";
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


// Enumerate all possible algebras that can be used in any given SOM, list not extensive right now 
pub struct StringGroup {value: String,}
pub struct BitsField {value: Bits,}
pub struct BinaryField {value: Vec<Bits>}
pub struct IntegerRing {value: isize,}
pub struct RealField {value: f64,}
pub struct ComplexField {value: Complex}


pub enum DistanceMetric {
    Euclidean,
    Minkowski,
    Chebyshev,
    InverseCorrelation,
    TanimotoDisimilarity,
    Levenshtein,
    Hamming,
    CrossEntropy,
    KLDivergence
}

trait GeneralizedDistance {
    fn distance(metric: DistanceMetric, v: Self, w:  Self) -> Self;
}

impl GeneralizedDistance for RealField {
    fn distance(metric: DistanceMetric, v: Self, w:  Self) -> Self {
       match metric {
            DistanceMetric::Euclidean => {
                println!("Processing Euclidean distance");
            }
            DistanceMetric::Minkowski => {
                println!("Processing Minkowski distance");
            }
            DistanceMetric::Chebyshev => {
                println!("Processing Chebyshev distance");
            }
            DistanceMetric::InverseCorrelation => {
                println!("Processing Inverse Correlation");
            }
            DistanceMetric::TanimotoDisimilarity => {
                println!("Processing Tanimoto Dissimilarity");
            }
            DistanceMetric::Levenshtein => {
                println!("Processing Levenshtein distance");
            }
            DistanceMetric::Hamming => {
                println!("Processing Hamming distance");
            }
            DistanceMetric::CrossEntropy => {
                println!("Processing Cross Entropy");
            }
            DistanceMetric::KLDivergence => {
                println!("Processing KL Divergence");
            }
        } 
    }
}



// 🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋
// Generalized Calculations of Various Metrics

pub fn distance_calc<T>(distance_type:&DistanceType, v:&DVector<f64>, w:&DVector<f64>) -> f64{

    match distance_type {
        DistanceType::Euclidean(_) => {
            println!("Handling Euclidean distance");
            //handle euclidean subtraction
            return 0.0
        },
        DistanceType::Minkowski => {
            println!("Handling Minkowski distance");
            // Add your logic for Minkowski distance here
            return 0.0
        },
        DistanceType::Correlation => {
            println!("Handling Correlation distance");
            // Add your logic for Correlation distance here
            return 0.0
        },
        DistanceType::TanimotoSimilarity => {
            println!("Handling Tanimoto Similarity");
            // Add your logic for Tanimoto Similarity here
            return 0.0
        },
        DistanceType::Levenshtein => {
            println!("Handling Levenshtein distance");
            // Add your logic for Levenshtein distance here
            return 0.0
        },
        DistanceType::Entropy => {
            println!("Handling Entropy-based distance");
            // Add your logic for Entropy here
            return 0.0
        },
        DistanceType::Hamming => {
            println!("Handling Hamming distance");
            // Add your logic for Hamming distance here
            return 0.0
        },
        DistanceType::MatrixNeighbourhood => {
            println!("Handling Matrix Neighbourhood distance");
            // Add your logic for Matrix Neighbourhood
            return 0.0
        }
    }
}



pub fn generalized_median<'a,T>(batch_vectors: &'a Vec<DVector<T>>, distance_type:&'a DistanceType) -> &'a DVector<T>{
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
            curr_dist_sum = curr_dist_sum + distance_calc(distance_type, v, w);
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

    return &batch_vectors[min_distance_index]
}


pub fn slice_average(slice: &[f64]) -> f64 {
    let sum: f64 = slice.iter().sum();
    sum / slice.len() as f64
}