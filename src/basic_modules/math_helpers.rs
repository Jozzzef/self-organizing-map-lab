//this is for non-learning based functions, la
use nalgebra::DVector;
use std::collections::HashSet;
use std::ops::Sub;
use std::ops::Add;

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

// 🫛 Sets (i.e. creation of non primitive types)
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

    // Toggle method to switch between Zero and One
    // This is the Only Binary Operation of Bits, add and subtract is just an implementation of this
    fn toggle(&mut self) -> Bits {
        *self = match *self {
            Bits::Zero => Bits::One,
            Bits::One => Bits::Zero,
        }
    }

}
impl Add for Bits {
    type Output = Bits;

    fn add(self, other: Bits) -> Bits {
       if (self != other){
        return self
       } else{
        return Bits.toggle(&self)
       }
    }
}


// Primitive Sets and Sets Created Above Are Applied Binary Operations & Additional Methods 
pub enum Algebras {
    StringGroup(String), //the binary operation being concatenation, 
    BitsGroup(Bits),
    BinaryField,
    ProbabilityField,
    IntegerRing, 
    RationalField,
    RealField,
    ComplexField,
}

pub enum DistanceMetrics {
    Euclidean(DVector<f64>),
    Minkowski,
    Chebyshev,
    InverseCorrelation,
    TanimotoDisimilarity,
    Levenshtein,
    Entropy,
    Hamming
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