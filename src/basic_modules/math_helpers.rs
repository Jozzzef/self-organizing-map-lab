//this is for non-learning based functions, la
use nalgebra::DVector;
use std::collections::HashSet;
use std::ops::Sub;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Div;
use std::fmt::Debug;
use rand::random;

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
#[derive(Debug, Clone, Copy)]
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

impl PartialEq for Bits {
    fn eq(&self, other: &Self) -> bool {
       self.as_u8() == other.as_u8() 
    }
}

// COMPLEX 
#[derive(Debug, Clone, Copy)]
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

impl PartialEq for Complex {
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b
    }
}


// Enumerate all possible algebras that can be used in any given SOM, list not extensive right now 
#[derive(Debug, Clone, PartialEq)]
pub enum AlgebraEnum {
    StringGroup(String),
    BitsField(Bits),
    BinaryField([Bits; 16]),
    IntegerRing(isize),
    RealField(f64),
    ComplexField(Complex)
}

#[derive(Debug)]
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

pub trait AlgebraTrait: Debug {
    fn clone_box(&self) -> Box<dyn AlgebraTrait>;
}
impl Clone for Box<dyn AlgebraTrait> {
    fn clone(&self) -> Box<dyn AlgebraTrait> {
        self.clone_box()
    }
}
impl AlgebraTrait for AlgebraEnum {
    fn clone_box(&self) -> Box<dyn AlgebraTrait> {
        match self {
            AlgebraEnum::StringGroup(s) => Box::new(AlgebraEnum::StringGroup(s.clone())),
            AlgebraEnum::BitsField(bits) => Box::new(AlgebraEnum::BitsField(bits.clone())),
            AlgebraEnum::BinaryField(binary_array) => { Box::new(AlgebraEnum::BinaryField(*binary_array)) }
            AlgebraEnum::IntegerRing(value) => Box::new(AlgebraEnum::IntegerRing(*value)),
            AlgebraEnum::RealField(value) => Box::new(AlgebraEnum::RealField(*value)),
            AlgebraEnum::ComplexField(complex) => Box::new(AlgebraEnum::ComplexField(complex.clone())),
        }
    }
}

// 🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋🌋
// Generalized Calculations of Various Metrics
pub fn distance(metric: &DistanceMetric,
    algebra: AlgebraEnum, 
    v: &DVector<Box<dyn AlgebraTrait>>, 
    w: &DVector<Box<dyn AlgebraTrait>>) 
    -> f64 {
        match algebra {
            AlgebraEnum::StringGroup(s) => {
                println!("StringGroup variant with value: {}", s);
                // Add logic for handling `StringGroup`
                match metric {
                    DistanceMetric::Levenshtein => {
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
            AlgebraEnum::BitsField(bits) => {
                println!("BitsField variant with value: {:?}", bits);
                // Add logic for handling `BitsField`
            }
            AlgebraEnum::BinaryField(binary_array) => {
                println!("BinaryField variant with array: {:?}", binary_array);
                // Add logic for handling `BinaryField`
            }
            AlgebraEnum::IntegerRing(integer) => {
                println!("IntegerRing variant with value: {}", integer);
                // Add logic for handling `IntegerRing`
            }
            AlgebraEnum::RealField(real) => {
                println!("RealField variant with value: {}", real);
                // Add logic for handling `RealField`
            }
            AlgebraEnum::ComplexField(complex) => {
                println!("ComplexField variant with value: {:?}", complex);
                // Add logic for handling `ComplexField`
            }
        }
    }

impl AlgebraFunctions for StringGroup {
    fn distance(metric: &DistanceMetric, 
        v: &DVector<Box<dyn AlgebraTrait>>, 
        w: &DVector<Box<dyn AlgebraTrait>>) -> f64 {
            match metric {
                DistanceMetric::Levenshtein => {
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
}

impl AlgebraFunctions for BitsField {
    fn distance(metric: &DistanceMetric, 
        v: &DVector<Box<dyn AlgebraTrait>>, 
        w: &DVector<Box<dyn AlgebraTrait>>) -> f64 {
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
}

impl AlgebraFunctions for BinaryField {
    fn distance(metric: &DistanceMetric, 
        v: &DVector<Box<dyn AlgebraTrait>>, 
        w: &DVector<Box<dyn AlgebraTrait>>) -> f64 {
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
}

impl AlgebraFunctions for IntegerRing {
    fn distance(metric: &DistanceMetric, 
        v: &DVector<Box<dyn AlgebraTrait>>, 
        w: &DVector<Box<dyn AlgebraTrait>>) -> f64 {
            match metric {
                DistanceMetric::Euclidean => {
                    println!("IntegerRing + Euclidean");
                    return 0.0;
                }
                DistanceMetric::Minkowski => {
                    println!("IntegerRing + Minkowski");
                    return 0.0;
                }
                _ => {
                    println!("IntegerRing with unsupported metric: {:?}", metric);
                    return 0.0;
                }
        }
    }
}

impl AlgebraFunctions for RealField {
    fn distance(metric: &DistanceMetric, 
        v: &DVector<Box<dyn AlgebraTrait>>, 
        w: &DVector<Box<dyn AlgebraTrait>>) -> f64 {
            match metric {
                DistanceMetric::Euclidean => {
                    println!("RealField + Euclidean");
                    return 0.0;
                }
                DistanceMetric::Minkowski => {
                    println!("RealField + Minkowski");
                    return 0.0;
                }
                DistanceMetric::Chebyshev => {
                    println!("RealField + Chebyshev");
                    return 0.0;
                }
                _ => {
                    println!("RealField with unsupported metric: {:?}", metric);
                    return 0.0;
                }      
        }
    }
}

impl AlgebraFunctions for ComplexField {
    fn distance(metric: &DistanceMetric, 
        v: &DVector<Box<dyn AlgebraTrait>>, 
        w: &DVector<Box<dyn AlgebraTrait>>) -> f64 {
            match metric {
                DistanceMetric::Euclidean => {
                    println!("RealField + Euclidean");
                    return 0.0;
                }
                DistanceMetric::Minkowski => {
                    println!("RealField + Minkowski");
                    return 0.0;
                }
                DistanceMetric::Chebyshev => {
                    println!("RealField + Chebyshev");
                    return 0.0;
                }
                _ => {
                    println!("RealField with unsupported metric: {:?}", metric);
                    return 0.0;
                }      
        }
    }
}

pub fn generalized_median<'a,T>(algebra_type: AlgebraEnum,  
                                distance_metric: &DistanceMetric,
                                batch_vectors: &mut Vec<DVector<Box<dyn AlgebraTrait>>>
                            ) -> DVector<Box<dyn AlgebraTrait>>{
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
            match algebra_type {
                AlgebraEnum::StringGroup => {
                    println!("Handling {:?}", algebra_type);
                    // Add specific logic for ConcreteAlgebraA
                    curr_dist_sum = curr_dist_sum + StringGroup::distance(distance_metric, v, w);
                },
                AlgebraEnum::BitsField => {
                    println!("Handling {:?}", algebra_type);
                    // Add specific logic for ConcreteAlgebraA
                    curr_dist_sum = curr_dist_sum + BitsField::distance(distance_metric, v, w);
                }
                AlgebraEnum::BinaryField => {
                    println!("Handling {:?}", algebra_type);
                    // Add specific logic for ConcreteAlgebraA
                    curr_dist_sum = curr_dist_sum + BinaryField::distance(distance_metric, v, w);
                }, 
                AlgebraEnum::IntegerRing => {
                    println!("Handling {:?}", algebra_type);
                    // Add specific logic for ConcreteAlgebraA
                    curr_dist_sum = curr_dist_sum + IntegerRing::distance(distance_metric, v, w);
                },
                AlgebraEnum::RealField => {
                    println!("Handling {:?}", algebra_type);
                    // Add specific logic for ConcreteAlgebraA
                    curr_dist_sum = curr_dist_sum + RealField::distance(distance_metric, v, w);
                },
                AlgebraEnum::ComplexField => {
                    println!("Handling {:?}", algebra_type);
                    // Add specific logic for ConcreteAlgebraA
                    curr_dist_sum = curr_dist_sum + ComplexField::distance(distance_metric, v, w);
                },
            }
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

pub fn generalized_random_value(algebra_type: AlgebraEnum) -> Box<dyn AlgebraTrait> {
    match algebra_type {
        AlgebraEnum::StringGroup => {
            println!("Handling {:?}", algebra_type);
            // Add specific logic for ConcreteAlgebraA
            let return_value: Box<dyn AlgebraTrait> = Box::new(StringGroup {value: String::from("String Value")});
            return return_value
        },
        AlgebraEnum::BitsField => {
            println!("Handling {:?}", algebra_type);
            let return_value: Box<dyn AlgebraTrait> = Box::new(BitsField {value: Bits::Zero});
            return return_value
        }
        AlgebraEnum::BinaryField => {
            println!("Handling {:?}", algebra_type);
            let return_value: Box<dyn AlgebraTrait> = Box::new(BinaryField {value: [Bits::Zero; 16]});
            return return_value
        }, 
        AlgebraEnum::IntegerRing => {
            println!("Handling {:?}", algebra_type);
            let return_value: Box<dyn AlgebraTrait> = Box::new(IntegerRing {value: 0});
            return return_value
        },
        AlgebraEnum::RealField => {
            println!("Handling {:?}", algebra_type);
            let real_random = random::<f64>();
            let return_value: Box<dyn AlgebraTrait> = Box::new(RealField {value: real_random});
            return return_value
        },
        AlgebraEnum::ComplexField => {
            println!("Handling {:?}", algebra_type);
            let return_value: Box<dyn AlgebraTrait> = Box::new(ComplexField {value: Complex{a: 0.0, b: 0.0}});
            return return_value
        },
    }
}


pub fn slice_average(slice: &[f64]) -> f64 {
    let sum: f64 = slice.iter().sum();
    sum / slice.len() as f64
}