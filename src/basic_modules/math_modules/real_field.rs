use nalgebra::{DVector,DMatrix};
use rand::random;
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

// shared functionality between modules imports
use crate::basic_modules::shared::DistanceMetric;
use crate::basic_modules::math_modules::shared_math::*;

//initialize the algebra's struct
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealField(pub f64);

//implement inherent operations to the algebra that are needed for SOMs
impl RealField {
    pub fn vector_distance(metric: DistanceMetric, v: &DVector<f64>, w: &DVector<f64>) -> f64 {
        match metric {
            DistanceMetric::Euclidean => {
                // println!("RealField + Euclidean");
                assert_eq!(v.len(), w.len(), "Vectors must have the same length");
                return (v - w).norm(); // Compute the difference and calculate the L2 norm
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

    pub fn vector_median(
        metric: DistanceMetric,
        batch_vectors: &mut Vec<DVector<f64>>,
    ) -> DVector<f64> {
        //the generalized definition of a median of a set of objects is "a new object which has the smallest sum of distances to all objects in that set".
        //an optional requirement is that the median has to already be a member of the existing set.
        //it is enforced here for efficiency purposes.
        //A generalized mean should be used for the case where it is not necessarily in the batch vector set but in the algebraic set.
        //gen_med = argmin_m{\sum_{i \in S} distance(i,m)}
        //handles abstract types automatically because of distance_calc func usage
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

    pub fn random_value() -> f64 {
        return random::<f64>();
    }

    // Best Matching Unit is the closest vector in the map to the current input vector (or median of batch input vectors)
    pub fn get_best_matching_unit(
        y: &DVector<f64>,
        som_map: &DMatrix<DVector<f64>>,
        metric: DistanceMetric,
    ) -> (DVector<f64>, Vec<usize>, f64) {
        let mut min_distance: f64 = 0.0;
        let mut min_distance_index: (usize, usize) = (0, 0);
        let mut is_first_loop: bool = true;
        // Apply a transformation to each element in place
        for i in 0..som_map.nrows() {
            for j in 0..som_map.ncols() {
                // Access each element by mutable reference
                let curr_dist = RealField::vector_distance(metric, y, &som_map[(i, j)]);
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
        ); //(bmu vector value, it's index in matrix, it's distance from y)
    }

    pub fn neighbourhood_update(
        input_vec: &DVector<f64>,
        bmu: &DVector<f64>,
        bmu_index: &Vec<usize>,
        map: &mut DMatrix<DVector<f64>>,
        lambda: &f64,
        learning_rate: &f64,
        num_loops: &usize,
    ) -> f64 {
        //**should probably memoize the neighbourhood set creation for all possible bmus

        //create neighbourhood sets
        //the elements are the indices (i,j) of each element within a neighbourhood,
        //the number of the nieghbouhood is ordered from 0 to k, where 0 is the bmu vector, k is the further neighbourhood
        let mut set_of_neighbourhoods: Vec<Vec<Vec<usize>>> = Vec::new();
        let mut n: usize = 0;
        let range_neighbourhood_indices = cartesian_product(vec![
            (0..(map.ncols())).collect::<Vec<usize>>(),
            (0..(map.nrows())).collect::<Vec<usize>>(),
        ]);
        let mut total_difference = 0.0;

        loop {
            if n == 0 {
                //bmu is the neighbourhood 0, wrap in another vector for type rules
                set_of_neighbourhoods.push(vec![bmu_index.clone()]);
            } else {
                //build possible sets, same index as the bmu_index elements indexes
                let mut possible_neigh_indices: Vec<Vec<usize>> = Vec::new();
                for index_val in bmu_index {
                    //n denotes the level of neighbourhood, the number of "steps" away it is from the bmu
                    let start: usize;
                    if n > *index_val {
                        start = 0;
                    } else {
                        start = index_val - n;
                    }
                    let end = index_val + n + 1; //+1 for the vec from range
                    let vec_from_range: Vec<usize> = (start..end).collect();
                    possible_neigh_indices.push(vec_from_range)
                }

                //this neighbourhoods possible range of values. still needs to remove the inner ones beloning to other neighbourhoods
                let mut neighbourhood_indices = cartesian_product(possible_neigh_indices);

                for i in 0..n {
                    //remove all the interior elements which belong to other neighbourhoods, since the cartesian product outputs them all
                    set_difference_for_nested_vectors(
                        &mut neighbourhood_indices,
                        &set_of_neighbourhoods[i],
                    );
                }

                //remove anything that cannot be in the map
                set_intersection_of_nested_vectors(
                    &mut neighbourhood_indices,
                    &range_neighbourhood_indices,
                );

                //if nothing in this neighbourhood, then finished building neighbourhoods
                if neighbourhood_indices.len() == 0 {
                    break;
                }

                //add to set of neighbourhoods
                set_of_neighbourhoods.push(neighbourhood_indices);
            }

            //index of our neighbourhood
            n += 1;
        }

        //now that neighbourhood sets are defined
        //update all values based on the following generalized formula:
        // current_element = current_element + changing_standardized_gaussian(x=neighbourhood_level)*(x_input - current_element)
        // changing_standardized_gaussian:= gaussian starts wide during training, ends thin near end of training
        for j in 0..set_of_neighbourhoods.len() - 1 {
            for p in 0..set_of_neighbourhoods[j].len() {
                let index_i = set_of_neighbourhoods[j][p][0];
                let index_j = set_of_neighbourhoods[j][p][1];
                let w: DVector<f64> = map[(index_i, index_j)].clone();
                let vec_diff = input_vec - w.clone();
                // edit the current vector in the neighbourhood
                let w_new = w.clone()
                    + (changing_standardized_gaussian(
                        j,
                        num_loops,
                        (map.ncols(), map.nrows()),
                        lambda,
                        learning_rate,
                    ) * vec_diff);
                //add rolling differences so we can see convergence
                let incr_distance =
                    (RealField::vector_distance(DistanceMetric::Euclidean, &w_new, &w)).abs();
                total_difference += incr_distance;
                //assign new vector
                *map.get_mut((index_i, index_j)).unwrap() = w_new;
            }
        }

        return total_difference;
    }

    //text functions
    pub fn read_csv_to_matrix(path: String) -> Result<DMatrix<f64>, Box<dyn Error>> {
        let relative_path = Path::new(&path);
        let current_dir = env::current_dir()?;
        let combined_path = current_dir.join(relative_path);
        let absolute_path = fs::canonicalize(combined_path)?;

        // Create a CSV reader using the default settings
        let mut rdr = csv::Reader::from_path(absolute_path)?;
        // Skip the first row (header row)
        let _headers = rdr.headers()?;

        // Initialize a vector to hold the flat data from the CSV
        let mut data: Vec<f64> = Vec::new();
        let mut num_rows = 0;
        let mut num_cols = 0;

        // Iterate over each record in the CSV
        for (i, result) in rdr.records().enumerate() {
            let record = result?; // Unwrap the result using `?`

            // Get the number of columns from the first record
            if i == 0 {
                num_cols = record.len();
            }

            // Parse each field as a f64 and collect it into the data vector
            for field in record.iter() {
                data.push(field.parse::<f64>()?);
            }
            num_rows += 1;
        }

        let matrix = DMatrix::from_row_slice(num_rows, num_cols, &data);
        Ok(matrix)
    }

    pub fn print_matrix_of_vectors(matrix: &DMatrix<DVector<f64>>, float_precision: usize) {
        let nrows = matrix.nrows();
        let ncols = matrix.ncols();
        println!("start {ncols} x {nrows} matrix");
        println!();
        for row in 0..nrows {
            println!("m={row}");
            for col in 0..ncols {
                let vector = &matrix[(row, col)]; // Access the vector in the matrix cell
                print!("    [");
                for (i, val) in vector.iter().enumerate() {
                    if i > 0 {
                        print!(", "); // Print a comma between vector elements
                    }
                    print!("{:.prec$}", val, prec = float_precision); // Format each vector value
                }
                print!("]   ");
            }
            println!(); // Newline after each matrix row
            println!();
        }
        println!("end {ncols} x {nrows} matrix");
    }
}

// implement well known algebra operations

// Implement Deref to allow transparent access to f64 methods
impl std::ops::Deref for RealField {
    type Target = f64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Add for RealField {
    type Output = RealField;

    fn add(self, other: RealField) -> RealField {
        return Self(self.0 + other.0);
    }
}

impl Sub for RealField {
    type Output = RealField;

    fn sub(self, other: RealField) -> RealField {
        return Self(self.0 - other.0);
    }
}

impl Mul for RealField {
    // add is equivalent to AND
    type Output = RealField;

    fn mul(self, other: RealField) -> RealField {
        return Self(self.0 * other.0);
    }
}

impl Div for RealField {
    // add is equivalent to NAND
    type Output = RealField;

    fn div(self, other: RealField) -> RealField {
        return Self(self.0 / other.0);
    }
}
