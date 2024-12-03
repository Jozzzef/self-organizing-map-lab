use nalgebra::{DVector,DMatrix};
use rand::Rng;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Sub;

//for text functions
use std::fs;
use std::path::Path;

// Import the Error trait
use std::env;
use std::error::Error;

// shared functionality between modules imports
use crate::basic_modules::shared::DistanceMetric;
//use crate::basic_modules::math_modules::shared_math::*;

#[derive(Debug, Clone, PartialEq)]
pub struct StringGroup(String);

impl StringGroup {
    pub fn vector_distance(metric: DistanceMetric, v: &DVector<Self>, w: &DVector<Self>) -> f64 {
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

    pub fn vector_median(
        metric: DistanceMetric,
        batch_vectors: &mut Vec<DVector<Self>>,
    ) -> DVector<Self> {
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

    pub fn random_value() -> String {
        let mut rng = rand::thread_rng();
        let uint_random: u32 = rng.gen_range(0..=1114111);
        let value = char::from_u32(uint_random);
        let return_value = String::from(value.unwrap());
        return return_value;
    }

    //text functions
    pub fn read_csv_to_matrix(path: String) -> Result<DMatrix<String>, Box<dyn Error>> {
        let relative_path = Path::new(&path);
        let current_dir = env::current_dir()?;
        let combined_path = current_dir.join(relative_path);
        let absolute_path = fs::canonicalize(combined_path)?;

        // Create a CSV reader using the default settings
        let mut rdr = csv::Reader::from_path(absolute_path)?;
        // Skip the first row (header row)
        let _headers = rdr.headers()?;

        // Initialize a vector to hold the flat data from the CSV
        let mut data: Vec<String> = Vec::new();
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
                data.push(field.parse::<String>()?);
            }
            num_rows += 1;
        }

        let matrix = DMatrix::from_row_slice(num_rows, num_cols, &data);
        Ok(matrix)
    }

}

impl Add for StringGroup {
    type Output = StringGroup;

    fn add(self, other: StringGroup) -> StringGroup {
        return Self(self.0 + &other.0);
    }
}

impl Sub for StringGroup {
    type Output = StringGroup;

    fn sub(self, other: StringGroup) -> StringGroup {
        return Self((*self.0).replace(&other.0, ""));
    }
}
