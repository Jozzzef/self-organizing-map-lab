// External Library Dependencies
use nalgebra::{DMatrix, DVector};
use std::cmp::max as tuple_max;

//import modules
pub mod math_modules;
pub use math_modules::*;
pub mod visual_modules;
pub use visual_modules::*;

//shared globally within basic_modules
pub mod shared;
use crate::basic_modules::shared::TypeStructs;
use crate::basic_modules::shared::DistanceMetric;

// SIMPLE SOM
pub fn simple_som<T>(
    input_data_file_path: String, // 🔣 training data csv
    map_size_2d: (usize, usize), // (#️⃣, #️⃣) size of map to be trained with the input data
    batch_size: Option<usize>, // 🥞 batch size is the number of vectors to take the median of to reduce training steps 
    lambda_radius_reduction_rate: Option<f64>, // 🫗 neighbourhood size effect reduction rate after each loop
    learning_rate: Option<f64>, // ✖️ the baseline multiplicative factor of how much you want the nodes/vectors to change on each map update, reduces after lambda changes converges
    value_type: Option<TypeStructs> // 🈯 the type of algebra used for the input data and the resulting map
) -> DMatrix<DVector<T>> {
    
    // set default values
    let batch_size = batch_size.unwrap_or(1); //default to 1 
    let lambda = lambda_radius_reduction_rate.unwrap_or(1.01).max(1.00000001); // the mulitplicative factor is at least 1.00000001 since we need an increasing inv_sigma value
    let mut learning_rate = learning_rate.unwrap_or(1.0); // default to a neutral multiplication
    let original_lr = learning_rate; //copy trait automatically invoked for f64 values, original learning rate stays static for comparison
    let value_type = value_type.unwrap_or(TypeStructs::RealField); //default to real numbers for the input matrix and the resulting map 

    // perform SOM based on algebra type
    match value_type {
        TypeStructs::RealField => {
            let input_matrix: DMatrix<f64> = RealField::read_csv_to_matrix(input_data_file_path).unwrap(); // 🔢 init matrix to take vectors as training inputs 

            let mut map_matrix: DMatrix<DVector<f64>> =
                DMatrix::from_fn(map_size_2d.0, map_size_2d.1, |_i, _j| {
                    DVector::from_fn(input_matrix.ncols(), |_i_2, _j_2| RealField::random_value())
                }); // the map matrix uses the user specified dimensions and just inputs random values to initialize the matrix

            let input_matrix = input_matrix.transpose(); //transpose to be able to take row as vector easier
            
           //debug prints 
            print!("input matrix");
            print!("{input_matrix}");

            print!("map matrix");
            RealField::print_matrix_of_vectors(&map_matrix, 2);

            //create buffer of total distances at each neighbourhood update, for convergence metric
            let mut diff_buff: Vec<f64> = vec![];
            //training loop starts here
            let mut num_of_loops_done: usize = 0; //this is the number of inputs used, but generalized since assuming we can go through data multiple times before convergence
            while learning_rate >= 0.0000001 {
                for j in 0..input_matrix.ncols() {
                    //calculate Best Matching Unit, i.e. matching vector = the vector with the smallest distance to the input vector
                    let column_vector = DVector::from_column_slice(&input_matrix.column(j).as_slice());
                    let (bmu_vec, bmu_index, bmu_dist) =
                        RealField::get_best_matching_unit(&column_vector, &map_matrix, DistanceMetric::Euclidean);
                    //update neighbourhood, updates map in place then returns the total distances from the udpate
                    let diff = RealField::neighbourhood_update(
                        &column_vector,
                        &bmu_vec,
                        &bmu_index,
                        &mut map_matrix,
                        &lambda,
                        &learning_rate,
                        &num_of_loops_done,
                    );
                    RealField::print_matrix_of_vectors(&map_matrix, 2);
                    diff_buff.push(diff);
                    println!("{:?}", diff_buff);
                    // converge rate should start at 1 and reduce non monotonically
                    let convergence_metric = convergence_calculator(&diff_buff, 0.2);
                    //learning rate decreases alongside convergence metric, this way I can optimize its rate of change later
                    learning_rate *= convergence_metric.powf(1.0 / 4.0);
                    learning_rate = learning_rate.clamp(0.0, original_lr);
                    num_of_loops_done += 1;
                }
            }
            return map_matrix;
        }
        TypeStructs::BinaryField => {

            let input_matrix: DMatrix<BinaryField> = read_csv_to_matrix(input_data_file_path).unwrap(); // 🔢 init matrix to take vectors as training inputs 
            let mut map_matrix: DMatrix<DVector<BinaryField>> =
                DMatrix::from_fn(map_size_2d.0, map_size_2d.1, |_i, _j| {
                    DVector::from_fn(input_matrix.ncols(), |_i_2, _j_2| BinaryField::random_value())
                }); // the map matrix uses the user specified dimensions and just inputs random values to initialize the matrix
            println!("Binary SOM not implemented yet");    
            return map_matrix
        }
        TypeStructs::Bits => {

            let input_matrix: DMatrix<Bits> = read_csv_to_matrix(input_data_file_path).unwrap(); // 🔢 init matrix to take vectors as training inputs 
            let mut map_matrix: DMatrix<DVector<Bits>> =
                DMatrix::from_fn(map_size_2d.0, map_size_2d.1, |_i, _j| {
                    DVector::from_fn(input_matrix.ncols(), |_i_2, _j_2| Bits::random_value())
                });// the map matrix uses the user specified dimensions and just inputs random values to initialize the matrix
            println!("Bits SOM not implemented yet");    
            return map_matrix
        }
        TypeStructs::Complex => {

            let input_matrix: DMatrix<Complex> = read_csv_to_matrix(input_data_file_path).unwrap(); // 🔢 init matrix to take vectors as training inputs 
            let mut map_matrix: DMatrix<DVector<Complex>> =
                DMatrix::from_fn(map_size_2d.0, map_size_2d.1, |_i, _j| {
                    DVector::from_fn(input_matrix.ncols(), |_i_2, _j_2| Complex::random_value())
                });// the map matrix uses the user specified dimensions and just inputs random values to initialize the matrix
            println!("Complex SOM not implemented yet");    
            return map_matrix
        }
        TypeStructs::IntegerRing => {

            let input_matrix: DMatrix<IntegerRing> = read_csv_to_matrix(input_data_file_path).unwrap(); // 🔢 init matrix to take vectors as training inputs 
            let mut map_matrix: DMatrix<DVector<IntegerRing>> =
                DMatrix::from_fn(map_size_2d.0, map_size_2d.1, |_i, _j| {
                    DVector::from_fn(input_matrix.ncols(), |_i_2, _j_2| IntegerRing::random_value())
                });// the map matrix uses the user specified dimensions and just inputs random values to initialize the matrix
            println!("IntegerRing SOM not implemented yet");    
            return map_matrix
        }
        TypeStructs::StringGroup => {

            let input_matrix: DMatrix<String> = StringGroup::read_csv_to_matrix(input_data_file_path).unwrap(); // 🔢 init matrix to take vectors as training inputs 
            let mut map_matrix: DMatrix<DVector<String>> =
                DMatrix::from_fn(map_size_2d.0, map_size_2d.1, |_i, _j| {
                    DVector::from_fn(input_matrix.ncols(), |_i_2, _j_2| StringGroup::random_value())
                });// the map matrix uses the user specified dimensions and just inputs random values to initialize the matrix
            println!("StringGroup SOM not implemented yet");
            return map_matrix
        }
        _ => {
            panic!("unrecognized algebra inputted.");
        }
    }

}

// 🛤️ converge is calculated by checking the ratio between a chunk of the first difference vs a chunk of the most recent differences in the training loop
pub fn convergence_calculator(diff_buffer: &Vec<f64>, comparison_size: f64) -> f64 {
    // Clamp value into range (0, 1].
    let comparison_size_clamped = comparison_size.clamp(0.001, 1.0);
    let upper_border = (diff_buffer.len() as f64 * comparison_size_clamped) as usize;
    let lower_border = diff_buffer.len() - upper_border;
    let upper_avg = slice_average(&diff_buffer[0..upper_border]);
    let upper_avg = if upper_avg == 0.0 { 1.0 } else { upper_avg };
    let mut lower_avg = if lower_border <= diff_buffer.len() {
        slice_average(&diff_buffer[lower_border..(diff_buffer.len())])
    } else {
        upper_avg
    };
    if lower_avg > upper_avg {
        lower_avg = upper_avg
    };
    let convergence_metric = lower_avg / upper_avg;
    return convergence_metric; //return the proprtion of the newer total distances vs the first total distances.
}

pub fn slice_average(slice: &[f64]) -> f64 {
    let sum: f64 = slice.iter().sum();
    let avg = sum / slice.len() as f64;
    if avg.is_nan() {
        1.0
    } else {
        avg
    }
}
//change_shape_of_map function? How to implement change of basis to accomplish this??