// External Library Dependencies
use nalgebra::{DMatrix, DVector};

//import child modules
pub mod math_modules;
pub use math_modules::convergence_calculator;
pub mod algebra_modules;
pub use algebra_modules::*;
pub mod visual_modules;
pub use visual_modules::print_matrix_of_vectors;
//sibling module imports
//shared globals
pub mod shared;
use shared::DistanceMetric;
// SIMPLE SOMs based on types

/// Simple SOM using f64 values
/// The operations used are implemented in the struct RealField (our custom general algebra of f64s)
/// paramaters: {
///    input_data_file_path: String, // 🔣 training data csv
///    map_size_2d: (usize, usize), // (#️⃣, #️⃣) size of map to be trained with the input data
///    batch_size: Option<usize>, // 🥞 batch size is the number of vectors to take the median of to reduce training steps 
///    lambda_radius_reduction_rate: Option<f64>, // 🫗 neighbourhood size effect reduction rate after each loop
///    learning_rate: Option<f64>, // ✖️ the baseline multiplicative factor of how much you want the nodes/vectors to change on each map update, reduces after lambda changes converges
/// }
pub fn simple_som_real_field(
    input_data_file_path: String, // 🔣 training data csv
    map_size_2d: (usize, usize), // (#️⃣, #️⃣) size of map to be trained with the input data
    batch_size: Option<usize>, // 🥞 batch size is the number of vectors to take the median of to reduce training steps 
    lambda_radius_reduction_rate: Option<f64>, // 🫗 neighbourhood size effect reduction rate after each loop
    learning_rate: Option<f64>, // ✖️ the baseline multiplicative factor of how much you want the nodes/vectors to change on each map update, reduces after lambda changes converges
) -> DMatrix<DVector<f64>> {
    
    // set default values
    let batch_size = batch_size.unwrap_or(1); //default to 1 
    let lambda = lambda_radius_reduction_rate.unwrap_or(1.01_f64).max(1.00000001_f64); // the mulitplicative factor is at least 1.00000001 since we need an increasing inv_sigma value
    let mut learning_rate = learning_rate.unwrap_or(1.0); // default to a neutral multiplication
    let original_lr = learning_rate; //copy trait automatically invoked for f64 values, original learning rate stays static for comparison

    let input_matrix: DMatrix<f64> = RealField::read_csv_to_matrix_real_field(input_data_file_path).unwrap(); // 🔢 init matrix to take vectors as training inputs 

    let mut map_matrix: DMatrix<DVector<f64>> =
        DMatrix::from_fn(map_size_2d.0, map_size_2d.1, |_i, _j| {
            DVector::from_fn(input_matrix.ncols(), |_i_2, _j_2| RealField::random_value())
        }); // the map matrix uses the user specified dimensions and just inputs random values to initialize the matrix

    let input_matrix = input_matrix.transpose(); //transpose to be able to take row as vector easier
    
    //debug prints 
    print!("input matrix");
    print!("{input_matrix}");

    print!("map matrix");
    let precision: usize = 2;
    print_matrix_of_vectors(&map_matrix, &precision);

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
            print_matrix_of_vectors(&map_matrix, &precision);
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