//Dependencies
use rand::random;
use nalgebra as nalg;
use nalgebra::{DMatrix, DVector};
use std::error::Error; use std::fmt::Debug;
// Import the Error trait
use std::fs;
use std::path::Path;
use std::env;
use std::collections::HashSet;
use std::f64::consts::PI;
use std::cmp::max as tuple_max;
//use csv::Reader;

// SOM functions

//SIMPLE SOM
pub fn simple_som(
    //params
    input_data_file_path:String, 
    map_size_2d:(usize,usize), 
    batch_size: Option<usize>, 
    label_col_index:Option<usize>

) -> DMatrix<DVector<f64>> {
    
    let input_matrix: DMatrix<f64> = read_csv_to_matrix(input_data_file_path).unwrap();

    let mut map_matrix: DMatrix<DVector<f64>> = DMatrix::from_fn(
        map_size_2d.0, 
        map_size_2d.1, 
        |i,j| DVector::from_fn(
            input_matrix.ncols(),
            |i_2, j_2| random::<f64>()));

    let batch_size = batch_size.unwrap_or(1); //default to 1

    let label_col_index = label_col_index.unwrap_or(map_size_2d.1 - 1); //default to the last column

    let input_matrix = input_matrix.transpose();
    print!("{input_matrix}");


    //loop starts here
    //calculate Best Matching Unit, i.e. matching vector = the vector with the smallest distance to the input vector
    //distance_calc(DistanceType::Euclidean);

    //update neighbourhood
    //neighbourhood_update();

    return map_matrix
}


// Helper Functions ----------------------------------------------------------------

pub fn read_csv_to_matrix(path:String) -> Result<DMatrix<f64>, Box<dyn Error>>  {
    
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




pub fn print_matrix_of_vectors(matrix: &DMatrix<DVector<f64>>, float_precision: usize){
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    println!("start {ncols} x {nrows} matrix");
    println!();
    for row in 0..nrows {
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




pub enum DistanceType {
    Euclidean,
    Minkowski,
    Correlation,
    TanimotoSimilarity,
    Levenshtein,
    Entropy,
    Hamming,
    MatrixNeighbourhood
}

pub fn distance_calc<T>(distance_type:&DistanceType, v:&DVector<T>, w:&DVector<T>) -> f64{

    match distance_type {
        DistanceType::Euclidean => {
            println!("Handling Euclidean distance");
            // Add your logic for Euclidean distance here
            return 0
        },
        DistanceType::Minkowski => {
            println!("Handling Minkowski distance");
            // Add your logic for Minkowski distance here
            return 0
        },
        DistanceType::Correlation => {
            println!("Handling Correlation distance");
            // Add your logic for Correlation distance here
            return 0
        },
        DistanceType::TanimotoSimilarity => {
            println!("Handling Tanimoto Similarity");
            // Add your logic for Tanimoto Similarity here
            return 0
        },
        DistanceType::Levenshtein => {
            println!("Handling Levenshtein distance");
            // Add your logic for Levenshtein distance here
            return 0
        },
        DistanceType::Entropy => {
            println!("Handling Entropy-based distance");
            // Add your logic for Entropy here
            return 0
        },
        DistanceType::Hamming => {
            println!("Handling Hamming distance");
            // Add your logic for Hamming distance here
            return 0
        },
        DistanceType::MatrixNeighbourhood => {
            println!("Handling Matrix Neighbourhood distance");
            // Add your logic for Matrix Neighbourhood
            return 0
        }
    }
}




pub fn get_best_matching_unit<T: Clone>(y: &DVector<T>, som_map:&DMatrix<DVector<T>>, distance_type:&DistanceType) -> (DVector<T>, (usize,usize), f64){
    //handles abstract types automatically because of distance_calc func usage
    let mut min_distance: f64 = 0.0;
    let mut min_distance_index: (usize,usize) = (0,0);
    let mut is_first_loop: bool = true;
    // Apply a transformation to each element in place
    for i in 0..som_map.nrows() {
        for j in 0..som_map.ncols() {
            // Access each element by mutable reference
            let curr_dist = distance_calc(distance_type, y, &som_map[(i, j)]);
            if !is_first_loop && curr_dist < min_distance{
                min_distance = curr_dist;
                min_distance_index = (i,j);
            } else {
                min_distance = curr_dist;
                is_first_loop = false;
            }
        }
    }
    let min_vector : DVector<T> = som_map[min_distance_index].clone();
    return (min_vector, min_distance_index, min_distance) //(bmu vector value, it's index in matrix, it's distance from y)
}




pub fn changing_standardized_gaussian(neigh_level: usize, current_input_index:usize, map_dim:(usize, usize), lambda: f64, effect_prop: f64) -> f64{
    // make effect prop more user friendly, since its going to be a small value. Shadow the paramter.
    let effect_prop = 1.0 + (effect_prop / 10.0);
    
    // start off condition: integral of gaussian (from x -> infin), where x=max{map's (ncols, nrows)} i.e. the max neigh level, is equal to .1. (1000 is used instead of infin)
    // let sigma=y, let effect_prop=a=1, and g(x,y)=e^{-yx^2}, solve definite integral euqation: G(1000)-G(sigma)=-.1 -> sigma = G^{-1}(G[1000] - .1) to receive:
    // sigma = ( -ln[ -x^2 * ( -e^{-x^2 * 1000}/x^2 -.1 ) ] ) / x^2
    // this is the initial sigma value
    // the larger the sigma, the smaller the width of the gaussian, therefore call it inverse sigma
    let x = tuple_max(map_dim.0 , map_dim.1) as f64;
    let mut inv_sigma = -f64::ln(-x.powi(2) * (-f64::exp(-x.powi(2) * 1000.0) * effect_prop / x.powi(2) - 0.1) / effect_prop) / x.powi(2);

    // inv_sigma changes linearly, larger to small
    // lambda = the constant of change, this iteratively gets multiplied to inv_sigma to increase inv_sigma's value over time
    
    inv_sigma *= lambda * (current_input_index as f64);

    // check if minimal gaussian width condition is met
    // end off condition: integral of gaussian (from x -> infin), where x=1, is ge or equal to .1. i.e. G(1000)-G(neigh_level) <= .1 ==> use the minimal value
    // G = ((-1/100 (e^{-100y}))
    let mut greater_bound_inv_sigma = 2.3025850929940455; // pre calculated, since effect prop = 1 will likely be the most popular value, can add others later
    let one: f64 = 1.0;
    if effect_prop != 1.0 {
        greater_bound_inv_sigma = -f64::ln(-one.powi(2) * (-f64::exp(-one.powi(2) * 1000.0) * effect_prop / one.powi(2) - 0.1) / effect_prop) / one.powi(2);
    }
    
    if inv_sigma > greater_bound_inv_sigma {
        inv_sigma = greater_bound_inv_sigma;
    }

    //custom gaussian ae^{-yx^2}, y=inverse sigma, a = effect size
    let neigh_level = neigh_level as f64;
    return (-1.0 * effect_prop * inv_sigma * neigh_level.powi(2)).exp()
}




pub fn generalized_median<'a,T>(batch_vectors: &'a Vec<DVector<T>>, distance_type:&'a DistanceType) -> &'a DVector<T>{
    //the generalized definition of a median of a set of objects is "a new object which has the smallest sum of distances to all objects in that set".
    //an optional requirement is that the median has to already be a member of the existing set. 
        //it is enforced here for efficiency purposes. 
        //A generalized mean should be used for the case where it is not necessarily in the batch vector set but in the algebraic set.
    //gen_med = argmin_m{\sum_{i \in S} distance(i,m)}
    //handles abstract types automatically because of distance_calc func usage
    let mut min_distance_index: usize = 0; // the index of that minimal distance vector
    let mut min_sum_distance: f64 = 0; // the distance of that minimal distance vector
    let mut has_first_loop_occured : bool = false;
    
    for i in 0..(batch_vectors.len()) {
        let mut curr_dist_sum: f64 = 0;
        
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




pub fn neighbourhood_update<T, N>(input_vec:DVector<T>, bmu:DVector<T>, bmu_index:Vec<usize>, map:DMatrix<DVector<T>>) -> DMatrix<DVector<T>> {
    
    //**should probably memoize the neighbourhood set creation for all possible bmus

    //create neighbourhood sets
    //the elements are the indices (i,j) of each element within a neighbourhood, 
    //the number of the nieghbouhood is ordered from 0 to k, where 0 is the bmu vector, k is the further neighbourhood
    let mut set_of_neighbourhoods: Vec<Vec<Vec<usize>>> = Vec::new();
    let mut n: usize = 0;
    let range_neighbourhood_indices = cartesian_product(vec![(0..(map.ncols())).collect::<Vec<usize>>(), (0..(map.nrows())).collect::<Vec<usize>>()]);

    loop {
        if n == 0 {
            //bmu is the neighbourhood 0, wrap in another vector for type rules
            set_of_neighbourhoods.push(vec![bmu_index.clone()]);
        } else {
            //build possible sets, same index as the bmu_index elements indexes
            let mut possible_neigh_indices: Vec<Vec<usize>>  = Vec::new();
            for index_val in &bmu_index {
                //n denotes the level of neighbourhood, the number of "steps" away it is from the bmu
                let start = index_val - n;
                let end = index_val + n + 1;
                let vec_from_range: Vec<usize> = (start..end).collect();
                possible_neigh_indices.push(vec_from_range)
            }
            
            //this neighbourhoods possible range of values. still needs to remove the inner ones beloning to other neighbourhoods
            let mut neighbourhood_indices = cartesian_product(possible_neigh_indices);

            for i in 0..n {
                //remove all the interior elements which belong to other neighbourhoods, since the cartesian product outputs them all
                set_difference_for_nested_vectors(&mut neighbourhood_indices, &set_of_neighbourhoods[i]);
            }

            //remove anything that cannot be in the map
            set_intersection_of_nested_vectors(&mut neighbourhood_indices, &range_neighbourhood_indices);

            //if nothing in this neighbourhood, then finished building neighbourhoods
            if neighbourhood_indices.len() == 0 {break;}

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
    for j in 0..set_of_neighbourhoods.len() {
        //distance_calc(DistanceType::MatrixNeighbourhood, bmu, bmu);

    }

    return map
}



//change_shape_of_map function? How to implement change of basis to accomplish this??