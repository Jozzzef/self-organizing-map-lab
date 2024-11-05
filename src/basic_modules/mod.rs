//Dependencies
use rand::random;
use nalgebra::{DMatrix, DVector};
use std::cmp::max as tuple_max;

mod math_helpers;
pub use math_helpers::*;
mod text_helpers;
pub use text_helpers::*;

// SOM functions

//SIMPLE SOM
pub fn simple_som(
    //params
    input_data_file_path:String, 
    map_size_2d:(usize,usize), 
    batch_size: Option<usize>, 
    convergence_threshold: f64,
    lambda_reduction_rate: f64

) -> DMatrix<DVector<f64>> {
    
    let input_matrix: DMatrix<f64> = read_csv_to_matrix(input_data_file_path).unwrap();

    let mut map_matrix: DMatrix<DVector<f64>> = DMatrix::from_fn(
        map_size_2d.0, 
        map_size_2d.1, 
        |_i,_j| DVector::from_fn(
            input_matrix.ncols(),
            |_i_2, _j_2| random::<f64>()));

    let batch_size = batch_size.unwrap_or(1); //default to 1

    let input_matrix = input_matrix.transpose();
    print!("{input_matrix}");

    //create buffer of total distances at each neighbourhood update, for convergence metric
    let mut diff_buff: Vec<f64> = vec![];
    //convergence metric starts closer to 1 and reduces and reduces non monotonically
    let mut convergence_metric: f64;
    //the learning rate only decreases when the convergence metric threshold is reached
    let mut learning_rate_lambda: f64 = 1.0;
    //training loop starts here
    while learning_rate_lambda != 0.0 {
        for j in 0..input_matrix.ncols() {
            //calculate Best Matching Unit, i.e. matching vector = the vector with the smallest distance to the input vector
            let column_vector = DVector::from_column_slice(&input_matrix.column(j).as_slice());
            let (bmu_vec, bmu_index, bmu_dist) = get_best_matching_unit(&column_vector, &map_matrix, &DistanceType::Euclidean);

            //update neighbourhood, updates map in place then returns the total distances from the udpate
            let diff = neighbourhood_update(&column_vector, &bmu_vec, &bmu_index, &mut map_matrix, &learning_rate_lambda);
            diff_buff.push(diff);
            convergence_metric = convergence_calculator(&diff_buff, 0.2);
            if convergence_metric <= convergence_threshold {
                //start learning rate reduction since natural convergence reached threshold
                let reduced_lambda = lambda_reduction_rate / 10.0;
                learning_rate_lambda = reduced_lambda.clamp(0.0, 1.0)
            }
        }
    }
    return map_matrix
}


// Best Matching Unit is the closest vector in the map to the current input vector (or median of batch input vectors)
pub fn get_best_matching_unit<T: Clone>(y: &DVector<T>, som_map:&DMatrix<DVector<T>>, distance_type:&DistanceType) -> (DVector<T>, Vec<usize>, f64){
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
    return (min_vector, vec![min_distance_index.0, min_distance_index.1], min_distance) //(bmu vector value, it's index in matrix, it's distance from y)
}



//The gaussian, i.e. the smoothing kernel, used to update the neighbourhood, changes as time goes one
pub fn changing_standardized_gaussian(neigh_level: usize, current_input_index:usize, map_dim:(usize, usize), lambda: f64, learning_rate: f64) -> f64{
    // make learning rate (i.e. how much effect the gaussian has) more user friendly, since its going to be a small value. Shadow the paramter.
    let learning_rate = 1.0 + (learning_rate / 10.0);
    
    // start off condition: integral of gaussian (from x -> infin), where x=max{map's (ncols, nrows)} i.e. the max neigh level, is equal to .1. (1000 is used instead of infin)
    // let sigma=y, let learning_rate=a=1, and g(x,y)=e^{-yx^2}, solve definite integral euqation: G(1000)-G(sigma)=-.1 -> sigma = G^{-1}(G[1000] - .1) to receive:
    // sigma = ( -ln[ -x^2 * ( -e^{-x^2 * 1000}/x^2 -.1 ) ] ) / x^2
    // this is the initial sigma value
    // the larger the sigma, the smaller the width of the gaussian, therefore call it inverse sigma
    let x = tuple_max(map_dim.0 , map_dim.1) as f64;
    let mut inv_sigma = -f64::ln(-x.powi(2) * (-f64::exp(-x.powi(2) * 1000.0) * learning_rate / x.powi(2) - 0.1) / learning_rate) / x.powi(2);

    // inv_sigma changes linearly, larger to small
    // lambda = the constant of change, this iteratively gets multiplied to inv_sigma to increase inv_sigma's value over time
    
    inv_sigma *= lambda * (current_input_index as f64);

    // check if minimal gaussian width condition is met
    // end off condition: integral of gaussian (from x -> infin), where x=1, is ge or equal to .1. i.e. G(1000)-G(neigh_level) <= .1 ==> use the minimal value
    // G = ((-1/100 (e^{-100y}))
    let mut greater_bound_inv_sigma = 2.3025850929940455; // pre calculated, since effect prop = 1 will likely be the most popular value, can add others later
    let one: f64 = 1.0;
    if learning_rate != 1.0 {
        greater_bound_inv_sigma = -f64::ln(-one.powi(2) * (-f64::exp(-one.powi(2) * 1000.0) * learning_rate / one.powi(2) - 0.1) / learning_rate) / one.powi(2);
    }
    
    if inv_sigma > greater_bound_inv_sigma {
        inv_sigma = greater_bound_inv_sigma;
    }

    //custom gaussian ae^{-yx^2}, y=inverse sigma, a = effect size
    let neigh_level = neigh_level as f64;
    return (-1.0 * learning_rate * inv_sigma * neigh_level.powi(2)).exp()
}



pub fn neighbourhood_update<T>(input_vec:&DVector<T>, bmu:&DVector<T>, bmu_index:&Vec<usize>, map:&mut DMatrix<DVector<T>>, lambda:&f64) -> f64 {
    
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
            for index_val in bmu_index {
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

    let total_difference = 0.0;
    return total_difference
}


pub fn convergence_calculator(diff_buffer: &Vec<f64>, comparison_size:f64) -> f64 { 
    // Clamp value into range (0, 1].
    let comparison_size_clamped = comparison_size.clamp(0.001, 1.0);
    let upper_border = (diff_buffer.len() as f64 * comparison_size_clamped) as usize;
    let lower_border = diff_buffer.len() - upper_border;
    let upper_avg = slice_average(&diff_buffer[0..upper_border]);
    let lower_avg = slice_average(&diff_buffer[lower_border..(diff_buffer.len()-1)]);
    let convergence_metric = lower_avg / upper_avg;
    
    return convergence_metric //return the proprtion of the newer total distances vs the first total distances. 
}


//change_shape_of_map function? How to implement change of basis to accomplish this??