//GENERAL MATH FUNCTIONS
use std::collections::HashSet;
use std::cmp::max as tuple_max;


// 🦂🦂🦂🦂🦂🦂🦂🦂🦂🦂🦂
//Set Theoretic Operations

pub fn cartesian_product(vectors: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    // Start with an initial product containing one empty vector
    let mut result: Vec<Vec<usize>> = vec![vec![]];

    for vec in vectors {
        // For each vector, create the new combinations
        result = result
            .into_iter()
            .flat_map(|prev| {
                vec.iter()
                    .map(move |&x| {
                        let mut new_combination = prev.clone();
                        new_combination.push(x);
                        return new_combination;
                    })
                    .collect::<Vec<Vec<usize>>>()
            })
            .collect();
    }

    return result;
}

pub fn set_difference_for_nested_vectors<T>(v: &mut Vec<Vec<T>>, w: &Vec<Vec<T>>) -> Option<T>
where
    T: PartialEq,
{
    // Retain only those sub-vectors in `v` that are NOT present in `w`
    v.retain(|vec_v| {
        !w.iter().any(|vec_w| vec_v == vec_w) // Remove if `vec_v` exists in `w`
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


// 🔣🔣🔣🔣🔣🔣🔣🔣
//SOM Based Functions

/// 🗼 The gaussian, i.e. the smoothing kernel, used to update the neighbourhood, changes as time goes one, gets smaller in width
pub fn changing_standardized_gaussian(
    neigh_level: usize,
    num_loops: &usize,
    map_dim: (usize, usize),
    lambda: &f64,
    learning_rate: &f64,
) -> f64 {
    let one: f64 = 1.0; //defined as a variable for ease of reading

    // start off condition: 
        // integral of gaussian (from x -> infin), where x=max{map's (ncols, nrows)} i.e. the max neigh level, is equal to .1. (1000 is used instead of infin)
        // let sigma=y, 
        // let learning_rate=a=1
        // g(x,y)=e^{-yx^2}
    // solve definite integral euqation: G(1000)-G(sigma)=-.1 -> sigma = G^{-1}(G[1000] - .1) to receive:
        // sigma = ( -ln[ -x^2 * ( -e^{-x^2 * 1000}/x^2 -.1 ) ] ) / x^2
        // this is the initial sigma value
        // the larger the sigma, the smaller the width of the gaussian, therefore call it inverse sigma
    let x = (tuple_max(map_dim.0, map_dim.1)) as f64; 
    
    // we do not want the gaussian morphing with the learning rate as it stands, so as if learning rate = 1
    // inv_sigma = 
        // (-f64::ln(-x.powi(2) * (-f64::exp(-x.powi(2) * 1000.0) * learning_rate / x.powi(2) - 0.1) / learning_rate) / x.powi(2)).abs(); 
    let mut inv_sigma =
        (-f64::ln(-x.powi(2) * (-f64::exp(-x.powi(2) * 1000.0) * one / x.powi(2) - 0.1) / one)
            / x.powi(2))
        .abs(); //absolute value added toreduce small negative values (non monotnically apporaching 0 as limit)

    // check if minimal gaussian width condition is met
    // end off condition: integral of gaussian (from x -> infin), where x=1, is ge or equal to .1. i.e. G(1000)-G(neigh_level) <= .1 ==> use the minimal value
    // G = ((-1/100 (e^{-100y}))
    //greater_bound_inv_sigma = (-f64::ln(-one.powi(2) * (-f64::exp(-one.powi(2) * 1000.0) * learning_rate / one.powi(2) - 0.1) / learning_rate) / one.powi(2)).abs();
    let greater_bound_inv_sigma = (-f64::ln(
        -one.powi(2) * (-f64::exp(-one.powi(2) * 1000.0) * one / one.powi(2) - 0.1) / one,
    ) / one.powi(2))
    .abs();

    // inv_sigma changes linearly, larger to small
    // lambda = the constant of change, this iteratively gets multiplied to inv_sigma to increase inv_sigma's value over time
    let num_loops_f = tuple_max(*num_loops, 1) as f64;
    inv_sigma *= lambda * num_loops_f;

    //custom gaussian ae^{-yx^2}, y=inverse sigma, a = effect size
    if inv_sigma > greater_bound_inv_sigma {
        inv_sigma = greater_bound_inv_sigma;
    }
    let neigh_level = neigh_level as f64;
    let custom_gaussian = learning_rate * (-1.0 * inv_sigma * neigh_level.powi(2)).exp();
    //println!("{custom_gaussian} | {learning_rate} | {inv_sigma}");
    return custom_gaussian;
}

/// 🛤️ converge is calculated by checking the ratio between a chunk of the first difference vs a chunk of the most recent differences in the training loop
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

/// the average value of any given slice of f64s
pub fn slice_average(slice: &[f64]) -> f64 {
    let sum: f64 = slice.iter().sum();
    let avg = sum / slice.len() as f64;
    if avg.is_nan() {
        1.0
    } else {
        avg
    }
}
