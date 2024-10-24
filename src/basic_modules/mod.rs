//Dependencies
use rand::random;
use nalgebra as nalg;
use nalgebra::{DMatrix, DVector};
use std::error::Error; // Import the Error trait
use std::fs;
use std::path::Path;
use std::env;
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

pub fn distance_calc<T>(distance_type:DistanceType, v:DVector<T>, w:DVector<T>){

    match distance_type {
        DistanceType::Euclidean => {
            println!("Handling Euclidean distance");
            // Add your logic for Euclidean distance here
        },
        DistanceType::Minkowski => {
            println!("Handling Minkowski distance");
            // Add your logic for Minkowski distance here
        },
        DistanceType::Correlation => {
            println!("Handling Correlation distance");
            // Add your logic for Correlation distance here
        },
        DistanceType::TanimotoSimilarity => {
            println!("Handling Tanimoto Similarity");
            // Add your logic for Tanimoto Similarity here
        },
        DistanceType::Levenshtein => {
            println!("Handling Levenshtein distance");
            // Add your logic for Levenshtein distance here
        },
        DistanceType::Entropy => {
            println!("Handling Entropy-based distance");
            // Add your logic for Entropy here
        },
        DistanceType::Hamming => {
            println!("Handling Hamming distance");
            // Add your logic for Hamming distance here
        },
        DistanceType::MatrixNeighbourhood => {
            println!("Handling Matrix Neighbourhood distance");
            // Add your logic for Matrix Neighbourhood
        }
    }
}




pub fn neighbourhood_update<T, N>(bmu:DVector<T>, bmu_index:Vec<usize>, map:DMatrix<DVector<T>>) -> DMatrix<DVector<T>> {
    
    //**should probably memoize the neighbourhood set creation for all possible bmus

    //create neighbourhood sets
    //the elements are the indices (i,j) of each element within a neighbourhood, 
    //the number of the nieghbouhood is ordered from 0 to k, where 0 is the bmu vector, k is the further neighbourhood
    let mut set_of_neighbourhoods: Vec<Vec<Vec<usize>>> = Vec::new();
    let mut index_while_loop: usize = 0;

    loop {
        if index_while_loop == 0 {
            //bmu is the neighbourhood 0, wrap in another vector for type rules
            set_of_neighbourhoods.push(vec![bmu_index.clone()]);
        } else {
            //build possible sets, same index as the bmu_index elements indexes
            let mut possible_neigh_indices: Vec<Vec<usize>>  = Vec::new();
            for index_val in &bmu_index {
                //index_while_loop denotes the level of neighbourhood, the number of "steps" away it is from the bmu
                let start = index_val - index_while_loop;
                let end = index_val + index_while_loop + 1;
                let vec_from_range: Vec<usize> = (start..end).collect();
                possible_neigh_indices.push(vec_from_range)
            }
            
            let mut neighbourhood_indices = cartesian_product(possible_neigh_indices);

            for i in 0..index_while_loop {
                //remove all the interior elements which belong to other neighbourhoods, since the cartesian product outputs them all
                set_difference_for_nested_vectors(&mut neighbourhood_indices, &set_of_neighbourhoods[i]);
            }

            if neighbourhood_indices.len() == 0 {
                break;
            }

            set_of_neighbourhoods.push(neighbourhood_indices);
        }

        index_while_loop += 1;
    }

    //now that neighbourhood sets are defined
    //update all values based on the following generalized formula:
        //
    for j in 0..set_of_neighbourhoods.len() {
        //distance_calc(DistanceType::MatrixNeighbourhood, bmu, bmu);
    }

    return map
}




pub fn get_best_matching_unit<T>(x: DVector<T>, map:&DMatrix<DVector<T>>) -> (DVector<T>, Vec<usize>){
    //handle types automatically

    //return (bmu vector, it's ordered indices to locate it within the map)
    return (x, vec![1,2,3])
}




pub fn generalized_median<T>(batch_vectors: Vec<DVector<T>>) -> DVector<T>{
    //handle types automatically

    return batch_vectors[0]
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

//change_shape_of_map function? How to implement change of basis to accomplish this??