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

    return map_matrix
}


// Helper Functions

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

pub fn euclidean_distance(){
    
}

pub fn neighbourhood_update(){
    
}

