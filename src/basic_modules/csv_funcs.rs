//linear algebra
use nalgebra::DMatrix;

//for text functions
use std::fs;
use std::path::Path;

// Import the Error trait
use std::env;
use std::error::Error;

pub fn read_csv_to_matrix_real_field(path: String) -> Result<DMatrix<f64>, Box<dyn Error>> {
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