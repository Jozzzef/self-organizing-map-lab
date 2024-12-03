use nalgebra::{DMatrix, DVector};
use std::fmt::Display;

/// Print a given matrix to the console, with specified precision
/// with strings the precision acts as a length limiter, with floats it reduces the decimal places to that length, and with integers it ignores it
pub fn print_matrix_of_vectors<T>(matrix: &DMatrix<DVector<T>>, precision: &usize) 
where 
    T: Display 
{
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
                print!("{:.prec$}", val, prec = precision); // Format each vector value
            }
            print!("]   ");
        }
        println!(); // Newline after each matrix row
        println!();
    }
    println!("end {ncols} x {nrows} matrix");
}