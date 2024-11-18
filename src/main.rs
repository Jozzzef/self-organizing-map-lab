//declare modules
mod basic_modules;

//path statements for modules
use basic_modules::simple_som;
use basic_modules::print_matrix_of_vectors;

//dependencies
use nalgebra as nalg;
use nalgebra::{DMatrix, DVector};
use clearscreen;

fn main() {
    clearscreen::clear().expect("failed to clear screen"); //clear terminal
    //println!("Hello, world!");
    let som_output: DMatrix<DVector<f64>> = simple_som(String::from("example_inputs/simple_input.csv"), (9,9), None, None, None);
    print_matrix_of_vectors(&som_output, 1);
    //println!("{som_output}");
    println!("exit");
}
