//declare modules
mod basic_modules;

//path statements for modules
use basic_modules::simple_som;
use basic_modules::print_matrix_of_vectors;
use basic_modules::basic_visualization;

//dependencies
use nalgebra as nalg;
use nalgebra::{DMatrix, DVector};
use clearscreen;

const IMAGE_PATH: &str = "./image_outputs/image.svg";

fn main() {
    clearscreen::clear().expect("failed to clear screen"); //clear terminal
    //println!("Hello, world!");
    let som_output: DMatrix<DVector<f64>> = simple_som(String::from("example_inputs/simple_input.csv"), (9,9), None, None, None);
    print_matrix_of_vectors(&som_output, 1);
    match basic_visualization(&som_output, 0.7, 0.9, IMAGE_PATH) {
        Ok(()) => println!("Image Outputted"),
        Err(e) => eprintln!("Error: {}", e)
    }   
    println!("exit");
}
