//declare modules
mod basic_modules;

//path statements for modules
use basic_modules::simple_som_real_field;
use basic_modules::visual_modules::{print_matrix_of_vectors, 
                                    basic_visualization};
                                    
//std imports
use std::env;
use std::fs;

//external dependencies
use nalgebra::{DMatrix, DVector};
use clearscreen;

fn main() {
    //setup
    clearscreen::clear().expect("failed to clear screen"); //clear terminal
    let input_file_path = String::from("example_inputs/simple_input.csv");
    
    //train SOM
    let som_output: DMatrix<DVector<f64>> = simple_som_real_field(
        input_file_path, 
        (9,9), 
        None, 
        None, 
        None);

    //print SOM to console
    let precision: usize = 1;
    print_matrix_of_vectors(&som_output, &precision);
    
    //create and export visualization (SVG file)
    let image_path : &str = concat!(env!("CARGO_MANIFEST_DIR"), "/image_outputs/image.svg");

    //create folder if it does not exist
    match fs::create_dir("image_outputs") {
        Ok(_) => println!("Image folder (/image_outputs) created successfully"),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
        println!("Image folder (/image_outputs) already exists")
        },
        Err(e) => println!("Error creating image folder (/image_outputs): {}", e),
    }

    match basic_visualization(&som_output, 0.7, 0.9, image_path) {
        Ok(()) => println!("Image Outputted"),
        Err(e) => eprintln!("Error: {}", e)
    } 

    //exit
    println!("exit");
}
