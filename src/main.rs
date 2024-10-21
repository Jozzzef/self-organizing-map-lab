//declare modules
mod basic_modules;

//path statements for modules
use basic_modules::simple_som;

//dependencies
use nalgebra as nalg;

fn main() {
    let vari = 3;
    println!("Hello, world!");
    let som_output: nalg::DMatrix<f64> = simple_som(1, (3,3), String::from("example_inputs/simple_input.csv"));
    println!("{som_output}");
    println!("exit");
}
