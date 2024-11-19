use plotters::prelude::*;
use nalgebra::{DMatrix, DVector};


fn vector_magnitude_for_hue(som_map: DMatrix<DVector<f64>>) -> DMatrix<i32> {
    
    return 1
}

fn basic_visualization(som_map: DMatrix<DVector<f64>>) -> Result<(), Box<dyn std::error::Error>> {
    let canvas = SVGBackend::new("plot.svg", (800,800)).into_drawing_area();
    canvas.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&canvas)
        .caption("Basic SOM Chart", ("sans-serif", 30))
        .margin(5) //in pixels
        .build_cartesian_2d(0..100, 0..100)?;
    
    chart
        .configure_mesh() //configure mesh/grid
            .x_labels(10) //label each dimension? param is the number of labels in the dimension
            .y_labels(10)
            .draw()?;

    let v: Vec<i32> = (0..10).collect();
    let w: Vec<i32> = (1..10).collect();
    let som_map_iter = v.into_iter().zip(w.into_iter());
    chart.draw_series(
        som_map_iter.map( |(x,y)| { 
            Rectangle::new(
                [(x,y), (x+1, y+1)],
                HSLColor(130.0, 0.9, 0.7).filled()
            )
        }   
        ))?;

    canvas.present()?;
    return Ok(());
}
