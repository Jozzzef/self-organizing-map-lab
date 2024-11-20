use plotters::prelude::*;
use nalgebra::{DMatrix, DVector};


pub fn vector_magnitude_for_hue(som_map: &DMatrix<DVector<f64>>) -> DMatrix<isize> {
    let mut hue_matrix : DMatrix<isize> = DMatrix::zeros(som_map.nrows(), som_map.ncols());

    for i in 0..som_map.nrows() {
        for j in 0..som_map.ncols() {
            let v: Vec<f64> = som_map[(i, j)].clone().as_slice().to_vec();
            let (_, h_in_tup) = v.into_iter().enumerate()
                .map(|(i, x)| (i as f64, x))
                .reduce(|(_, acc), (i_x, x)| { 
                    return (i_x, acc + (x * 10_f64.powf(i_x)))
                }).unwrap();
            let h = h_in_tup.round() as isize;
            hue_matrix[(i,j)] = h;
        }
    }

    hue_matrix = hue_matrix.add_scalar(hue_matrix.min()); //remove any potential negatives
    let hue_max = hue_matrix.max();
    let max_digits = hue_max.abs().to_string().len();
    let mut hue_partition_for_360 = vec![Vec::new(); max_digits];
    let mut hue_partition_for_360_without_index = vec![Vec::new(); max_digits];

    for i in 0..hue_matrix.nrows() {
        for j in 0..hue_matrix.ncols() {
            let hue: isize = hue_matrix[(i, j)].clone();
            let num_digits = hue.abs().to_string().len();
            hue_partition_for_360[num_digits-1].push( ((i,j), hue) );
            hue_partition_for_360_without_index[num_digits-1].push(hue);
        }
    }

    for i in 0..max_digits {
        let w = hue_partition_for_360_without_index[i].clone();
        let input_max = *w.iter().max().unwrap() as f64;
        let input_min = *w.iter().min().unwrap() as f64;
        let output_max = ((i+1) * 360 / max_digits) as f64;
        let output_min = if i == 0 {0.0} else {(i * 360 / max_digits) as f64};
        for (index, hue) in hue_partition_for_360[i].clone() { 
            let hue = hue as f64;
            let ratio: f64 = (hue - input_min) / (input_max - input_min);
            let result: f64 = output_min + ratio * (output_max - output_min);
            hue_matrix[index] = result.round() as isize;
        }
    }

    return hue_matrix 
}

pub fn dmatrix_to_vec<T: Clone>(matrix: &DMatrix<T>) -> Vec<Vec<T>> {
    let (rows, cols) = matrix.shape();
    let mut result = Vec::with_capacity(rows);
    
    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            row.push(matrix[(i, j)].clone());
        }
        result.push(row);
    }
    
    result
}

pub fn basic_visualization(som_map: &DMatrix<DVector<f64>>, 
    saturation: f64, 
    lightness: f64,
    image_path: &str) 
    -> Result<(), Box<dyn std::error::Error>> {
    
    let canvas = SVGBackend::new(image_path, (800,800)).into_drawing_area();
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

    let hue_matrix = vector_magnitude_for_hue(&som_map);
    let vec_hue_matrix = dmatrix_to_vec(&hue_matrix);
    chart.draw_series(
        vec_hue_matrix
            .iter()
            .zip(0..)
            .flat_map(|(l, y)| {
                l
                    .iter()
                    .zip(0..)
                    .map(move |(v, x)| (x, y, v))}
            )
            .map(|(x, y, v)| {
                Rectangle::new(
                    [(x,y), (x+1, y+1)],
                    HSLColor(*v as f64, saturation, lightness).filled()
            )
        }   
        ))?;

    canvas.present()?;
    return Ok(());
}
