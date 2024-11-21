use plotters::{data::float, prelude::*};
use nalgebra::{DMatrix, DVector};
use std::cmp::max as tuple_max;

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
    let max_val_hue_matrix = hue_matrix.max();
    let min_val_hue_matrix = hue_matrix.min();
    let max_dim = tuple_max(hue_matrix.nrows() , hue_matrix.ncols()) as isize;
    let partitions_tile_size = (max_val_hue_matrix - min_val_hue_matrix) / (max_dim as isize) + 1;
    let mut hue_partition_for_360 = vec![Vec::new(); max_dim as usize];
    let mut hue_partition_for_360_without_index = vec![Vec::new(); max_dim as usize];

    for i in 0..hue_matrix.nrows() {
        for j in 0..hue_matrix.ncols() {
            let hue: isize = hue_matrix[(i, j)].clone();
            for k in 0..max_dim {
                let min_k = min_val_hue_matrix + k * partitions_tile_size;
                let max_k = min_val_hue_matrix + (k + 1)  * partitions_tile_size;
                if min_k <= hue && hue < max_k {
                    hue_partition_for_360[k as usize].push( ((i,j), hue) );
                    hue_partition_for_360_without_index[k as usize].push(hue);
                }
            }
        }
    }

    for i in 0..max_dim {
        let w = hue_partition_for_360_without_index[i as usize].clone();
        let input_max = *w.iter().max().unwrap() as f64;
        let input_min = *w.iter().min().unwrap() as f64;
        let output_max = ((i+1) * 360 / max_dim) as f64;
        let output_min = if i == 0 {0.0} else {(i * 360 / max_dim) as f64};
        for (index, hue) in hue_partition_for_360[i as usize].clone() { 
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
    
    let max_dim = tuple_max(som_map.nrows(), som_map.ncols());

    let canvas = SVGBackend::new(image_path, (800,800)).into_drawing_area();
    canvas.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&canvas)
        .caption("Basic SOM Chart", ("sans-serif", 30))
        .margin(5) //in pixels
        .build_cartesian_2d(0..max_dim, 0..max_dim)?;
    
    chart
        .configure_mesh() //configure mesh/grid
            .x_labels(max_dim / 2) //label each dimension? param is the number of labels in the dimension
            .y_labels(max_dim / 2)
            .draw()?;

    let hue_matrix = vector_magnitude_for_hue(&som_map);
    let vec_hue_matrix = dmatrix_to_vec(&hue_matrix);
    // for debugging
    

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
                let v_float = *v as f64;
                println!("{x}, {y}, {v_float}");
                let hsl_colors = HSLColor(v_float, saturation, lightness).filled();
                println!("{:#?}", hsl_colors);
                let mut rect = Rectangle::new(
                    [(x,y), (x+1, y+1)],
                    HSLColor(v_float, saturation, lightness).filled()
                );
                rect.set_margin(1, 1, 1, 1);
                return rect
            }   
        ))?;

    canvas.present()?;
    return Ok(());
}
