use plotters::prelude::*;
use nalgebra::{coordinates, DMatrix, DVector};
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


fn hue_to_rgb(p: f64, q: f64, mut t: f64) -> f64 {
    if t < 0.0 { t += 1.0; }
    if t > 1.0 { t -= 1.0; }
    if t < 1.0/6.0 { return p + (q - p) * 6.0 * t; }
    if t < 1.0/2.0 { return q; }
    if t < 2.0/3.0 { return p + (q - p) * (2.0/3.0 - t) * 6.0; }
    p
}

fn hsl_to_rgb(mut h: f64, s: f64, l: f64) -> (f64, f64, f64) {
    // 1. Normalize hue to 0-1 range
    h = h / 360.0;

    // Handle grayscale case
    if s == 0.0 {
        let gray = (l * 255.0).round();
        return (gray, gray, gray);
    }

    // 2. Calculate q and p
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - (l * s)
    };
    let p = 2.0 * l - q;

    let r = (hue_to_rgb(p, q, h + 1.0/3.0) * 255.0).round();
    let g = (hue_to_rgb(p, q, h) * 255.0).round();
    let b = (hue_to_rgb(p, q, h - 1.0/3.0) * 255.0).round();
    return (r, g, b)
}


pub fn basic_visualization(som_map: &DMatrix<DVector<f64>>, 
    saturation: f64, 
    lightness: f64,
    image_path: &str) 
    -> Result<(), Box<dyn std::error::Error>> {
    
    let max_dim = tuple_max(som_map.nrows(), som_map.ncols());

    let canvas_tuple = SVGBackend::new(image_path, (1000,800)).into_drawing_area().split_horizontally(700);
    canvas_tuple.0.fill(&WHITE)?;
    canvas_tuple.1.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&canvas_tuple.0)
        .caption("Basic SOM Chart", ("sans-serif", 30))
        .margin(5) //in pixels
        .margin_bottom(40)
        .margin_right(40)
        .y_label_area_size(35)
        .x_label_area_size(35)
        .build_cartesian_2d(0..max_dim, 0..max_dim)?;
    
    chart
        .configure_mesh() //configure mesh/grid
        .x_labels(max_dim / 2) //label each dimension? param is the number of labels in the dimension
        .y_labels(max_dim / 2)
        .disable_x_mesh()
        .disable_y_mesh()
        .label_style(("sans-serif", 20))
        .draw()?;

    let hue_matrix = vector_magnitude_for_hue(&som_map);
    let vec_hue_matrix = dmatrix_to_vec(&hue_matrix);
    // for debugging
   println!("{hue_matrix}"); 

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
                let (r,g,b) = hsl_to_rgb(v_float, saturation, lightness);
                let style = ShapeStyle {
                    color: RGBAColor(r as u8, g as u8, b as u8, 1.0),
                    filled: true,
                    stroke_width: 1,
                };                
                let mut rect = Rectangle::new(
                    [(x,y), (x+1, y+1)],
                    style
                );
                rect.set_margin(1, 1, 1, 1);
                return rect
            }   
        ))?;

    // for legend split
    let mut chart_for_legend = ChartBuilder::on(&canvas_tuple.1)
        .caption("The Diagonal Of SOM Matrix", ("sans-serif", 30))
        .margin(5) //in pixels
        .margin_bottom(40)
        .margin_right(40)
        .y_label_area_size(35)
        .top_x_label_area_size(35)
        .build_cartesian_2d(0..1, 0..max_dim)?;

    let som_map_to_vec = dmatrix_to_vec(som_map);
    chart_for_legend.draw_series(
        som_map_to_vec 
            .iter()
            .zip(0..)
            .flat_map(|(l, y)| {
                l
                    .iter()
                    .zip(0..)
                    .map(move |(vec, x)| (x, y, vec))}
            )
            .filter(|(x,y,vec)| x == y)
            .map(|(x, y, vec)| {
                let v_float = hue_matrix[(x,y)] as f64;
                let (r,g,b) = hsl_to_rgb(v_float, saturation, lightness);
                let style = ShapeStyle {
                    color: RGBAColor(r as u8, g as u8, b as u8, 1.0),
                    filled: true,
                    stroke_width: 1,
                };                
                let mut rect = Rectangle::new(
                    [(0,y), (1, y+1)],
                    style
                );
                rect.set_margin(1, 1, 1, 1);
                return rect 
                //Text::new(format!("vec @ ({x},{y}): {:?}", vec), (x,y), ("sans-serif", 10));
            }   
        ))?;
        
    canvas_tuple.0.present()?;
    canvas_tuple.1.present()?;
    return Ok(());
}
