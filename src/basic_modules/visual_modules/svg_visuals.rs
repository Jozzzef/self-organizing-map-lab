use nalgebra::{DMatrix, DVector};
use plotters::prelude::*;
use std::cmp::max as tuple_max;
use rand::Rng;

// SVG VISUALS

pub fn vector_magnitude_for_hue(som_map: &DMatrix<DVector<f64>>) -> DMatrix<isize> {
    let mut hue_matrix: DMatrix<isize> = DMatrix::zeros(som_map.nrows(), som_map.ncols());

    for i in 0..som_map.nrows() {
        for j in 0..som_map.ncols() {
            let v: Vec<f64> = som_map[(i, j)].clone().as_slice().to_vec();
            let (_, h_in_tup) = v
                .into_iter()
                .enumerate()
                .map(|(i, x)| (i as f64, x))
                .reduce(|(_, acc), (i_x, x)| return (i_x, acc + (x * 10_f64.powf(i_x))))
                .unwrap();
            let h = h_in_tup.round() as isize;
            hue_matrix[(i, j)] = h;
        }
    }

    hue_matrix = hue_matrix.add_scalar(hue_matrix.min()); //remove any potential negatives
    let max_val_hue_matrix = hue_matrix.max();
    let min_val_hue_matrix = hue_matrix.min();
    let max_dim = tuple_max(hue_matrix.nrows(), hue_matrix.ncols()) as isize;
    let partitions_tile_size = (max_val_hue_matrix - min_val_hue_matrix) / (max_dim as isize) + 1;
    let mut hue_partition_for_260 = vec![Vec::new(); max_dim as usize];
    let mut hue_partition_for_260_without_index = vec![Vec::new(); max_dim as usize];

    for i in 0..hue_matrix.nrows() {
        for j in 0..hue_matrix.ncols() {
            let hue: isize = hue_matrix[(i, j)].clone();
            for k in 0..max_dim {
                let min_k = min_val_hue_matrix + k * partitions_tile_size;
                let max_k = min_val_hue_matrix + (k + 1) * partitions_tile_size;
                if min_k <= hue && hue < max_k {
                    hue_partition_for_260[k as usize].push(((i, j), hue));
                    hue_partition_for_260_without_index[k as usize].push(hue);
                }
            }
        }
    }

    for i in 0..max_dim {
        let w = hue_partition_for_260_without_index[i as usize].clone();
        let input_max = *w.iter().max().unwrap_or(&0) as f64;
        let input_min = *w.iter().min().unwrap_or(&0) as f64;
        let output_max = ((i + 1) * 260 / max_dim) as f64;
        let output_min = if i == 0 {
            0.0
        } else {
            (i * 260 / max_dim) as f64
        };
        for (index, hue) in hue_partition_for_260[i as usize].clone() {
            let hue = hue as f64;
            let ratio: f64 = (hue - input_min) / (input_max - input_min);
            let result: f64 = output_min + ratio * (output_max - output_min);
            hue_matrix[index] = result.round() as isize;
        }
    }

    return hue_matrix;
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
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 1.0 / 2.0 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

fn hsl_to_rgb(mut h: f64, s: f64, l: f64) -> (f64, f64, f64) {
    // 1. Normalize hue to 0-1 range
    h = h / 260.0;

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

    let r = (hue_to_rgb(p, q, h + 1.0 / 3.0) * 255.0).round();
    let g = (hue_to_rgb(p, q, h) * 255.0).round();
    let b = (hue_to_rgb(p, q, h - 1.0 / 3.0) * 255.0).round();
    return (r, g, b);
}

pub fn basic_visualization(
    som_map: &DMatrix<DVector<f64>>,
    saturation: f64,
    lightness: f64,
    image_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let max_dim = tuple_max(som_map.nrows(), som_map.ncols());
    let y_size: i32 = 800;
    let x_size: i32 = 1000;

    let canvas_tuple = SVGBackend::new(image_path, (x_size as u32, y_size as u32))
        .into_drawing_area()
        .split_horizontally(700);
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
            .flat_map(|(l, y)| l.iter().zip(0..).map(move |(v, x)| (x, y, v)))
            .map(|(x, y, v)| {
                let v_float = *v as f64;
                let (r, g, b) = hsl_to_rgb(v_float, saturation, lightness);
                let style = ShapeStyle {
                    color: RGBAColor(r as u8, g as u8, b as u8, 1.0),
                    filled: true,
                    stroke_width: 1,
                };
                let mut rect = Rectangle::new([(x, y), (x + 1, y + 1)], style);
                rect.set_margin(1, 1, 1, 1);
                return rect;
            }),
    )?;

    // for legend split
    let mut chart_for_legend = ChartBuilder::on(&canvas_tuple.1)
        .caption("Legend For Vectors", ("sans-serif", 30))
        .margin_top(40) //in pixels
        .y_label_area_size(40)
        .x_label_area_size(40)
        .build_cartesian_2d(0_f32..2_f32, 0_f32..6_f32)?;


    //get random vectors to sample from and put in a vector of tuples
    let mut legend_tuples : Vec<(String, (f32, f32), f64)> = vec![];

    let num_of_legend_vals = 6; 

    for i in 0..num_of_legend_vals{
        let matrix_i_start = som_map.nrows() / num_of_legend_vals * i;
        let matrix_j_start = som_map.ncols() / num_of_legend_vals * i;
        let matrix_i_end = som_map.nrows() / num_of_legend_vals * (i+1);
        let matrix_j_end = som_map.ncols() /num_of_legend_vals * (i+1);
        let random_i = rand::thread_rng().gen_range(matrix_i_start..matrix_i_end); 
        let random_j = rand::thread_rng().gen_range(matrix_j_start..matrix_j_end);

        //custom formatting for stringed vector
        let s = som_map.index((random_i, random_j))
            .iter()
            .map(|x| format!("{:.2}", x))
            .collect::<Vec<_>>()
            .join(", \n");
        let s = ["[", &s, "] => "].join("");

        legend_tuples.push((
            s,
            (0.0, 0.5 + i as f32),
            *hue_matrix.index((random_i, random_j)) as f64
        ));
    }

    chart_for_legend.draw_series(
       legend_tuples.iter().map(|t| {
        return Text::new(
            t.0.clone(),
            t.1,
            ("sans-serif", 12).into_font().color(&BLACK),
        );
    }))?;

    chart_for_legend.draw_series(
        legend_tuples.iter().map(|t| {
            let (r, g, b) = hsl_to_rgb(t.2, saturation, lightness);
            let style = ShapeStyle {
                color: RGBAColor(r as u8, g as u8, b as u8, 1.0),
                filled: true,
                stroke_width: 3,
            };
            return Rectangle::new(
                [(1.1, t.1.1 - 0.3), (1.8, t.1.1 + 0.3)],
                style
            );
     }))?;

    canvas_tuple.0.present()?;
    canvas_tuple.1.present()?;
    return Ok(());
}
