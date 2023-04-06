use opencv::core::Scalar;

pub fn color(r: f64, g: f64, b: f64) -> Scalar {
    Scalar::new(b, g, r, 255.)
}

pub fn same_magnitude<T>(mut i: T, mut j: T, mag: f64) -> bool
where T: PartialOrd + Into<f64> + Copy {
    if i > j {
        (i, j) = (j, i);
    }
    let r = Into::<f64>::into(i) / Into::<f64>::into(j);
    (1. - r) < mag
}

pub fn sim_perc<T>(i: T, j: T) -> f64
where T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + Into<f64> + Copy {
    let diff = Into::<f64>::into(i - j).abs();
    let avg = Into::<f64>::into(i + j) * 0.5;
    1. - (diff / avg)
}
