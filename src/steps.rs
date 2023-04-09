use opencv::{
    core::{self, Mat, Point, Rect},
    prelude::*,
    highgui,
    imgproc,
};
use crate::{Res, WIN, utils};

pub fn flip(src: &Mat) -> Res<Mat> {
    let mut dst = Mat::default();
    core::flip(&src, &mut dst, 1)?;
    Ok(dst)
}

pub struct FilterOuput {
    display: Mat,
    mask: Mat,
}
pub fn hsv_filter(src: &Mat, s: [i32; 3], e: [i32; 3]) -> Res<FilterOuput> {
    let mut mask = Mat::default();

    core::in_range(
        &src,
        &Mat::from_slice(&s)?,
        &Mat::from_slice(&e)?,
        &mut mask
    )?;

    let mut display = Mat::default();
    core::bitwise_and(&src, &src, &mut display, &mask)?;
    let display = convert(&display, imgproc::COLOR_HSV2BGR)?;

    Ok(FilterOuput { display, mask })
}

pub enum BitwiseOperation {
    And, Xor, Or,
}

pub fn combine_binaries<const N: usize>(masks: [&Mat; N], operation: BitwiseOperation) -> Res<Mat> {
    match N {
        0 => return Err("Nothing provided".into()),
        2 => return Ok(core::or_mat_mat(masks[0], masks[1])?.to_mat()?),
        _ => (),
    }

    let mut dst = masks[0].clone();
    let noarr = core::no_array();
    for m in masks.iter().skip(1) {
        match operation {
            BitwiseOperation::Xor => core::bitwise_xor(&dst.clone(), m, &mut dst, &noarr)?,
            BitwiseOperation::And => core::bitwise_and(&dst.clone(), m, &mut dst, &noarr)?,
            BitwiseOperation::Or  => core::bitwise_or( &dst.clone(), m, &mut dst, &noarr)?,
        }
    }

    Ok(dst)
}

pub fn convert(src: &Mat, code: i32) -> Res<Mat> {
    let mut dst = Mat::default();
    imgproc::cvt_color(&src, &mut dst, code, 0)?;
    Ok(dst)
}

pub fn blur(src: &Mat, bw: i32, bh: i32) -> Res<Mat> {
    let mut dst = Mat::default();

    imgproc::blur(src, &mut dst,
        core::Size::new(bw, bh),
        Point::new(-1, -1),
        core::BORDER_DEFAULT
    )?;

    Ok(dst)
}

pub struct BlobInfo {
    area: i32,
    centroid: Point,
    
    bbox: Rect,
}
impl std::fmt::Display for BlobInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})x{}[{}, {}, {}, {}]",
            self.centroid.x, self.centroid.y,
            self.area,
            self.bbox.x, self.bbox.y, self.bbox.width, self.bbox.height
        )
    }
}

pub struct BlobbingRet {
    blobbed: Mat,
    info: Vec<BlobInfo>,
}
pub fn remove_blobs(image: &Mat, width: i32, height: i32, min_size: i32) -> Res<BlobbingRet> {
    let mut labels = Mat::default();
    let mut stats = Mat::default();
    let mut centroids = Mat::default();

    imgproc::connected_components_with_stats(
        image, &mut labels, &mut stats, &mut centroids,
        8, core::CV_32S
    )?;

    let mut dst = Mat::new_nd_with_default(&[height, width], core::CV_8UC1, 0.into())?;

    let mut info = Vec::with_capacity(centroids.rows() as usize - 1);
    let mut mask = Mat::new_nd_with_default(&[height, width], core::CV_8UC1, 0.into())?;

    for blob in 1..stats.rows() {
        let area = *stats.at_2d(blob, imgproc::CC_STAT_AREA)?;
        if area < min_size {
            continue;
        }
        info.push(
            BlobInfo {
                area,
                centroid: core::Point::new(
                    *centroids.at_2d::<f64>(blob, 0)? as i32,
                    *centroids.at_2d::<f64>(blob, 1)? as i32,
                ),
                bbox: Rect::new(
                    *stats.at_2d(blob, imgproc::CC_STAT_LEFT)?,
                    *stats.at_2d(blob, imgproc::CC_STAT_TOP)?,
                    *stats.at_2d(blob, imgproc::CC_STAT_WIDTH)?,
                    *stats.at_2d(blob, imgproc::CC_STAT_HEIGHT)?
                )
            }
        );

        let with = Mat::new_nd_with_default(&[1], core::CV_8UC1, blob.into())?;
        core::compare(&labels, &with, &mut mask, core::CMP_EQ)?;

        dst = combine_binaries([
            &dst, &mask
        ], BitwiseOperation::Or)?;
    }

    info.sort_by(|a, b| a.area.cmp(&b.area).reverse());

    Ok(BlobbingRet {
        blobbed: dst,
        info,
    })
}

pub fn close(src: &Mat, kernel_size: i32, iterations: i32) -> Res<Mat> {
    let mut dst = Mat::default();

    imgproc::morphology_ex(
        src,
        &mut dst,
        imgproc::MORPH_CLOSE,
        &Mat::ones(kernel_size, kernel_size, core::CV_8U)?,
        Point::new(-1, -1),
        iterations,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;

    Ok(dst)
}

static TR_MIN_SIZE: &str = "min pixels";

static TR_KERNEL_SIZE: &str = "kernel size";
static TR_ITERATIONS: &str = "iterations";

static TR_W_AREAS: &str       = "W           areas";
static TR_W_XPOS: &str        = "W            xpos";
static TR_W_Y_TO_HEIGHT: &str = "W y to height";
static TR_W_MAKE_SQUARE: &str = "W          square";

static TR_SIM_CAP: &str = "similarity cap";

const C_W_MAX: i32 = 50;

static mut COLORS: Option<Vec<([i32; 3], [i32; 3])>> = None;
pub fn init() -> Res<()> {
    // trackbars
    {
        use highgui::{create_trackbar as create, set_trackbar_pos as set};

        create(TR_MIN_SIZE, WIN, None, 1000, None)?;
        set(TR_MIN_SIZE, WIN, 30)?;

        create(TR_KERNEL_SIZE, WIN, None, 20, None)?;
        create(TR_ITERATIONS, WIN, None, 10, None)?;
        set(TR_KERNEL_SIZE, WIN, 5)?;
        set(TR_ITERATIONS, WIN, 1)?;

        create(TR_W_AREAS, WIN, None, 100, None)?;
        set(TR_W_AREAS, WIN, 50)?;
        create(TR_W_XPOS, WIN, None, 100, None)?;
        set(TR_W_XPOS, WIN, 50)?;
        create(TR_W_Y_TO_HEIGHT, WIN, None, 100, None)?;
        set(TR_W_Y_TO_HEIGHT, WIN, 50)?;
        create(TR_W_MAKE_SQUARE, WIN, None, 100, None)?;
        set(TR_W_MAKE_SQUARE, WIN, 50)?;

        create(TR_SIM_CAP, WIN, None, 100, None)?;
        set(TR_SIM_CAP, WIN, 50)?;
    }

    let read_colors = |fname: &str| -> Res<([i32; 3], [i32; 3])> {
        let s = std::fs::read_to_string(fname)?;
        let s: Vec<&str> = s.split(',').collect();

        Ok((
            [s[0].parse()?, s[1].parse()?, s[2].parse()?],
            [s[3].parse()?, s[4].parse()?, s[5].parse()?],
        ))
    };

    unsafe {
        COLORS = Some(vec![
            read_colors("test/color0.csv")?,
            read_colors("test/color1.csv")?,
            read_colors("test/color2.csv")?,
        ]);
    }

    Ok(())
}

pub fn get_steps(src: &Mat, width: i32, height: i32) -> Res<Vec<(&'static str, Vec<Mat>)>> {
    let flipped = flip(src)?;

    // let blurred = blur(&flipped, 10, 10)?;
    // ret.push(("Blurred", vec![blurred.clone()]));
    
    let hsv = convert(&flipped, imgproc::COLOR_BGR2HSV)?;

    let Some(cols) = (unsafe { &COLORS }) else {
        return Err("No colors?".into());
    };

    let (filter0, filter1) = (
        hsv_filter(&hsv, cols[0].0, cols[0].1)?,
        hsv_filter(&hsv, cols[1].0, cols[1].1)?,
    );

    let close_kernel_size = highgui::get_trackbar_pos(TR_KERNEL_SIZE, WIN)?;
    let close_iterations = highgui::get_trackbar_pos(TR_ITERATIONS, WIN)?;
    let (closed0, closed1) = (
        close(&filter0.mask, close_kernel_size, close_iterations)?,
        close(&filter1.mask, close_kernel_size, close_iterations)?,
    );

    let min_size = highgui::get_trackbar_pos(TR_MIN_SIZE, WIN)?;
    let (blobbed0, blobbed1) = (
        remove_blobs(&closed0, width, height, min_size)?,
        remove_blobs(&closed1, width, height, min_size)?
    );

    let mut to_draw = flipped.clone();

    let mut cand: Option<(f64, (&BlobInfo, &BlobInfo))> = None;
    for mut b0 in &blobbed0.info {
        for mut b1 in &blobbed1.info {
            if b0.centroid.y > b1.centroid.y {
                (b1, b0) = (b0, b1);
            }

            let (similarity, components) = get_score(b0, b1)?;
            let cap = highgui::get_trackbar_pos(TR_SIM_CAP, WIN)? as f64 * 0.01;
            if similarity < cap {
                continue;
            }
            if cand.is_none() || cand.unwrap().0 < similarity {
                cand = Some((similarity, (b0, b1)));
            }

            let c = (similarity - cap) / (1. - cap);
            let color = utils::color(255.0 * (1.0 - c), 255.0 * c, 0.);

            let scale = b0.bbox.width as f64 / width as f64;

            for b in [b0, b1] {
                imgproc::rectangle(
                    &mut to_draw,
                    b.bbox,
                    color,
                    (scale * 10. * 2.5) as i32,
                    imgproc::LINE_8,
                    0,
                )?;
            }

            imgproc::put_text(
                &mut to_draw,
                &format!("{:.0}", similarity * 100.),
                b0.bbox.tl(),
                imgproc::FONT_HERSHEY_SIMPLEX,
                scale * 10.,
                color,
                (scale * 10. * 2.5) as i32,
                imgproc::LINE_8,
                false
            )?;

            let rad = scale * 10. * 7.5;
            let clen = components.len();
            for (i, (c, p)) in components.into_iter().enumerate() {
                let center = Point::new(
                    b0.bbox.x + b0.bbox.width / 2
                        + (i - clen / 2) as i32
                            * ((rad * 2.) as i32 + 5),
                    b0.bbox.y
                );
                imgproc::circle(
                    &mut to_draw,
                    center,
                    rad as i32,
                    utils::color(255.0 * (1.0 - c), 255.0 * c, 0.),
                    imgproc::FILLED,
                    imgproc::LINE_8,
                    0,
                )?;
                if p {
                    let thickness = scale * 10. * 2.;
                    imgproc::circle(
                        &mut to_draw,
                        center,
                        (rad + thickness) as i32,
                        utils::color(255.0, 0., 0.),
                        thickness as i32,
                        imgproc::LINE_8,
                        0,
                    )?;
                }
            }
        }
    }

    let mut ret = vec![
        ("Flipped", vec![flipped]),
        ("HSV Filter", vec![
            filter0.display,
            filter1.display,
        ]),
        ("Closing", vec![
            convert(&closed0, imgproc::COLOR_GRAY2BGR)?,
            convert(&closed1, imgproc::COLOR_GRAY2BGR)?,
        ]),
        ("Blobbing", vec![
            convert(&blobbed0.blobbed, imgproc::COLOR_GRAY2BGR)?,
            convert(&blobbed1.blobbed, imgproc::COLOR_GRAY2BGR)?,
        ]),
        ("Result", vec![to_draw]),
    ];
    let Some((_, (c0, c1))) = cand else {
        return Ok(ret);
    };

    let final_pos = Point::new(
        (c0.centroid.x + c1.centroid.x) / 2,
        (c0.centroid.y + c1.centroid.y) / 2,
    );

    let ind = ret.len() - 1;
    let to_draw = &mut (ret[ind].1[0]);

    imgproc::draw_marker(
        to_draw,
        final_pos,
        utils::color(255.0, 0., 0.),
        imgproc::MARKER_TILTED_CROSS,
        50,
        5,
        imgproc::LINE_8,
    )?;

    Ok(ret)
}

const SINGLE_COMPONENT_PENALTY_THRESHOLD: f64 = 0.3;
const SINGLE_COMPONENT_PENALTY: f64 = 0.75;
fn get_score(b0: &BlobInfo, b1: &BlobInfo) -> Res<(f64, Vec<(f64, bool)>)> {
    use utils::sim_perc as sim;

    let areas = sim(b0.area, b1.area);

    let bboxes_overlap =
        (b0.bbox.x <= (b1.bbox.x + b1.bbox.width)) &&
        ((b0.bbox.x + b0.bbox.width) >= b1.bbox.x);

    let xpos = if bboxes_overlap {
        sim(b0.centroid.x, b1.centroid.x) 
    } else {
        0.
    };

    let y_to_height = sim(
        (b0.centroid.y - b1.centroid.y).abs() as f64,
        (b0.bbox.height + b1.bbox.height) as f64 * 0.5,
    );
    let make_square = sim(
        ((b0.bbox.width as f64 + b0.bbox.width as f64) * 0.5) /
        (b1.bbox.height as f64 + b1.bbox.height as f64),
        1.,
    );

    let values = [areas, xpos, y_to_height, make_square];
    let weights = [
        highgui::get_trackbar_pos(TR_W_AREAS,       WIN)?,
        highgui::get_trackbar_pos(TR_W_XPOS,        WIN)?,
        highgui::get_trackbar_pos(TR_W_Y_TO_HEIGHT, WIN)?,
        highgui::get_trackbar_pos(TR_W_MAKE_SQUARE, WIN)?,
    ];
    let weights = weights.map(|w| w as f64 / C_W_MAX as f64);

    let (mut sum, mut weigth_sum, mut total_penalty) = (0., 0., 1.);

    let mut got_penalties = [false; 4];
    for ((component_value, component_weight), penalty) in values.into_iter().zip(weights).zip(&mut got_penalties) {
        let weighted_value = component_value * component_weight;

        sum += weighted_value;
        weigth_sum += component_weight;

        if weighted_value < SINGLE_COMPONENT_PENALTY_THRESHOLD {
            *penalty = true;
            total_penalty *= SINGLE_COMPONENT_PENALTY;
        }
    }

    Ok((
        (sum / weigth_sum) * total_penalty,
        values.into_iter().zip(got_penalties).collect(),
    ))
}
