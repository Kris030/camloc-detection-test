
use opencv::{
    core::{self, Mat, Point, Scalar, Rect},
    prelude::*,
    highgui,
    imgproc,
};
use crate::{Res, WIN};

fn flip(src: &Mat) -> Res<Mat> {
    let mut dst = Mat::default();
    core::flip(&src, &mut dst, 1)?;
    Ok(dst)
}

struct FilterOuput {
    display: Mat,
    mask: Mat,
}
fn hsv_filter(src: &Mat, s: [i32; 3], e: [i32; 3]) -> Res<FilterOuput> {
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

#[allow(unused)]
enum BitwiseOperation {
    And, Xor, Or,
}

fn combine_binaries<const N: usize>(masks: [&Mat; N], operation: BitwiseOperation) -> Res<Mat> {
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

fn convert(src: &Mat, code: i32) -> Res<Mat> {
    let mut dst = Mat::default();
    imgproc::cvt_color(&src, &mut dst, code, 0)?;
    Ok(dst)
}

#[allow(unused)]
fn blur(src: &Mat, bw: i32, bh: i32) -> Res<Mat> {
    let mut dst = Mat::default();

    imgproc::blur(src, &mut dst,
        core::Size::new(bw, bh),
        Point::new(-1, -1),
        core::BORDER_DEFAULT
    )?;

    Ok(dst)
}

#[allow(unused)]
#[derive(Debug)]
struct BlobInfo {
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

#[allow(unused)]
struct BlobbingRet {
    blobbed: Mat,
    info: Vec<BlobInfo>,
}

fn remove_small_blobs(image: &Mat, width: i32, height: i32, min_size: i32) -> Res<BlobbingRet> {
    let mut labels = Mat::default();
    let mut stats = Mat::default();
    let mut centroids = Mat::default();

    imgproc::connected_components_with_stats(
        image, &mut labels, &mut stats, &mut centroids,
        8, core::CV_32S
    )?;

    let mut dst = Mat::new_nd_with_default(&[height, width], core::CV_8UC1, 0.into())?;
    
    // let mut filtered = 0;
    // let mut filtered_pixels = 0;
    
    let mut info = Vec::with_capacity(centroids.rows() as usize - 1);
    let mut mask = Mat::new_nd_with_default(&[height, width], core::CV_8UC1, 0.into())?;

    for blob in 1..stats.rows() {
        let area = *stats.at_2d(blob, imgproc::CC_STAT_AREA)?;
        if area < min_size {
            // filtered += 1;
            // filtered_pixels += area;
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

    // let all_rows = stats.rows() - 1;
    // eprintln!("filtered {filtered} ({filtered_pixels} pixels) of {} ({:.2}%)", all_rows, filtered as f64 / all_rows as f64 * 100.);

    info.sort_by(|a, b| a.area.cmp(&b.area).reverse());

    Ok(BlobbingRet {
        blobbed: dst,
        info,
    })
}

static TR: &str = "min_pixels";
pub fn init() -> Res<()> {
    highgui::create_trackbar(TR, WIN, None, 10_000, None)?;

    let read_colors = |fname: &str| -> Res<([i32; 3], [i32; 3])> {
        let s = std::fs::read_to_string(fname)?;
        let s: Vec<&str> = s.split(',').collect();

        Ok((
            [s[0].parse()?, s[1].parse()?, s[2].parse()?],
            [s[3].parse()?, s[4].parse()?, s[5].parse()?],
        ))
    };

    unsafe {
        colors = Some(vec![
            read_colors("test/color0.csv")?,
            read_colors("test/color1.csv")?,
            read_colors("test/color2.csv")?,
        ]);
    }

    Ok(())
}

#[allow(non_upper_case_globals)]
static mut colors: Option<Vec<([i32; 3], [i32; 3])>> = None;

pub fn get_steps(src: &Mat, width: i32, height: i32) -> Res<Vec<Vec<Mat>>> {
    let mut ret = vec![];

    let flipped = flip(src)?;
    ret.push(vec![flipped.clone()]);
    // let blurred = blur(&flipped, 10, 10)?;
    // ret.push(vec![blurred.clone()]);
    
    let hsv = convert(&flipped, imgproc::COLOR_BGR2HSV)?;

    let Some(cols) = (unsafe { &colors }) else {
        return Err("No colors?".into());
    };

    let (
        filter0,
        filter1,
    ) = (
        hsv_filter(&hsv, cols[0].0, cols[0].1)?,
        hsv_filter(&hsv, cols[1].0, cols[1].1)?,
    );
    ret.push(vec![
        filter0.display,
        filter1.display,
    ]);

    let min_size = highgui::get_trackbar_pos(TR, WIN)?;
    let (blobbed0, blobbed1) = (
        remove_small_blobs(&filter0.mask, width, height, min_size)?,
        remove_small_blobs(&filter1.mask, width, height, min_size)?
    );
    ret.push(vec![
        convert(&blobbed0.blobbed, imgproc::COLOR_GRAY2BGR)?,
        convert(&blobbed1.blobbed, imgproc::COLOR_GRAY2BGR)?,
    ]);

    let mut candidates = vec![];
    for b0 in &blobbed0.info {
        for b1 in &blobbed1.info {
            if shapes_similar(b0, b1) {
                candidates.push([b0, b1]);
            }
        }
    }
    if candidates.is_empty() {
        return Ok(ret);
    }

    // TODO: if there's more
    let cand = candidates[0];
    println!("{:.2}",
        ((cand[0].bbox.width as f64 + cand[0].bbox.width as f64) / 2.) /
        (cand[1].bbox.height as f64 + cand[1].bbox.height as f64)
    );

    let final_pos = Point::new(
        (cand[0].centroid.x + cand[1].centroid.x) / 2,
        (cand[0].centroid.y + cand[1].centroid.y) / 2,
    );

    let mut to_draw = flipped;
    for c in &cand {
        imgproc::rectangle(
            &mut to_draw,
            Rect::new(
                c.bbox.x,
                c.bbox.y,
                c.bbox.width,
                c.bbox.height,
            ),
            color(255.0, 0., 0.),
            5,
            imgproc::LINE_8,
            0,
        )?;
    }
    imgproc::draw_marker(
        &mut to_draw,
        final_pos,
        color(255.0, 0., 0.),
        imgproc::MARKER_CROSS,
        50,
        10,
        imgproc::LINE_8,
    )?;
    ret.push(vec![to_draw]);

    Ok(ret)
}

pub fn same_magnitude<T>(mut i: T, mut j: T, mag: f64) -> bool
where T: PartialOrd + Into<f64> + Copy {
    if i > j {
        (i, j) = (j, i);
    }
    let r: f64 = i.into() / j.into();
    (1. - r) < mag
}

fn shapes_similar(b0: &BlobInfo, b1: &BlobInfo) -> bool {
    let areas = same_magnitude(b0.area, b1.area, 0.3);
    let xpos =  same_magnitude(b0.centroid.x, b1.centroid.x, 0.1);
    let y_pos_and_height = same_magnitude(
        (b0.centroid.y - b1.centroid.y).abs(),
        (b0.bbox.height + b1.bbox.height) / 2,
        0.2
    );
    let make_square = same_magnitude(
        ((b0.bbox.width as f64 + b0.bbox.width as f64) / 2.) /
        (b1.bbox.height as f64 + b1.bbox.height as f64),
        1.,
        0.2
    );

    areas && xpos && y_pos_and_height && make_square
}

#[allow(unused)]
fn color(r: f64, g: f64, b: f64) -> Scalar {
    Scalar::new(b, g, r, 255.)
}
