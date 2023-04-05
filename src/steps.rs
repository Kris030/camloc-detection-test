
use opencv::{
    core::{self, Mat, Point, Scalar, Vector as CVec},
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

fn blur(src: &Mat, bw: i32, bh: i32) -> Res<Mat> {
    let mut dst = Mat::default();

    imgproc::blur(src, &mut dst,
        core::Size::new(bw, bh),
        Point::new(-1, -1),
        core::BORDER_DEFAULT
    )?;

    Ok(dst)
}


struct BlobbingRet {
    blobbed: Mat,
    centroids: Vec<(i32, core::Point)>,
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
    
    let mut filtered = 0;
    let mut filtered_pixels = 0;
    
    let mut centr = Vec::with_capacity(centroids.rows() as usize - 1);
    let mut mask = Mat::new_nd_with_default(&[height, width], core::CV_8UC1, 0.into())?;

    for blob in 1..stats.rows() {
        let size = *stats.at_2d::<i32>(blob, imgproc::CC_STAT_AREA)?;
        if size < min_size {
            filtered += 1;
            filtered_pixels += size;
            continue;
        }
        centr.push((
            size,
            core::Point::new(
                *centroids.at_2d::<f64>(blob, 0)? as i32,
                *centroids.at_2d::<f64>(blob, 1)? as i32,
            )
        ));

        let with = Mat::new_nd_with_default(&[1], core::CV_8UC1, blob.into())?;
        core::compare(&labels, &with, &mut mask, core::CMP_EQ)?;

        dst = combine_binaries([
            &dst, &mask
        ], BitwiseOperation::Or)?;
    }

    let all_rows = stats.rows() - 1;
    eprintln!("filtered {filtered} ({filtered_pixels} pixels) of {} ({:.2}%)", all_rows, filtered as f64 / all_rows as f64 * 100.);

    centr.sort_by(|(a, _), (b, _)| a.cmp(b).reverse());

    Ok(BlobbingRet {
        blobbed: dst,
        centroids: centr,
    })
}

static TR: &str = "min_pixels";
pub fn init() -> Res<()> {
    highgui::create_trackbar(TR, WIN, None, 5000, None)?;

    let read_colors = |fname: &str| -> Result<([i32; 3], [i32; 3]), Box<dyn std::error::Error>> {
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
    let blurred = blur(&flipped, 10, 10)?;
    
    let hsv = convert(&blurred, imgproc::COLOR_BGR2HSV)?;

    let Some(cols) = (unsafe { &colors }) else {
        return Err("".into());
    };

    let (
        filter0,
        filter1,
        // filter2
    ) = (
        // [50,  50,  0]   [100, 255, 255]
        // [80,  115, 200] [100, 255, 255]
        // [100, 0,   140] [150, 200, 255]

        hsv_filter(&hsv, cols[0].0, cols[0].1)?,
        hsv_filter(&hsv, cols[1].0, cols[1].1)?,
        // hsv_filter(&hsv, cols[2].0, cols[2].1)?,
    );
    ret.push(vec![
        filter0.display,
        filter1.display,
        // filter2.display,
    ]);

    let merged = combine_binaries([
        &filter0.mask,
        &filter1.mask,
        // &filter2.mask,
    ], BitwiseOperation::Or)?;
    ret.push(vec![convert(&merged, imgproc::COLOR_GRAY2BGR)?]);

    let min_size = highgui::get_trackbar_pos(TR, WIN)?;
    let BlobbingRet { blobbed, centroids } = remove_small_blobs(&merged, width, height, min_size)?;

    let blobbed_bgr = convert(&blobbed, imgproc::COLOR_GRAY2BGR)?;
    ret.push(vec![blobbed_bgr.clone()]);

    let mut contours = CVec::<CVec<Point>>::new();
    let mut hierarchy = Mat::default();
    imgproc::find_contours_with_hierarchy(
        &blobbed,
        &mut contours,
        &mut hierarchy,
        imgproc::RETR_TREE, imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0)
    )?;

    let mut contoured = blobbed_bgr;
    imgproc::draw_contours(
        &mut contoured,
        &contours, -1,
        color(0., 0., 255.),
        5,
        imgproc::LINE_8,
        &hierarchy,
        i32::MAX,
        Point::new(0, 0),
    )?;

    if contours.is_empty() {
        return Ok(ret);
    }
    ret.push(vec![contoured.clone()]);

    imgproc::draw_marker(
        &mut contoured,
        centroids[0].1,
        Scalar::new(0., 0., 255., 255.),
        imgproc::MARKER_CROSS,
        100, 5, imgproc::LINE_8,
    )?;

    for cnt in contours {
        let mut approx = Mat::default();
        // let p = cnt.get(0)?;
        // let (x1, y1) = (p.x, p.y);
        imgproc::approx_poly_dp(
            &cnt,
            &mut approx,
            0.01 * imgproc::arc_length(&cnt, true)?,
            true
        )?;

        if approx.rows() == 4 {
            let br = imgproc::bounding_rect(&cnt)?;
            let (_x, _y, w, h) = (br.x, br.y, br.width, br.height);

            let ratio = w as f64 / h as f64;
            if (0.9..=1.1).contains(&ratio) {
                imgproc::draw_contours(
                    &mut contoured,
                    &CVec::<CVec<Point>>::from_iter([cnt]),
                    -1,
                    color(0., 255., 255.),
                    5,
                    imgproc::LINE_8,
                    &core::no_array(),
                    i32::MAX,
                    Point::new(0, 0),
                )?;
            } else {
                imgproc::draw_contours(
                    &mut contoured,
                    &CVec::<CVec<Point>>::from_iter([cnt]),
                    -1,
                    color(0., 255., 0.),
                    5,
                    imgproc::LINE_8,
                    &core::no_array(),
                    i32::MAX,
                    Point::new(0, 0),
                )?;
            }
        }
    }

    ret.push(vec![contoured.clone()]);

    Ok(ret)
}

fn color(r: f64, g: f64, b: f64) -> Scalar {
    Scalar::new(r, g, b, 255.)
}
