pub mod steps;
pub mod utils;

use opencv::{
    highgui::{named_window, WINDOW_GUI_NORMAL, wait_key, destroy_all_windows, imshow},
    videoio::{CAP_ANY, VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT},
    prelude::*,
    imgproc,
    core,
};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub static WIN: &str = "test";
fn main() -> Res<()> {
    let mut cap = VideoCapture::new(0, CAP_ANY)?;
    
    named_window(WIN, WINDOW_GUI_NORMAL)?;

    let (w, h) = (
        cap.get(CAP_PROP_FRAME_WIDTH)?  as i32,
        cap.get(CAP_PROP_FRAME_HEIGHT)? as i32,
    );

    steps::init()?;

    let mut frame = Mat::default();
    loop {
        if !cap.read(&mut frame)? {
            break;
        }

        let steps = steps::get_steps(&frame, w, h)?;

        let sw = steps.len() as i32;
        let sh = steps.iter()
            .map(|s| s.1.len() as i32)
            .max()
            .ok_or("What....")?;

        let mut img = Mat::new_nd_with_default(&[
            (sh * h),
            (sw * w),
        ], core::CV_8UC3, 0xff.into())?;

        for (x, (name, step)) in steps.iter().enumerate() {
            let x = x as i32;
            let xw = x * w;
            for (y, disp) in step.iter().enumerate() {
                let y = y as i32;
                let yh = y * h;

                disp.copy_to(
                    &mut img
                    .row_bounds(yh, (y + 1) * h)?
                    .col_bounds(xw, (x + 1) * w)?
                )?;

                imgproc::put_text(
                    &mut img,
                    name,
                    core::Point::new(xw, 50),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    2.,
                    utils::color(255., 255., 255.),
                    5,
                    imgproc::LINE_8,
                    false
                )?;
            }
        }

        imshow(WIN, &img)?;

        if wait_key(1)? == 'q' as i32 {
            break;
        }
    }

    destroy_all_windows()?;

    Ok(())
}
