pub mod processors;
pub mod steps;

use opencv::{
    highgui::{named_window, WINDOW_GUI_NORMAL, wait_key, destroy_all_windows, imshow},
    videoio::{CAP_ANY, VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT},
    core::{self, CV_8UC3, Range},
    prelude::*,
    imgproc,
};
use steps::get_steps;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> Res<()> {
    static WIN: &str = "test";

    let mut cap = VideoCapture::new(0, CAP_ANY)?;
    if !cap.is_opened()? {
        return Ok(());
    }

    named_window(WIN, WINDOW_GUI_NORMAL)?;

    let w = cap.get(CAP_PROP_FRAME_WIDTH)? as usize;
    let h = cap.get(CAP_PROP_FRAME_HEIGHT)? as usize;

    let ss = get_steps().len() + 1;

    let sw = (ss as f64).sqrt().ceil() as usize;
    let sh = (ss as f64 / sw as f64).ceil() as usize;

    let mut img = Mat::new_nd_with_default(&[
        (sh * h) as i32,
        (sw * w) as i32,
    ], CV_8UC3, 0.into())?;
    let mut frame = Mat::default();

    loop {
        if !cap.read(&mut frame)? {
            break;
        }

        let (mut x, mut y) = (1, 0);
        let mut si = 1;

        frame.copy_to(
            &mut img
            .row_range(&Range::new(0, h as i32)?)?
            .col_range(&Range::new(0, w as i32)?)?
        )?;

        for s in get_steps() {
            let (xw, yh) = (x * w, y * h);

            let (new_frame, conversion) = s(frame, w, h)?;
            frame = if let Some(conversion) = conversion {
                conversion(new_frame)?
            } else {
                new_frame
            };

            frame.copy_to(
                &mut img
                .row_range(&Range::new(yh as i32, ((y + 1) * h) as i32)?)?
                .col_range(&Range::new(xw as i32, ((x + 1) * w) as i32)?)?
            )?;

            imgproc::put_text(
                &mut img,
                &si.to_string(),
                core::Point::new(xw as i32 + 10, (yh + h) as i32 - 50),
                imgproc::FONT_HERSHEY_SIMPLEX,
                3.,
                core::Scalar::new(255., 255., 255., 0.),
                5,
                imgproc::LINE_8,
                false
            )?;

            si += 1;
            x += 1;
            if x == sw {
                x = 0;
                y += 1;
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
