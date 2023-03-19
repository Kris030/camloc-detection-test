use opencv::core::{self, Mat};
use crate::Res;

pub type Conversion = Option<Box<dyn FnOnce(Mat) -> Res<Mat>>>;
pub type ProcessingFunction = Box<dyn FnOnce(Mat, usize, usize) -> Res<(Mat, Conversion)>>;

fn create_flip() -> ProcessingFunction {
    Box::new(|f, _, _| {
        let mut r = Mat::default();
        core::flip(&f, &mut r, 1)?;
        Ok((r, None))
    })
}

fn create_hsl_filter(s: [i32; 3], e: [i32; 3]) -> ProcessingFunction {
    use opencv::core::{in_range, bitwise_and};

    Box::new(move |f, _, _| {
        let mut r = Mat::default();

        in_range(
            &f,
            &Mat::from_slice(&s)?,
            &Mat::from_slice(&e)?,
            &mut r
        )?;

        let c = move |fr: Mat| {
            let mut res = Mat::default();
            bitwise_and(&f, &f, &mut res, &fr)?;
            Ok(res)
        };

        Ok((r, Some(Box::new(c))))
    })
}

pub fn get_steps() -> Vec<ProcessingFunction> {
    vec![
        create_flip(),
        create_hsl_filter([0, 0, 112], [255, 255, 255]),
        create_flip(),
    ]
}
