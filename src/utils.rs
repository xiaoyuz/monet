use std::path::Path;

use anyhow::Result;
use candle_core::{Device, Error, Tensor};
use image::ImageBuffer;

/// Saves an image to disk using the image crate, this expects an input with shape
/// (c, height, width).
pub async fn save_image<P: AsRef<Path>>(img: &Tensor, p: P) -> Result<()> {
    let p = p.as_ref();
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        return Err(Error::Msg(
            "save_image expects an input of shape (3, height, width)".to_string(),
        )
        .into());
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match ImageBuffer::from_raw(width as u32, height as u32, pixels) {
            Some(image) => image,
            None => return Err(Error::Msg("error saving image".to_string()).into()),
        };
    image.save(p).map_err(candle_core::Error::wrap)?;
    Ok(())
}

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else {
        let device = Device::cuda_if_available(0)?;
        if !device.is_cuda() {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(device)
    }
}

pub fn output_filename(basename: &str, sample_idx: i64, timestep_idx: Option<usize>) -> String {
    let filename = match basename.rsplit_once('.') {
        None => format!("{basename}.{sample_idx}.png"),
        Some((filename_no_extension, extension)) => {
            format!("{filename_no_extension}.{sample_idx}.{extension}")
        }
    };
    match timestep_idx {
        None => filename,
        Some(timestep_idx) => match filename.rsplit_once('.') {
            None => format!("{filename}-{timestep_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}-{timestep_idx}.{extension}")
            }
        },
    }
}
