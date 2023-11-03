pub mod model;
pub mod trans;
pub mod utils;

use anyhow::Result;
use clap::Parser;
use model::StableDiffusionVersion;
use thiserror::Error;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{prelude::__tracing_subscriber_SubscriberExt, util::SubscriberInitExt};

const GUIDANCE_SCALE: f64 = 7.5;

#[derive(Error, Debug)]
pub enum SError {
    #[error("{0}")]
    String(&'static str),
    #[error("{0}")]
    Owned(String),
}

#[derive(Parser, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A very realistic photo of a rusty robot walking on a sandy beach"
    )]
    prompt: String,

    #[arg(long, default_value = "")]
    uncond_prompt: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The height in pixels of the generated image.
    #[arg(long)]
    height: Option<usize>,

    /// The width in pixels of the generated image.
    #[arg(long)]
    width: Option<usize>,

    /// The UNet weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    unet_weights: Option<String>,

    /// The CLIP weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    clip_weights: Option<String>,

    /// The VAE weight file, in .safetensors format.
    #[arg(long, value_name = "FILE")]
    vae_weights: Option<String>,

    #[arg(long, value_name = "FILE")]
    /// The file specifying the tokenizer to used for tokenization.
    tokenizer: Option<String>,

    /// The size of the sliced attention or 0 for automatic slicing (disabled by default)
    #[arg(long)]
    sliced_attention_size: Option<usize>,

    /// The number of steps to run the diffusion for.
    #[arg(long, default_value_t = 30)]
    n_steps: usize,

    /// The number of samples to generate.
    #[arg(long, default_value_t = 1)]
    pub num_samples: i64,

    /// The max number of threads work in cocurrency
    #[arg(long, default_value_t = 1)]
    pub num_threads: usize,

    /// The name of the final image to generate.
    #[arg(long, value_name = "FILE", default_value = "sd_final.png")]
    final_image: String,

    #[arg(long, value_enum, default_value = "xl")]
    sd_version: StableDiffusionVersion,

    /// Generate intermediary images at each step.
    #[arg(long, action)]
    intermediary_images: bool,

    #[arg(long)]
    use_flash_attn: bool,

    #[arg(long)]
    use_f16: bool,

    #[arg(long, value_name = "FILE")]
    img2img: Option<String>,

    /// The strength, indicates how much to transform the initial image. The
    /// value must be between 0 and 1, a value of 1 discards the initial image
    /// information.
    #[arg(long, default_value_t = 0.8)]
    img2img_strength: f64,
}

impl Args {
    pub fn check(&self) -> Result<()> {
        let is = self.img2img_strength;
        if !(0. ..=1.).contains(&is) {
            anyhow::bail!("img2img-strength should be between 0 and 1, got {is}")
        }

        if self.tracing {
            let (chrome_layer, _) = ChromeLayerBuilder::new().build();
            tracing_subscriber::registry().with(chrome_layer).init();
        }

        Ok(())
    }
}
