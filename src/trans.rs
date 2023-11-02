use candle_transformers::models::stable_diffusion::{
    build_clip_transformer,
    ddim::DDIMScheduler,
    unet_2d::UNet2DConditionModel,
    vae::{AutoEncoderKL, DiagonalGaussianDistribution},
    StableDiffusionConfig,
};

use crate::{
    model::{ModelFile, StableDiffusionVersion},
    utils::{device, output_filename, save_image},
    Args,
};
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use tokenizers::Tokenizer;

use crate::GUIDANCE_SCALE;

pub struct TransformComponent {
    n_steps: usize,
    final_image: String,
    num_samples: i64,
    img2img: Option<String>,
    img2img_strength: f64,
    intermediary_images: bool,
    dtype: DType,
    sd_config: StableDiffusionConfig,
    text_embeddings: Tensor,
    vae: AutoEncoderKL,
    init_latent_dist: Option<DiagonalGaussianDistribution>,
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    device: Device,
}

impl TransformComponent {
    pub fn new(args: Args) -> Result<Self> {
        args.check()?;
        let dtype = if args.use_f16 { DType::F16 } else { DType::F32 };
        let sd_config = match args.sd_version {
            StableDiffusionVersion::V1_5 | StableDiffusionVersion::Ghibli => {
                StableDiffusionConfig::v1_5(args.sliced_attention_size, args.height, args.width)
            }
            StableDiffusionVersion::V2_1 => {
                StableDiffusionConfig::v2_1(args.sliced_attention_size, args.height, args.width)
            }
            StableDiffusionVersion::Xl => {
                StableDiffusionConfig::sdxl(args.sliced_attention_size, args.height, args.width)
            }
        };

        let scheduler = sd_config.build_scheduler(args.n_steps)?;
        let device = device(args.cpu)?;

        let which = match args.sd_version {
            StableDiffusionVersion::Xl => vec![true, false],
            _ => vec![true],
        };
        let text_embeddings = which
            .iter()
            .map(|first| {
                text_embeddings(
                    &args.prompt,
                    &args.uncond_prompt,
                    args.tokenizer.clone(),
                    args.clip_weights.clone(),
                    args.sd_version,
                    &sd_config,
                    args.use_f16,
                    &device,
                    dtype,
                    *first,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let text_embeddings = Tensor::cat(&text_embeddings, D::Minus1)?;
        println!("{text_embeddings:?}");

        println!("Building the autoencoder.");
        let vae_weights = ModelFile::Vae.get(args.vae_weights, args.sd_version, args.use_f16)?;
        let vae = sd_config.build_vae(vae_weights, &device, dtype)?;
        let init_latent_dist = match &args.img2img {
            None => None,
            Some(image) => {
                let image = image_preprocess(image)?.to_device(&device)?;
                Some(vae.encode(&image)?)
            }
        };

        println!("Building the unet.");
        let unet_weights = ModelFile::Unet.get(args.unet_weights, args.sd_version, args.use_f16)?;
        let unet = sd_config.build_unet(unet_weights, &device, 4, args.use_flash_attn, dtype)?;

        Ok(Self {
            n_steps: args.n_steps,
            final_image: args.final_image,
            num_samples: args.num_samples,
            img2img: args.img2img,
            img2img_strength: args.img2img_strength,
            intermediary_images: args.intermediary_images,
            dtype,
            sd_config,
            text_embeddings,
            vae,
            init_latent_dist,
            unet,
            scheduler,
            device,
        })
    }

    pub fn run(&self) -> Result<()> {
        let n_steps = self.n_steps;
        let num_samples = self.num_samples;
        let t_start = if self.img2img.is_some() {
            n_steps - (n_steps as f64 * self.img2img_strength) as usize
        } else {
            0
        };

        for idx in 0..num_samples {
            let timesteps = self.scheduler.timesteps();
            let mut latents = self.init_latents(t_start, timesteps)?;

            println!("starting sampling");
            for (timestep_index, &timestep) in timesteps.iter().enumerate() {
                if timestep_index < t_start {
                    continue;
                }
                latents = self.step_process(idx, timestep_index, timestep, latents)?;
            }

            println!(
                "Generating the final image for sample {}/{}.",
                idx + 1,
                num_samples
            );
            let image = self.vae.decode(&(&latents / 0.18215)?)?;
            let image = ((image / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
            let image = (image.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?.i(0)?;
            let image_filename = output_filename(&self.final_image, idx + 1, num_samples, None);
            save_image(&image, image_filename)?
        }
        Ok(())
    }

    fn init_latents(&self, t_start: usize, timesteps: &[usize]) -> Result<Tensor> {
        let latents = match &self.init_latent_dist {
            Some(init_latent_dist) => {
                let latents = (init_latent_dist.sample()? * 0.18215)?.to_device(&self.device)?;
                if t_start < timesteps.len() {
                    let noise = latents.randn_like(0f64, 1f64)?;
                    self.scheduler
                        .add_noise(&latents, noise, timesteps[t_start])?
                } else {
                    latents
                }
            }
            None => {
                let latents = Tensor::randn(
                    0f32,
                    1f32,
                    (1, 4, self.sd_config.height / 8, self.sd_config.width / 8),
                    &self.device,
                )?;
                // scale the initial noise by the standard deviation required by the scheduler
                (latents * self.scheduler.init_noise_sigma())?
            }
        };
        let latents = latents.to_dtype(self.dtype)?;
        Ok(latents)
    }

    fn step_process(
        &self,
        idx: i64,
        timestep_index: usize,
        timestep: usize,
        latents: Tensor,
    ) -> Result<Tensor> {
        let start_time = std::time::Instant::now();
        let latent_model_input = Tensor::cat(&[&latents, &latents], 0)?;

        let latent_model_input = self
            .scheduler
            .scale_model_input(latent_model_input, timestep)?;
        let noise_pred =
            self.unet
                .forward(&latent_model_input, timestep as f64, &self.text_embeddings)?;
        let noise_pred = noise_pred.chunk(2, 0)?;
        let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
        let noise_pred =
            (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * GUIDANCE_SCALE)?)?;
        let latents = self.scheduler.step(&noise_pred, timestep, &latents)?;
        let dt = start_time.elapsed().as_secs_f32();

        let n_steps = self.n_steps;
        println!("step {}/{n_steps} done, {:.2}s", timestep_index + 1, dt);

        if self.intermediary_images {
            let image = self.vae.decode(&(&latents / 0.18215)?)?;
            let image = ((image / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
            let image = (image * 255.)?.to_dtype(DType::U8)?.i(0)?;
            let image_filename = output_filename(
                &self.final_image,
                idx + 1,
                self.num_samples,
                Some(timestep_index + 1),
            );
            save_image(&image, image_filename)?;
        }

        Ok(latents)
    }
}

#[allow(clippy::too_many_arguments)]
fn text_embeddings(
    prompt: &str,
    uncond_prompt: &str,
    tokenizer: Option<String>,
    clip_weights: Option<String>,
    sd_version: StableDiffusionVersion,
    sd_config: &StableDiffusionConfig,
    use_f16: bool,
    device: &Device,
    dtype: DType,
    first: bool,
) -> Result<Tensor> {
    let tokenizer_file = if first {
        ModelFile::Tokenizer
    } else {
        ModelFile::Tokenizer2
    };
    let tokenizer = tokenizer_file.get(tokenizer, sd_version, use_f16)?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let pad_id = match &sd_config.clip.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    println!("Running with prompt \"{prompt}\".");
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    while tokens.len() < sd_config.clip.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

    let mut uncond_tokens = tokenizer
        .encode(uncond_prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    while uncond_tokens.len() < sd_config.clip.max_position_embeddings {
        uncond_tokens.push(pad_id)
    }
    let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), device)?.unsqueeze(0)?;

    println!("Building the Clip transformer.");
    let clip_weights_file = if first {
        ModelFile::Clip
    } else {
        ModelFile::Clip2
    };
    let clip_weights = clip_weights_file.get(clip_weights, sd_version, false)?;
    let clip_config = if first {
        &sd_config.clip
    } else {
        sd_config.clip2.as_ref().unwrap()
    };
    let text_model = build_clip_transformer(clip_config, clip_weights, device, DType::F32)?;
    let text_embeddings = text_model.forward(&tokens)?;
    let uncond_embeddings = text_model.forward(&uncond_tokens)?;
    let text_embeddings = Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(dtype)?;
    Ok(text_embeddings)
}

fn image_preprocess<T: AsRef<std::path::Path>>(path: T) -> anyhow::Result<Tensor> {
    let img = image::io::Reader::open(path)?.decode()?;
    let (height, width) = (img.height() as usize, img.width() as usize);
    let height = height - height % 32;
    let width = width - width % 32;
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?
        .unsqueeze(0)?;
    Ok(img)
}
