# Monet

`Monet` is an AI image generation tool developed inspired by Candle
(https://github.com/huggingface/candle). It is serverless and ready to use out
of the box.

## How to play

- Build Monet

```
make release
```

- Command help

```
./monet --help
```

- Generate image by prompts example

```
./monet --prompt "Beautiful pakistani girl, wearing high-quality modern dress, on a balcony with a beautiful nightscape background, beautiful smile on her face, short black hair, tall height, looking into the camera, very fair skin tone, photo captured with Canon EOS R5 camera with an 85mm f/1.4 prime lens" --sd-version xl
```

- img2img example

```
./monet --img2img sd_final.png --prompt "Beautiful city after rain"
```