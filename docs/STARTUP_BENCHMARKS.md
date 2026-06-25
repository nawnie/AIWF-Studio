# Studio Startup and Model-Load Benchmark Receipts

Run benchmarks with `--genlog`. Use the same models, storage, drivers, launch flags, prompt, dimensions, steps, and seed for every comparison.

## Required receipts

| Test | Cold definition | Warm definition | Record |
|---|---|---|---|
| Studio launch | fresh Python process and cold OS cache where practical | second launch | process start → browser-ready message |
| SD/SDXL | first load after process start | same pipeline retained | model load, prompt encode, denoise, VAE decode, total |
| Flux | first selected checkpoint | repeated prompt and model | transformer/T5 load, prompt cache hit, first image, total |
| Flux.2 Klein | first quantized/full-precision load | same checkpoint retained | residency mode, peak VRAM, load and total |
| Z-Image | first load | same checkpoint retained | residency mode, load and total |
| Wan 5B | first text encoder/transformer/VAE load | compatible second job | load, prompt encode, latent prep, denoise, VAE decode, encode |
| Wan high/low pair | first pair load | repeated compatible job | high/low switch time, PCIe/offload mode, total |
| Video Lab | first short clip | second same preset | probe, filter/encode rate, output duration, manifest status |
| RIFE | first checkpoint load | second clip same model | model load, pairs/sec, peak RAM/VRAM, encode time |

## Windows launch recipe

```bat
webui.bat --genlog
```

Record the wall-clock times printed in the console and attach:

- `aiwf.log`
- the relevant `outputs/genlog/generation-log.jsonl` rows
- Video Lab `job.json` and `ffmpeg.log`
- GPU model, VRAM, RAM, CPU, storage type, driver, and launch flags

## Acceptance targets

Studio v4 is accepted when:

- the UI no longer performs full checkpoint warmup by default;
- repeat Flux/Klein/Z-Image generations do not rebuild compatible pipelines;
- repeat Wan jobs reuse compatible cached/shared components;
- RIFE memory usage grows with chunk size rather than clip length;
- Video Lab output duration and A/V sync match the planned trim;
- no image/Wan/RIFE job loses GPU ownership to another job;
- cancellation never publishes a partial result as completed.
