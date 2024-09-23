# PuLID ComfyUI

[PuLID](https://github.com/ToTheBeginning/PuLID) ComfyUI native implementation.

![basic workflow](examples/pulid_wf.jpg)

## Important updates

- **2024.05.12:** Added attention masking and the Advanced node, allows fine tuning of the generation.
- **2024.09.15:** Added support for PuLID-FLUX-v0.9.0 model, including FP8 optimization for consumer-grade GPUs.

## Notes

The code now supports both the original XL model and the new FLUX model. The FLUX model includes optimizations for consumer-grade GPUs.

## The 'method' parameter

`method` applies the weights in different ways. `Fidelity` is closer to the reference ID, `Style` leaves more freedom to the checkpoint. Sometimes the difference is minimal. `neutral` doesn't do any normalization.

## New FLUX Options

When using the FLUX model, you have additional options:

- `fidelity`: A slider to adjust the fidelity of the output. Lower values grant higher resemblance to the reference image.
- `projection`: Choose between `ortho` and `ortho_v2` projection methods.

## Installation

- [PuLID pre-trained model](https://huggingface.co/huchenlei/ipadapter_pulid/resolve/main/ip-adapter_pulid_sdxl_fp16.safetensors?download=true) goes in `ComfyUI/models/pulid/` (for XL model)
- [PuLID-FLUX-v0.9.0 model](https://huggingface.co/ToTheBeginning/PuLID-FLUX-v0.9.0) should be placed in `ComfyUI/models/pulid/PuLID-FLUX-v0.9.0/`
- The EVA CLIP is EVA02-CLIP-L-14-336, but should be downloaded automatically (will be located in the huggingface directory).
- `facexlib` dependency needs to be installed, the models are downloaded at first use
- Finally you need InsightFace with [AntelopeV2](https://huggingface.co/MonsterMMORPG/tools/tree/main), the unzipped models should be placed in `ComfyUI/models/insightface/models/antelopev2`.

## Usage

1. Select your desired model (XL, FLUX-BF16, or FLUX-FP8) in the node settings.
2. For FLUX models, you can adjust the `fidelity` and `projection` options for fine-tuning.
3. Use the `method` parameter to choose between different weight application methods.
4. Adjust the `weight` parameter to control the strength of the PuLID effect.

Enjoy using PuLID with the new FLUX model support!

