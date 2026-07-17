# Decoupling-face-inpainting

This workspace combines two research codebases into one face inpainting framework.

## Modules

### facial_structure_estimator

Training-stage auxiliary module.

This module is adapted from the LAP 3D face reconstruction project. During inpainting training, it is loaded as a frozen pretrained network and produces differentiable supervision signals:

- `canonical_image_lap`
- `canonical_normal_lap`
- `recon_normal_lap`

It is not the main inference-time model.

Standalone export is still available for inspection:

```shell
python demo.py --input ./images --result ./results --checkpoint_lap ./demo/checkpoint300.pth --supervision_only
```

### inpainting_module

Main inference-stage module.

This module is adapted from MAT, a mask-aware transformer image inpainting model. It is the primary model used during inference to complete masked or missing regions in face images.

During training:

- The Transformer/final output is passed through `facial_structure_estimator`.
- Its `canonical_normal_lap` and `recon_normal_lap` are compared with the same LAP outputs from the ground-truth image.
- The CNN/first-stage output is passed through `facial_structure_estimator`.
- Its `canonical_image_lap` is compared with the same LAP output from the ground-truth image.
- Gradients pass through the frozen estimator outputs back into the inpainting generator.

Example:

```shell
python generate_image.py --network pretrained/CelebA-HQ_512.pkl --dpath input_images --mpath masks --outdir outputs
```

## Training Flow

1. Prepare face images and masks.
2. Train `inpainting_module` with the frozen `facial_structure_estimator` checkpoint enabled in the loss.
3. Optimize the Transformer branch with `canonical_normal_lap` and `recon_normal_lap` supervision.
4. Optimize the CNN branch with `canonical_image_lap` supervision.
5. Save the trained inpainting model for inference.

Example:

```shell
python train.py \
    --outdir output_path \
    --data training_data_path \
    --gpus 1 \
    --cfg celeba512 \
    --structure_checkpoint ../facial_structure_estimator/demo/checkpoint300.pth \
    --structure_image_weight 1.0 \
    --structure_normal_weight 1.0 \
    --structure_canonical_normal_weight 1.0 \
    --structure_recon_normal_weight 1.0
```

## Inference Flow

1. Provide an input image and mask.
2. Run `inpainting_module`.
3. The inpainting output is the final result.

The facial structure estimator is intentionally kept out of the inference path unless a future experiment explicitly needs structure estimation at test time.
