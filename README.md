# PanopticML

Default plugin for panoptic, provides all machine learning functionalities such as:
- image embeddings using OpenAI CLIP
- images clustering using FAISS Kmeans or HDBScan
- images to images similarity using FAISS Index L2
- text to images similarity using FAISS Index L2 + CLIP

## Compatibility

PanopticML 1.x is built for the plugin API of Panoptic 1.0 and later. With Panoptic 0.x, use PanopticML 0.0.11 or earlier.

While Panoptic 1.0 is in pre-release, PanopticML 1.x is published as a pre-release too, so pip only installs it when asked: `pip install --pre panopticml` or `pip install panopticml==1.0.0rc1`.

## Available models

Several models are available in this plugin to compute the embeddings: 
- [google mobilenet](https://huggingface.co/google/mobilenet_v2_1.0_224): a really light model with lower performances but made to run on bad devices, no support for text similarity
- [openAI CLIP](https://huggingface.co/openai/clip-vit-base-patch32): default model for panoptic, good performances, runs well on most computers with a decent CPU, support text similarity and has a good understanding of semantics
- [meta dinosV2](https://huggingface.co/facebook/dinov2-base): model that will have better performances than CLIP on pure visual similarities but less understanding of themes and semantics, no support for text similarity, runs well on CPU
- [google SIGLIP2](https://huggingface.co/google/siglip2-so400m-patch16-naflex): computing heavy model, NVDIA GPU recommended but works also on CPU (it will take a long time to compute embeddings), way better results than CLIP on text similarity and semantics, and pretty good visual features
- [meta DINOv3](https://huggingface.co/docs/transformers/main/model_doc/dinov3): successor of DINOv2, stronger visual features, no support for text similarity
- [NVIDIA C-RADIOv4](https://huggingface.co/nvidia/C-RADIOv4-H): heavy agglomerative vision model (512px input), GPU recommended, no support for text similarity
- [apple MobileCLIP2](https://huggingface.co/apple/MobileCLIP2-S2): fast CLIP-like model (S2 or L-14), supports text similarity
- [google EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2): multimodal embedding model (text, 100+ languages, images in one space), supports text similarity, GPU recommended (heavy on CPU)
- auto transformers: want to tryout any huggingface multimodal model ? you can just provide its id to panopticML and should be able to use it directly

## Apple Silicon

On Apple Silicon Macs, embeddings are computed on the GPU (MPS) in reduced precision, and three models get extra compute engines:
- CLIP: the Neural Engine runs next to the GPU, ~1.9x faster (M4 Pro: ~1150 images/s instead of ~600)
- SIGLIP2: the Neural Engine runs next to the GPU, ~2.3x faster (M4 Pro: ~65 images/s instead of ~28)
- EmbeddingGemma 2: MLX runs the vision tower on the GPU (~2x faster than PyTorch), and the Neural Engine runs next to it, ~3x faster overall (M4 Pro: ~8 images/s instead of ~2.5)

The first time a model is used, it is converted for the Neural Engine in the background (one to three minutes, the GPU computes meanwhile) and cached in `~/.cache/panopticml`. Neural Engine vectors are computed in fp16, slightly less precise than the GPU's (cosine 0.97 to 0.9998 to the full-precision vector on photos, against more than 0.9998 on the GPU). Search quality is unchanged: on Flickr30k (1000 photos, 5000 captions), text-to-image and image-to-text recall with the Neural Engine is within 0.5 point of the GPU's for all three models. Turn the `neural_engine` plugin setting off to compute on the GPU only.

Environment variables:
- `PANOPTICML_DEVICE`: force the PyTorch device (`cpu`, `cuda`, `mps`)
- `PANOPTICML_ANE=0`, `PANOPTICML_MLX=0`: never use the Neural Engine / MLX
- `PANOPTICML_CACHE`: where converted Core ML models are stored (default `~/.cache/panopticml`)

## Clustering functions
- Kmeans: specify a number of clusters, really fast
- HDBScan: automatic number of clusters, a bit slower and excludes a lot of images
- Images to text: considering several texts, create a cluster per text and put each image to the text it is the closest to
- Find duplicates: create clusters with only images sharing a lot of similarity together, good to find duplicates or near duplicates