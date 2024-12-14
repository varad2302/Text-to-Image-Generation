# Text-to-Image-Generation
An implementation of the RAT-Diffusion model to generate images of birds based on textual descriptions

### Installation

Clone this repo.
```
git clone https://github.com/varad2302/Text-to-Image_Generation
conda env create -f environment.yml
conda activate RAT
```

### Datasets Preparation
1. Download the preprocessed metadata for [birds_dataset](https://drive.google.com/file/d/1s-R4dDrfry6W8jFv0KFe3Q8_gtCtFzSG/view?usp=drive_link), [birds_extra](https://drive.google.com/file/d/13o3HM7KacIciqJOtIBZco4IRzOebSB5Y/view?usp=drive_link) and save them to `dataset/`
2. Download the [bird_dataset](http://www.vision.caltech.edu/datasets/cub_200_2011/) image data and extract them to `dataset/bird/dataset`,Download the [bird_extra](https://drive.google.com/file/d/1oHz3sUPZ_dKDjNOIxZSMRXq-yX2EytXR/view?usp=drive_link) image data and extract them to `dataset/bird/extra`,

---
### Training on extrapolated data

**Train RAT-Diffusion models:**
  - : `python train.py --model "DiT_VQ" --image-size 256 --epochs 1 --global-batch-size 8 --vae "ema"--num-workers 0 ` 

### Fine-tuning on the original dataset

**Train RAT-Diffusion models:**
  - : `python train.py --model "DiT_VQ" --image-size 256 --epochs 1 --global-batch-size 8 --vae "ema" --ckpt "path to Pytorch checkpoint" --num-workers 0 ` 


### Generate Samples
  - : `python generate_samples.py --model "model name" --image-size 256 --vae "ema" --ckpt "path to trained model checkpoint" --num-workers 0 `
