# double_layer_vit
## Installation

Install [PyTorch 1.9](https://pytorch.org/) then `pip install .` at the root of this repository.

We release models with a Vision Transformer backbone initialized from the [improved ViT](https://arxiv.org/abs/2106.10270) models.

Double Layer Segmentation models with ViT backbone:
<table>
  <tr>
    <th colspan="2">Download</th>
  </tr>
<tr>
    <td><a href="https://drive.google.com/uc?export=download&id=1iq6jDziuF1qoTbTG5Hr0o8hMDt896qzS&confirm=t&uuid=7abccd2f-f707-4e6f-b88f-ef70e5f588db">model</a></td>
    <td><a href="https://drive.google.com/uc?export=download&id=11_1Yt_LsDVsVU72cKjLW_M-MzZ4Kjgug">config</a></td>
  </tr>
</table>

## Inference

You can generate segmentation maps from your own data with:
```python
python -m segm.inference --model-path models/checkpoint.pth -i data/images/ -o segmaps/ 
```

The color coding rule for the segmentation mask is defined as DLS[0,255,255], Drusen[255,255,0], Background[128,0,255].

During the inference, all the images are rescaled to 512x512 then obtained prediction masks with the same resolution will be rescaled back to 500 x 768.

### model input
500(w) x 768(h), rgb(24bit per pixel, but grayscale) with JPG format

The resolution of the original B-scan image was 500 pixels in width, and 1536 pixels in height. The B-scan was then systematically cropped into 500x768 accordingly so that the retinal layer comes to the center of the cropped image.

### model output
500(w) x 768(h), rgb(24bit per pixel) with JPG format

## Image cropping

Usage: 

python crop_bscan.py --inputdir [path to the original B-scan (500x1536) directory] --outputdir [path to the output directory]

## Generate enface projection maps

Usage: 

python enface_proj.py  --inputdir [path to the directory where the prediction masks are stored] --outputdir [path to the directory you want to store projection maps] --eyeid [specify eyeID e.g. "P2202_OS"]

## BibTex

```
@article{kihara2022detection,
  title={Detection of Nonexudative Macular Neovascularization on Structural OCT Images Using Vision Transformers},
  author={Kihara, Yuka and Shen, Mengxi and Shi, Yingying and Jiang, Xiaoshuang and Wang, Liang and Laiginhas, Rita and Lyu, Cancan and Yang, Jin and Liu, Jeremy and Morin, Rosalyn and others},
  journal={Ophthalmology Science},
  volume={2},
  number={4},
  pages={100197},
  year={2022},
  publisher={Elsevier}
}
```


## Acknowledgements

The Vision Transformer code is based on [timm](https://github.com/rwightman/pytorch-image-models) library and the semantic segmentation pipeline is using [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) and [segmenter](https://github.com/rstrudel/segmenter).
