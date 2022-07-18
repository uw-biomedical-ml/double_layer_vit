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
    <td><a href="https://drive.google.com/drive/u/0/folders/1ajPRMaIFXI5Gef-H-T9jKDo9hVqvZ1Lk/checkpoint.pth">model</a></td>
    <td><a href="https://drive.google.com/drive/u/0/folders/1ajPRMaIFXI5Gef-H-T9jKDo9hVqvZ1Lk/variant.yml">config</a></td>
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

## BibTex

```
@article{yuka2022,
  title={Detection of Nonexudative Macular Neovascularization on Structural Optical Coherence Tomography Images using Vision Transformers},
  author={Yuka Kihara, Mengxi Shen, Yingying Shi, Xiaoshuang Jiang, Liang Wang, Rita Laiginhas, Cancan Lyu, Jin Yang, Jeremy Liu, Rosalyn Morin, Randy Lu, Hironobu F, William J. Feuer, Giovanni Gregori, Philip J. Rosenfeld, Aaron Y. Lee},
  journal={Ophthalmology Science},
  year={2022}
}
```


## Acknowledgements

The Vision Transformer code is based on [timm](https://github.com/rwightman/pytorch-image-models) library and the semantic segmentation pipeline is using [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) and [segmenter](https://github.com/rstrudel/segmenter).
