# AutoVC

> This the implementation of this blog:
>
> * <https://ithelp.ithome.com.tw/m/articles/10262975>
> * <https://ithelp.ithome.com.tw/m/articles/10263395>

However, it doesn't performe well. Maybe I haven't find the "magic number."
If there is any bug in the code or you find the magic numbers. Tell me in the issue, please.

## Environment

* [CUDA](https://developer.nvidia.com/cuda-downloads) and [PyTorch](https://pytorch.org/)

```shell
pip install librosa tqdm gdown pyaudio wave pydub noisereduce
```

## AutoVC

* Paper: [AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss](https://arxiv.org/abs/1905.05879)
* Introduction
  * **Train** your model with `train.ipynb`, some pretrained models are [here](https://drive.google.com/file/d/18YMXyxUOmSULAMfKT_QT_eRFAhqVqEZh/view?usp=sharing) with bottleneck=16, 32 and 44. Also, I have [a corpus](https://drive.google.com/file/d/1Qq4WdRhAT2GNCdGcSpGLCpA21Uy-N3hc/view?usp=share_link) which is a subset of [VCTK corpus](https://datashare.ed.ac.uk/handle/10283/2950), you can test with it.
  * After get a model, you may want to test your voice as input of AutoVC, just run the inference part of `train.ipynb`, but you may need to revise some parameters.
* Reference:
  1. AutoVC repo: <https://github.com/auspicious3000/autovc.git>
  2. melGAN repo: <https://github.com/descriptinc/melgan-neurips.git>
  3. tutorial (melGAN): <https://ithelp.ithome.com.tw/articles/10261515>
