# AutoVC

> This is the implementation of the blogs mentioned below.
>
> * <https://ithelp.ithome.com.tw/m/articles/10262975>
> * <https://ithelp.ithome.com.tw/m/articles/10263395>

However, it doesn't perform well. If you find any bug or magic number. Tell me in the issue, please.

## Environment

* [CUDA](https://developer.nvidia.com/cuda-downloads) and [PyTorch](https://pytorch.org/)

```shell
pip install librosa tqdm gdown pyaudio wave pydub noisereduce
```

## AutoVC

* Paper: [AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss](https://arxiv.org/abs/1905.05879)
* Introduction
  * **Train** your model using `train.ipynb`. [Pretrained models](https://drive.google.com/file/d/18YMXyxUOmSULAMfKT_QT_eRFAhqVqEZh/view?usp=sharing) with bottleneck sizes of 16, 32, and 44 are available. Additionally, I have [a corpus](https://drive.google.com/file/d/1Qq4WdRhAT2GNCdGcSpGLCpA21Uy-N3hc/view?usp=share_link) that is a subset of the [VCTK corpus](https://datashare.ed.ac.uk/handle/10283/2950), which you can use for testing.
  * After obtaining a model, you may wish to test your voice as input for AutoVC. Simply execute the inference section of the `train.ipynb` file, but be prepared to make adjustments to certain parameters.
* Reference:
  1. AutoVC repo: <https://github.com/auspicious3000/autovc.git>
  2. melGAN repo: <https://github.com/descriptinc/melgan-neurips.git>
  3. tutorial (melGAN): <https://ithelp.ithome.com.tw/articles/10261515>
