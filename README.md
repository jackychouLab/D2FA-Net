# D<sup>2</sup>FA-Net: Decoupled Deformable Frequency-Aware Network for Radar Object Detection

We are very grateful for the source code provided by [`RODNet`](https://github.com/yizhou-wang/RODNet), which our project extends upon. This is the official implementation of our D<sup>2</sup>FA-Net papers. 

Please cite our paper if this repository is helpful for your research:

```
@article{D2FA-Net,
  title={D^{2}FA-Net: Decoupled Deformable Frequency-Aware Network for Radar Object Detection},
  author={Zhou, Jianhong and Ke, Feng and Zhai, Yikui, and Jiang, Ziyi and Zhang, Xiu Yin and Gao, Feifei},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={-},
  number={-},
  pages={-},
  year={-},
  publisher={IEEE}
}
```

## Installation

```commandline
cd $D2FA-Net_ROOT
git clone https://github.com/jackychouLab/D2FA-Net.git
```

Create a conda environment for D2FA-Net. Tested under Python 3.10.
```commandline
conda create -n D2FA-Net python=3.10 -y
conda activate D2FA-Net
```

Note: This work uses CUDA 12.8 and cuDNN 8.9.
```commandline
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Install `cruw-devkit` package. 
```commandline
cd cruw-devkit
pip install -r requirements.txt --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install pynvml fvcore thop timm einops --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy==1.26.4 opencv-python==4.11.0.86 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install . --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
cd ..
```

Setup package.
```commandline
pip install -e . --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple --no-build-isolation
cd rodnet/ops/tdc_deform_ext
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST="X.X"
pip install -e . --no-build-isolation --no-build-isolation -v
cd ..
cd tdc_deform_ext_3d
pip install -e . --no-build-isolation --no-build-isolation -v
```
**Note:** This work uses an NVIDIA RTX 5000 Ada GPU. Therefore, when setting
```commandline
export TORCH_CUDA_ARCH_LIST="X.X"
 ```
`X.X` should be set to `8.9`. If an NVIDIA RTX 5090 GPU is used instead, `X.X` should be set to `12.0`.

Please adjust this value according to the compute capability of the specific GPU being used.

Download the new CRUW[`Key:yad3`](https://pan.baidu.com/s/1kUAv3iHbzcBEYEviD7pQBg) files and use them to replace all files within the compiled CRUW directory
```commandline
rm -r {Your Environment Path}/lib/python3.10/site-packages/cruw
rm -r {Your Environment Path}/lib/python3.10/site-packages/cruw_devkit-1.1.dist-info
mv $New_cruw {Your Environment Path}/lib/python3.10/site-packages
mv $New_cruw_devkit-1.1.dist-info {Your Environment Path}/lib/python3.10/site-packages
```

## Prepare data for CRUW dataset

Download [ROD2021 dataset](https://www.cruwdataset.org/download#h.mxc4upuvacso). 
Follow [this script](https://github.com/yizhou-wang/RODNet/blob/master/tools/prepare_dataset/reorganize_rod2021.sh) to reorganize files as below.

```
data_root
  - sequences
  | - train
  | | - <SEQ_NAME>
  | | | - IMAGES_0
  | | | | - <FRAME_ID>.jpg
  | | | | - ***.jpg
  | | | - RADAR_RA_H
  | | |   - <FRAME_ID>_<CHIRP_ID>.npy
  | | |   - ***.npy
  | | - ***
  | | 
  | - test
  |   - <SEQ_NAME>
  |   | - RADAR_RA_H
  |   |   - <FRAME_ID>_<CHIRP_ID>.npy
  |   |   - ***.npy
  |   - ***
  | 
  - annotations
  | - train
  | | - <SEQ_NAME>.txt
  | | - ***.txt
  | - test
  |   - <SEQ_NAME>.txt
  |   - ***.txt
  - calib
```

Convert data and annotations to `.pkl` files.
```commandline
python tools/prepare_dataset/prepare_data.py \
        --config configs/<CONFIG_FILE> \
        --data_root <DATASET_ROOT> \
        --split train,test \
        --out_data_dir data/<DATA_FOLDER_NAME>
```

## Train models

```commandline
python forward_train.py
```

## Model Weights

The optimal weights of D<sup>2</sup>FANet on the CRUW dataset are available for download from [`Key:54wb`](https://pan.baidu.com/s/1SkKBCTAxJy4TFeXwSnuEaQ).

###### If you encounter any issues with code or data reproduction, please contact me at jackychou_lab@126.com.