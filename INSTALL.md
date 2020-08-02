## Installation

### Requirements

- Linux (Windows is not officially supported)
- Python 3.5+ (Python 2 is not supported)
- PyTorch 1.1 or higher
- CUDA 9.0 or higher
- NCCL 2
- GCC 4.9 or higher
- [mmcv](https://github.com/open-mmlab/mmcv) with version number 0.2.13 or 0.2.14 

We have tested the following versions of OS and CUDA:

- OS: Ubuntu 16.04/18.04
- CUDA: 10.0


### Install mmdetection

a. Create a conda virtual environment and activate it.

```shell
conda create -n ppdet python=3.7 -y
conda activate ppdet
```

b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
```

c. Clone the PPDet repository.

```shell
git clone https://github.com/nerminsamet/ppdet.git
cd ppdet
```
d. Install mmcv

```shell
pip install mmcv==0.2.13
```

e. Install PPDet (other dependencies will be installed automatically).

```shell
python setup.py develop
# or "pip install -v -e ."
```

Note:

1. The git commit id will be written to the version number with step e, e.g. 1.0.rc. The version will also be saved in trained models.
It is recommended that you run step e each time you pull some updates from github. **If C/CUDA codes are modified, then step e is compulsory.**

2. Following the above instructions, PPDet is installed on `dev` mode, **any local modifications made to the code will take effect without the need to reinstall it** (unless you submit some commits and want to update the version number).


### Prepare datasets

It is recommended to symlink the dataset root to `$PPDet/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```
COCO_DIR
├── annotations
├── images
│    ├── train2017
│    ├── val2017
│    ├── test2017
```


### Scripts

[Here](https://gist.github.com/hellock/bf23cd7348c727d69d48682cb6909047) is
a script for setting up mmdetection with conda.

### Multiple versions

If there are more than one mmdetection on your machine, and you want to use them alternatively, the recommended way is to create multiple conda environments and use different environments for different versions.

Another way is to insert the following code to the main scripts (`train.py`, `test.py` or any other scripts you run)
```python
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
```
or run the following command in the terminal of corresponding folder.
```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```
