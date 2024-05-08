# Contrastive Gaussian Clustering

This repository contains the implementation associated with the paper "Contrastive Gaussian Clustering: Weakly Supervised 3D Scene Segmentation", click in the next link for more details.

> [**Contrastive Gaussian Clustering: Weakly Supervised 3D Scene Segmentation**]()
> Myrna C. Silva, Mahtab Dahaghin, Matteo Toso,  Alessio Del Bue
> Istituto Italiano di Tecnologia - IIT

<img  width="1000"  alt="image"  src='assets/teaser_github.png'>

## Standard Installation
Clone the repository locally
```bash
git clone https://github.com/lkeab/gaussian-grouping.git
cd contrastive-gaussians
```
We provide a conda environment setup file including all the dependencies. Create the conda environment `contrastive-gaussians` by running:
```bash
conda create -n contrastive-gaussians python=3.8 -y
conda activate contrastive-gaussians

conda install pytorch==1.12.1  torchvision==0.13.1  torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install plyfile==0.8.1
pip install tqdm scipy wandb opencv-python scikit-learn lpips

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```
You will also need the the GroundingDINO (and SAM) set up to select an object by text:
```bash
git clone https://github.com/hkchengrex/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
cd ..
```
