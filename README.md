## CriDiff
> The official implement of MICCAI 2024 paper [CriDiff: Criss-cross Injection Diffusion Framework via Generative Pre-train for Prostate Segmentation.](https://arxiv.org/abs/2406.14186)
![Structure Figure](Figure/Structure.png)
# Environment Installation
```
conda create -n CriDiff python=3.8 -y
conda activate CriDiff
git clone https://github.com/LiuTingWed/CriDiff.git
cd CriDiff
pip install -r requirements.txt
```
## Datasets Preparation
### Download Datasets
4 datasets need download (NCI-ISBI, ProstateX, Promise12, CCH-TRUSPS) from:
\
[Google Driver](https://drive.google.com/file/d/1riv-XTmlrcI_VHFJ_a18LdKGogTKH8Za/view?usp=drive_link) | [Baidu Driver (6666)](https://pan.baidu.com/s/1Pq1L64Q6R86XBTdYvDpjYQ?pwd=6666)
\
**I'm not sure about the copyright status of these datasets. If you are the owner of these datasets, please submit an issue to let me know so that I can remove them accordingly.**
#### Check data branch like this:
![Data_branch](Figure/Data_branch.png)
\
The body and detail are generated by **extract_boundary/generate_body_detail.py.** 
\
Please check this **.py** for more details.

## Download Pre-train Weight
[Google Driver (PVT_b2)](https://drive.google.com/file/d/1snw4TYUCD5z4d3aaId1iBdw-yUKjRmPC/view)
## Training & Inference & Evaluation
### Generative pretrain
This stage relies on [accelerate](https://github.com/huggingface/accelerate), please install it and set it up.
\
``
python generative_pretrain/train_generator_accelerate.py --dataset_root xxx/DATASET_NAME/images/train
`` 
### Training
Before trainning, please check **--dataset_root, --cp_condition_net, --cp_stage1, --checkpoint_save_dir** in train.py
\
``
python -m torch.distributed.launch --nproc_per_node=2 train.py
`` 
### Why can't the model perform training and validation simultaneously?
The output of diffusion models is related to the randomly sampled noise: different noise leads to different outputs. I have not addressed the issue of fluctuating model performance between the training and validation stages, for detailed descriptions please refer to [this link](https://github.com/lucidrains/med-seg-diff-pytorch/issues/18). 
Therefore, I would recommend saving all checkpoints, and then using two separate GPUs for validation to ensure that others can also achieve consistent performance. Well, I hope someone smarter than me tell me why :-).
## Thanks 
This repository refer to [med-seg-diff-pytorch](https://github.com/lucidrains/med-seg-diff-pytorch) and [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch). Some very concise diffusion frameworks are helpful to me.
## Citation
```
@article{liu2024cridiff,
  title={CriDiff: Criss-cross Injection Diffusion Framework via Generative Pre-train for Prostate Segmentation},
  author={Liu, Tingwei and Zhang, Miao and Liu, Leiye and Zhong, Jialong and Wang, Shuyao and Piao, Yongri and Lu, Huchuan},
  journal={arXiv preprint arXiv:2406.14186},
  year={2024}
}
```
## Any questions please contact with tingweiliu@mail.dlut.edu.cn

