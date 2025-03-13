

### <div align="center">👉 PlanGen: Towards Unified Layout Planning and Image Generation in Auto-Regressive Vision Language Models<div> 
<!-- ### <div align="center"> 💥 Arxiv 2025！ <div>  -->
#### <div align="center"> Runze He, Bo Cheng, Yuhang Ma, Qingxiang Jia,  Shanyuan Liu, Ao Ma, Xiaoyu Wu, Liebucha Wu, Dawei Leng†, Yuhui Yin(✝Corresponding Author) <div>

<div align="center">
  <a href="https://360cvgroup.github.io/PlanGen/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2410.14324"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:PlanGen&color=red&logo=arxiv"></a> &ensp;
  <!-- <a href=""><img src="https://img.shields.io/static/v1?label=App&message=ComfyUI&&color=green"></a> &ensp; -->
</div>

---
## 🔥 News 
- **[2025/3/14]** We initialized this github repository and released the code.
- **[2025/3/14]** We released the paper of [PlanGen](https://arxiv.org/abs/2410.14324).

<!-- ## 🕓 Schedules
- **[Temporary uncertainty]** We plan to release the 2nd generation HiCo(more lightweight). -->

<!-- ## 💻 Quick Demos
Image demos can be found on the [HiCo](https://360cvgroup.github.io/HiCo_T2I/). Some of them are contributed by the community. You can customize your own personalized generation using the following reasoning code. -->

## 🔧 Quick Start
<!-- ### 0. Experimental environment -->
<!-- We tested our inference code on a machine with a 24GB 3090 GPU and CUDA environment version 12.1. -->

### 1. Setup repository and environment
```
git clone https://github.com/360CVGroup/PlanGen.git
cd PlanGen
conda create -n plangen python=3.10
conda activate plangen
pip install -r requirements.txt
```
### 2. Prepare the models
```
git lfs install

# HiCo checkpoint
git clone https://huggingface.co/qihoo360/HiCo_T2I models/controlnet
```
### 3. Customize your own creation
```
# layout2image generation
python train.py --cfg project/plangen/cfg/uni/h_text_ump+oimsam.py --opt test=True resume=/home/jovyan/boomcheng-data-shcdt/herunze/code/base/project/janus/out/h_text_ump+oimsam/checkpoint-200000 test_data.data_name='1k' test_data.task_type='uni'

# layout-image joint generation
python train.py --cfg project/plangen/cfg/uni/h_text_ump+oimsam.py --opt test=True resume=/home/jovyan/boomcheng-data-shcdt/herunze/code/base/project/janus/out/h_text_ump+oimsam/checkpoint-200000 test_data.data_name='1k' test_data.task_type='uni_2stage'

# image layout understanding
python train.py --cfg project/plangen/cfg/uni/h_text_ump+oimsam.py --opt test=True resume=/home/jovyan/boomcheng-data-shcdt/herunze/code/base/project/janus/out/h_text_ump+oimsam/checkpoint-200000 test_data.data_name='1k' test_data.task_type='mmu'

# object removal
python train.py --cfg project/plangen/cfg/uni/h_text_ump+oimsam.py --opt test=True resume=/home/jovyan/boomcheng-data-shcdt/herunze/code/base/project/janus/out/h_text_ump+oimsam/checkpoint-200000 test_data.data_name='coco' use_teacher_forcing=True pad_edit_box=0.1 use_neg_box=True trans_data_to_rm=True ## rm

# layout-guided image editing
python train.py --cfg project/plangen/cfg/uni/h_text_ump+oimsam.py --opt test=True resume=/home/jovyan/boomcheng-data-shcdt/herunze/code/base/project/janus/out/h_text_ump+oimsam/checkpoint-200000 test_data.data_name='edit_coco' use_teacher_forcing=True pad_edit_box=0.1 use_neg_box=False ## edit
```
## 🔥 Train

```
python train.py --cfg project/plangen/cfg/uni/h_text_ump+oimsam.py
```

<!-- ## BibTeX
```
@misc{cheng2024hicohierarchicalcontrollablediffusion,
      title={HiCo: Hierarchical Controllable Diffusion Model for Layout-to-image Generation}, 
      author={Bo Cheng and Yuhang Ma and Liebucha Wu and Shanyuan Liu and Ao Ma and Xiaoyu Wu and Dawei Leng and Yuhui Yin},
      year={2024},
      eprint={2410.14324},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.14324}, 
}
``` -->
## License
This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).

