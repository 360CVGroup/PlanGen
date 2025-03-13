# _base_ = './testset.py'

seed=0
dirname='janus'
output_dir=None
logging_dir='logging_dir'
find_unused_parameters=True
# janus_path="deepseek-ai/Janus-Pro-1B"
# layoutsam_path="HuiZhang0812/LayoutSAM"
# layoutsam_eval_path="HuiZhang0812/LayoutSAM-eval"
# sdxl_path="stabilityai/stable-diffusion-xl-base-1.0"
janus_path="/home/jovyan/boomcheng-data-shcdt/herunze/models/Janus-Pro-1B"
layoutsam_path="/home/jovyan/boomcheng-data-shcdt/herunze/datasets/LayoutSAM"
layoutsam_eval_path="/home/jovyan/boomcheng-data-shcdt/herunze/datasets/LayoutSAM-eval"
sdxl_path="/home/jovyan/boomcheng-data-shcdt/herunze/models/stable-diffusion-xl-base-1.0"
coco_200_path = '/home/jovyan/boomcheng-data-shcdt/herunze/code/base/PlanGen_coco_data'
# coco_200_path = 'coco_data'

system_cls_path='project.plangen.plangen_base'
working_dir='project/plangen/out'

train_data=[
    dict(task_type='t2i', data_name='hico', batch_size=8),
]
test_data=dict(task_type='t2i', data_name='hico', batch_size=1)
pin_memory=True
dataloader_num_workers=16

max_train_steps=1000000
checkpointing_steps=5000
validation_steps=5000
metric_steps=10000
max_val_len=3
max_test_len=20
use_metric=True
use_teacher_forcing=False
tune_token_when_lora=True

test=False
val=False
func=None
only_metric=False
sam_debug=False

gradient_accumulation_steps=1
checkpoints_total_limit=3
resume='latest'
report_to='tensorboard'

scale_lr=None
lr_scheduler='constant'
lr_warmup_steps=0
max_grad_norm=1.0
adam_beta1=0.9
adam_beta2=0.999
adam_epsilon=1e-08
adam_weight_decay=0.01
learning_rate=5e-5

mixed_precision='bf16'
# mixed_precision='fp16'
# mixed_precision=None
gradient_checkpointing_enable=False

use_numhw_tokens=False
use_textual=False

# is_edit=False
use_special_tokens=False
tuning_mode='all'
lora_rank=256
lora_alpha=128


local_rank=-1
val_batch_size=1


num_frames=25
height=576
width=1024

janus_hw=384

O1=False
O2=False
O3=False
O4=True
debug=False
no_full=False

num_inference_steps=25

noise_offset=None
conditioning_dropout_prob=None
unuse_image_emb=False
unet_tuning='temp' # all,lora

use_mmu_loss=False
use_centerhw=False
use_smooth_labels=False

plan_lr_scale=None
use_blank_token=False
dropout_grounding=0
dropout_caption=0

scale_emb_grad=None

eta_min=None

use_2d_rope=False

dataset_same=False

use_bg_box=False

is_edit=False

pad_edit_box=0
use_neg_box=True
trans_data_to_rm=False

use_grounding_in_user=False

oim_split=None
oim_start=0

neg_prompt='low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers.'

use_info=False
use_creati_detail=False

max_seq_len=700
debug_max_seq_len=None

test_start=0
creati_path=None
cal_creati=False
prefetch_factor=None

score_plan=False
score_creati=False

use_edit_uni=False
use_local_edit_loss=False
use_des_for_edit_region=False

use_random_one_box=False


beam_search=False

score_edit=False

gen=True

parallel_size=1

use_showo=False

cfg_weight=None

save_data=False