# layout2image generation
python train.py --cfg project/plangen/cfg/uni/h_text_ump+oimsam.py --opt test=True resume=/home/jovyan/boomcheng-data-shcdt/herunze/code/base/project/janus/out/h_text_ump+oimsam/checkpoint-200000 test_data.data_name='1k' test_data.task_type='uni'

# layout-image joint generation
python train.py --cfg project/plangen/cfg/uni/h_text_ump+oimsam.py --opt test=True resume=/home/jovyan/boomcheng-data-shcdt/herunze/code/base/project/janus/out/h_text_ump+oimsam/checkpoint-200000 test_data.data_name='1k' test_data.task_type='uni_2stage'

# image layout understanding
python train.py --cfg project/plangen/cfg/uni/h_text_ump+oimsam.py --opt test=True resume=/home/jovyan/boomcheng-data-shcdt/herunze/code/base/project/janus/out/h_text_ump+oimsam/checkpoint-200000 test_data.data_name='1k' test_data.task_type='mmu'

# object removal
python train.py --cfg project/plangen/cfg/uni/h_text_ump+oimsam.py --opt test=True resume=/home/jovyan/boomcheng-data-shcdt/herunze/code/base/project/janus/out/h_text_ump+oimsam/checkpoint-200000 test_data.data_name='rm_coco' use_teacher_forcing=True pad_edit_box=0.1 use_neg_box=True ## rm

# layout-guided image editing
python train.py --cfg project/plangen/cfg/uni/h_text_ump+oimsam.py --opt test=True resume=/home/jovyan/boomcheng-data-shcdt/herunze/code/base/project/janus/out/h_text_ump+oimsam/checkpoint-200000 test_data.data_name='edit_coco' use_teacher_forcing=True pad_edit_box=0.1 use_neg_box=False ## edit