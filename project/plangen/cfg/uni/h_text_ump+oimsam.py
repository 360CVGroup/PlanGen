_base_ = '../base.py'

train_data=[
    dict(task_type='uni', data_name=['hico_full', 'oim', 'sam'], batch_size=3),
    dict(task_type='mmu', data_name=['hico_full', 'oim', 'sam'], batch_size=3),
    dict(task_type='plan', data_name='layout', batch_size=2),
]
test_data=dict(task_type='uni', data_name='hico', batch_size=8)

use_textual=True
use_special_tokens=True

tuning_mode='stage3'

max_train_steps=200000

gradient_accumulation_steps=1