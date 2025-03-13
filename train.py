import argparse
import logging
import math
import os
from rich import print
import datasets
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
import diffusers
from diffusers.optimization import get_scheduler
from mmengine.config import Config, DictAction
from accelerate import DistributedDataParallelKwargs as DDPK
from src.utils.funcs import *
import wandb

# torch.autograd.set_detect_anomaly(True)conda activate /home/jovyan/boomcheng-data-shcdt/herunze/omnigen; cd /home/jovyan/boomcheng-data-shcdt/herunze/code/base

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for System.")
    parser.add_argument("--cfg", type=str, default=None, metavar='FILE', required=True)
    parser.add_argument('--opt', nargs='+', action=DictAction)
    args = parser.parse_args()

    ############################
    cfg_file = args.cfg
    cfg = Config.fromfile(cfg_file)
    if args.opt is not None:
        cfg.merge_from_dict(args.opt)
    args = cfg
    ############################
    
    if args.output_dir is None:
        expname = cfg_file.rsplit('.',1)[0].rsplit('/',1)[-1]
        if args.dirname is None:
            args.dirname = cfg_file.rsplit('.',1)[0].rsplit('/',2)[-2]

        if args.working_dir is None:
            args.output_dir = os.path.join('./out', args.dirname, expname)
        else:
            args.output_dir = os.path.join(args.working_dir, expname)

        mkdir(args.output_dir)

    return args

def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs_handlers = []
    if args.find_unused_parameters:
        kwargs_handlers = [DDPK(find_unused_parameters=True)]
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=kwargs_handlers,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = get_logger(__name__, log_level="INFO")
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None: set_seed(args.seed)
    if accelerator.is_main_process: print(args)

    from importlib import import_module
    clas = getattr(import_module(args.system_cls_path), 'System')
    model = clas(args, accelerator=accelerator)
    model = accelerator.prepare(model)
    accelerator._models = []

    train_dataloader, train_dataset = accelerator.unwrap_model(model).setup_data(accelerator)
    train_dataset[0]

    if getattr(args, 'train_batch_size', None) is None:
        args.train_batch_size = sum([t['batch_size'] for t in args.train_data])

    if args.scale_lr: args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process: accelerator.init_trackers("train")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume is not None:
        global_step = accelerator.unwrap_model(model).resume(accelerator)

    if args.test:
        accelerator.unwrap_model(model).validation(global_step)
        return
    elif args.func:
        func = getattr(accelerator.unwrap_model(model), args.func)
        func(global_step)
        return
    
    params_lr, params = accelerator.unwrap_model(model).get_trainable_para_lr(accelerator)
    optimizer = torch.optim.AdamW(
        params_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, ncols=300)
    progress_bar.set_description("Steps")
    progress_bar.update(global_step)

    for epoch in range(first_epoch, args.num_train_epochs):
        accelerator.unwrap_model(model).train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(model):
                loss = model(batch)
                if len(loss) == 2:
                    loss, loss_dict = loss

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)

                accelerator.unwrap_model(model).post_grad()

                # names = []
                # for name, param in model.named_parameters():
                #     if param.grad is None:
                #         names.append(name)
                # save_jsonl('no_grad.jsonl', names)
                # import pdb;pdb.set_trace()

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        accelerator.unwrap_model(model).save_para(global_step, accelerator)

                # if (global_step % args.metric_steps == 0 or global_step == 1) and args.use_metric:
                #     if accelerator.is_main_process:
                #         accelerator.unwrap_model(model).validation(global_step, accelerator=accelerator, test_mode=True, val_num=args.max_val_len)

                # if global_step % args.validation_steps == 0:
                if (global_step % args.validation_steps == 0 or global_step == 1)  and args.use_metric:
                    if accelerator.is_main_process:
                        accelerator.unwrap_model(model).validation(global_step, accelerator=accelerator)

            loss_dict.update(out_dir=args.output_dir)
            logs = {"loss": loss.detach().item(), **loss_dict, "lr": lr_scheduler.get_last_lr()[0]}

            accelerator.log(logs, step=global_step)

            progress_bar.set_postfix(**logs)

    global_step += 1
    accelerator.unwrap_model(model).validation(global_step)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()

    wandb.finish()

if __name__ == "__main__":
    # import torch.autograd.profiler as profiler
    # from torch.autograd.profiler import ProfilerActivity
    # with torch.profiler.profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
    # ) as prof:
    main()