import os
import wandb
import datetime

def get_clime_paths(base_path, suffix):
    save_path = os.path.join(base_path, f'clime_{suffix}')
    log_path = os.path.join(base_path, f'clime_{suffix}_logs')
    save_path_model = os.path.join(base_path, f'clime_{suffix}_model')
    wandb_run_name = get_wandb(savepath=save_path)
    
    return save_path, log_path, save_path_model, wandb_run_name


def get_cogent_paths(base_path, suffix):
    save_path = os.path.join(base_path, f'cogent_{suffix}')
    log_path = os.path.join(base_path, f'cogent_{suffix}_logs')
    save_path_model = os.path.join(base_path, f'cogent_{suffix}_model')
    wandb_run_name = get_wandb(savepath=save_path)
    
    return save_path, log_path, save_path_model, wandb_run_name


def get_wandb(savepath):
    now = datetime.datetime.now()
    current_time = now.strftime("%d.%b.%Y-%-I:%M:%S%p")

    run_name = savepath.rstrip("/").split("/")[-1] + "-" + current_time 
    return run_name


hs_col = "Hate Speech"
cs_col = "Counterspeech"
category_col = "Category"
target_col = "Target"