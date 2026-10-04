"""
    Adapted from github.com/kyrochi/...

    This script saturates available GPU compute.
"""
import numpy as np
import os
import pandas as pd
import random
import subprocess
import threading
import time
import torch
from collections import deque

def gather_df(df_dir: str) -> pd.DataFrame:
    # walk the directory and create a df from all the saved sub dfs
    df_list = []
    for root, dirs, files in os.walk(df_dir):
        for file in files:
            if file.endswith('.df'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                df_list.append(df)
    if df_list:
        df = pd.concat(df_list, ignore_index=True)        
        return df
    else:
        raise ValueError(f"No .df files found in directory: {df_dir}")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_command_on_gpu_capture_output(gpu_id, command, job_id, total_jobs, thread_number, last_output):
    try:
        process = subprocess.Popen(
            f"CUDA_VISIBLE_DEVICES={gpu_id} {command}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                last_output[thread_number] = output.strip()
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            last_output[thread_number] = f"Error: {stderr.strip()}"
            return False
        return True
    except Exception as e:
        last_output[thread_number] = f"Exception: {e}"
        return False

def manage_gpu_jobs(gpu_ids, commands, concurrency_per_gpu=1):
    total_jobs = len(commands)
    command_queue = deque(commands)
    gpu_states = {gpu_id: 0 for gpu_id in gpu_ids}  # Count of running jobs per GPU
    running_threads = {}  # job_id -> thread object
    job_id_counter = 0
    thread_number_counter = 0
    thread_info = {} # thread_number -> {'gpu_id': int, 'job_id': int}
    last_output = {} # thread_number -> last line of output

    start_time = time.time()

    def display_status():
        print("\n" + "=" * 100)
        print(f"{'Thread':<8} | {'GPU':<3} | {'Job':<7} | Latest Output")
        print("-" * 100)
        for thread_num in sorted(thread_info.keys()):
            info = thread_info[thread_num]
            gpu = info['gpu_id']
            job_id = info['job_id']
            output = last_output.get(thread_num, "No output yet")
            print(f"{thread_num:<8} | {gpu:<3} | {job_id + 1:<3}/{total_jobs:<3} | {output}")
        print("=" * 100 + "\n")
        print(f"Total runtime: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
        print(f"Total Running threads: {len(running_threads)}")
        print("=" * 100 + "\n")

    while command_queue or any(count > 0 for count in gpu_states.values()):
        # Check for finished threads
        finished_job_ids = [job_id for job_id, thread in running_threads.items() if not thread.is_alive()]
        for job_id in finished_job_ids:
            thread = running_threads.pop(job_id)
            thread_num = None
            for tn, info in thread_info.items():
                if info['job_id'] == job_id:
                    thread_num = tn
                    break
            if thread_num is not None:
                gpu_id = thread_info[thread_num]['gpu_id']
                del thread_info[thread_num]
                if thread_num in last_output:
                    del last_output[thread_num]
                gpu_states[gpu_id] -= 1

        # Assign new jobs to GPUs with available slots
        for gpu_id in gpu_ids:
            while gpu_states[gpu_id] < concurrency_per_gpu and command_queue:
                command = command_queue.popleft()
                job_id = job_id_counter
                job_id_counter += 1
                thread_number = thread_number_counter + 1
                thread_number_counter += 1
                thread = threading.Thread(
                    target=run_command_on_gpu_capture_output,
                    args=(gpu_id, command, job_id, total_jobs, thread_number, last_output)
                )
                thread_info[thread_number] = {'gpu_id': gpu_id, 'job_id': job_id}
                running_threads[job_id] = thread
                gpu_states[gpu_id] += 1
                thread.start()

        display_status()
        time.sleep(1)

    print("All commands have been executed.")

def get_command(
    out_dir,
    n_head,
    n_embd,
    learning_rate,
    weight_decay,
    mup_multiplier,
    seed,
    id,
):
    command = (
        f"python hp_train.py "
        f"--out_dir={out_dir} "
        f"--wandb_log=True "
        f"--wandb_project='mu-transfer-multi-seed-large' "
        f"--eval_interval=1000 "
        f"--log_interval=1 "
        f"--eval_iters=50 "
        f"--eval_only=False "
        f"--init_from='scratch' "
        f"--dataset='shakespeare_char' "
        f"--gradient_accumulation_steps=1 "
        f"--batch_size=256 "
        f"--block_size=1024 "
        f"--n_layer=3 "
        f"--n_head={n_head} "
        f"--n_embd={n_embd} "
        f"--dropout=0.0 "
        f"--bias=False "
        f"--init_std=0.02 "
        f"--learning_rate={learning_rate} "
        f"--max_iters=6000 "
        f"--weight_decay={weight_decay} "
        f"--beta1=0.9 "
        f"--beta2=0.95 "
        f"--grad_clip=1.0 "
        f"--decay_lr=False "
        f"--mup=True "
        f"--mup_multiplier={mup_multiplier} "
        f"--seed={seed} "
        f"--backend='nccl' "
        f"--device='cuda' "
        f"--dtype='float32' "
        f"--compile=False "
        f"--coord_check=False"
    )
    return command

if __name__ == "__main__":
    import argparse
    import datetime
    import numpy as np
    import random

    MAX_THREADS = 128

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--concurrency_per_gpu', type=int)
    # args = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed')
    args = parser.parse_args()

    embedding_dimensions = [256, 512, 1024, 2048]
    etas = [10**p for p in np.linspace(-5, -1, 18)]
    lambda_mults = [2**p for p in np.linspace(-4, 3, 18)]

    head_size = 64
    base_dim = 256

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"mu-transfer-char-2/{now}"
    os.makedirs(out_dir, exist_ok=True)

    commands = []
    command_count = 0
    for seed in [args.seed]:
        for emb in embedding_dimensions:
            for eta in etas:
                for lambda_m in lambda_mults:
                    mup_multiplier = emb // base_dim
                    n_heads = emb // head_size  
                    weight_decay = lambda_m # Don't divide by eta, we handle this using the muP code!
                    
                    command = get_command(
                        out_dir=out_dir,
                        n_head=n_heads,
                        n_embd=emb,
                        learning_rate=eta,
                        weight_decay=weight_decay,
                        mup_multiplier=mup_multiplier,
                        seed=seed,
                        id=command_count
                    )
                    commands.append(command)
                    command_count += 1

    random.shuffle(commands)

    manage_gpu_jobs(
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7], 
        commands=commands,
        concurrency_per_gpu=2
    )

