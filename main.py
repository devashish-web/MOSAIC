import time
import torch
import tracemalloc

from args import args
from utils.taskflow import TaskFlow
from utils.set_seed import set_random_seed


if args.seed is not None:
    set_random_seed(args.seed)

tracemalloc.start()

start_time = time.perf_counter()

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

task = TaskFlow(args)
task.run()

end_time = time.perf_counter()
total_time = end_time - start_time

current, peak_cpu = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Total end-to-end time: {total_time:.2f} s")
print(f"Peak CPU memory usage: {peak_cpu / (1024 ** 2):.2f} MB")

if torch.cuda.is_available():
    peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Peak GPU memory usage: {peak_gpu_mem_mb:.2f} MB")
