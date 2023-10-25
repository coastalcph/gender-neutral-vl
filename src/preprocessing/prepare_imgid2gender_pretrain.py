import os
import sys
import json
import math
import concurrent
import spacy
import pandas as pd
from tqdm import tqdm
from bias.tools.utils import infer_gender
from concurrent.futures._base import TimeoutError
from itertools import repeat
from timeit import default_timer as timer

dataset = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3]

# Initializa variables
chunk_size = 1000
timeout = 1500
num_workers = min(12, os.cpu_count())
DATA_DICT = json.load(open(input_file, "r"))
image_ids = list(DATA_DICT.keys())
N = len(image_ids)
print(f"{N} unique images in {dataset} (train).")
num_chunks = math.ceil(N / chunk_size)
print(f"Total num chunks to be processed: {num_chunks}")

nlp = spacy.load("en_core_web_sm")


def stop_process_pool(executor: concurrent.futures.ProcessPoolExecutor):
    for pid, process in executor._processes.items():
        process.terminate()
        print(f"Killed forked process {pid} due to timeout")
    executor.shutdown(wait=False)
    print("Killed executor due to timeout")
    return


def task(key_id, nlp):
    caps = DATA_DICT[key_id]
    gender_ = infer_gender(caps, nlp)
    return gender_


results = []
start_time = timer()
for i in range(num_chunks):
    interval_start = timer()

    result = []
    chunk = image_ids[i * chunk_size : min(N, i * chunk_size + chunk_size)]

    executor = concurrent.futures.ProcessPoolExecutor(num_workers)

    result_futures = tqdm(
        executor.map(task, chunk, repeat(nlp), timeout=timeout, chunksize=32),
        total=len(chunk),
        desc=f"Segmenting chunk {i + 1}",
    )

    try:
        for future in result_futures:
            result.append(future)
        executor.shutdown(wait=True)
    except TimeoutError:
        print(f"Timeout after {timeout} seconds.")
        stop_process_pool(executor)

    if len(result) < len(chunk):
        for i in range(len(result), len(chunk)):
            print(f"Image {chunk[i]} failed with: Timeout error")

    results.extend(result)

    interval_end = timer()
    interval_time = interval_end - interval_start
    throughput = len(result) / interval_time
    print(f"Interval time: {interval_time:.2f}s")
    print(f"Throughput (img/s): {throughput:.4f}")

    end_time = timer()

    print(f"Processed examples: {len(results)}")
    print(f"Elapsed time: {end_time - start_time:.2f}s")

df = pd.DataFrame(
    {
        "img_id": list(DATA_DICT.keys()),
        "captions": pd.Series(list(DATA_DICT.values())),
        "gender": results,
    }
)
df.to_pickle(output_file)
print(f"File {output_file} written")
