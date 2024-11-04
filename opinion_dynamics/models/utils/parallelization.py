import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

def get_total_score(parallel, function, n, num_runs):
    if parallel == 'sequential':
        return sequential(function, n, num_runs)
    elif parallel == 'pool':
        return pool(function, n, num_runs)
    elif parallel == 'process_pool_executor':
        return process_pool_executor(function, n, num_runs)
    else:
        raise ValueError("Invalid parallelization method")

def sequential(function, n, num_runs):
    results = []
    for _ in range(num_runs):
        results.append(function(n=n))
    return results

def pool(function, n, num_runs):
    with Pool() as pool:
        results = pool.starmap(function, [(n,) for _ in range(num_runs)])
    print(results)
    return results

def process_pool_executor(function, n, num_runs):
    with ProcessPoolExecutor() as executor:
        results = executor.map(function, [n] * num_runs)
    return results