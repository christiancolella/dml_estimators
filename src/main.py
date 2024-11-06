import numpy as np
import pandas as pd
import dml_estimators

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def simulate_att_did(K: list, n_obs: int, n_sims: int):
    att_did_summary = []
    att_did = dml_estimators.ATTDID()

    for k in K:
        simulation_results = np.zeros((n_sims, 2))
        seeds = np.random.randint(0, n_sims ** 2, n_sims)
        
        dml1_start = time.time()
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(att_did.run_simulation, i, k, 'dml1', n_obs, seeds[i]) for i in range(n_sims)]

            for future in as_completed(futures):
                i, result = future.result()
                simulation_results[i] = result
                
        dml1_end = time.time()
        
        dml1_summary = ['dml1', k] + list(np.average(simulation_results, axis=0)) + [dml1_end - dml1_start]
        att_did_summary.append(dml1_summary)
        
    for k in K:
        simulation_results = np.zeros((n_sims, 2))
        seeds = np.random.randint(0, n_sims ** 2, n_sims)
        
        dml2_start = time.time()
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(att_did.run_simulation, i, k, 'dml2', n_obs, seeds[i]) for i in range(n_sims)]

            for future in as_completed(futures):
                i, result = future.result()
                simulation_results[i] = result
                
        dml2_end = time.time()
        
        dml2_summary = ['dml2', k] + list(np.average(simulation_results, axis=0)) + [dml2_end - dml2_start]
        att_did_summary.append(dml2_summary)

    att_did_df = pd.DataFrame(att_did_summary, columns=['procedure', 'n_folds', 'bias', 'mse', 'runtime'])
    att_did_df.set_index(['procedure', 'n_folds'], inplace=True)

    print(f'\n[ATT-DID] n_obs={n_obs} n_sims={n_sims}\n')
    print(att_did_df)
    
def simulate_late(K: list, n_obs: int, n_sims: int):
    late_summary = []
    late = dml_estimators.ATTDID()

    for k in K:
        simulation_results = np.zeros((n_sims, 2))
        seeds = np.random.randint(0, n_sims ** 2, n_sims)
        
        dml1_start = time.time()
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(late.run_simulation, i, k, 'dml1', n_obs, seeds[i]) for i in range(n_sims)]

            for future in as_completed(futures):
                i, result = future.result()
                simulation_results[i] = result
                
        dml1_end = time.time()
        
        dml1_summary = ['dml1', k] + list(np.average(simulation_results, axis=0)) + [dml1_end - dml1_start]
        late_summary.append(dml1_summary)
        
    for k in K:
        simulation_results = np.zeros((n_sims, 2))
        seeds = np.random.randint(0, n_sims ** 2, n_sims)
        
        dml2_start = time.time()
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(late.run_simulation, i, k, 'dml2', n_obs, seeds[i]) for i in range(n_sims)]

            for future in as_completed(futures):
                i, result = future.result()
                simulation_results[i] = result
                
        dml2_end = time.time()
        
        dml2_summary = ['dml2', k] + list(np.average(simulation_results, axis=0)) + [dml2_end - dml2_start]
        late_summary.append(dml2_summary)

    late_df = pd.DataFrame(late_summary, columns=['procedure', 'n_folds', 'bias', 'mse', 'runtime'])
    late_df.set_index(['procedure', 'n_folds'], inplace=True)

    print(f'\n[LATE] n_obs={n_obs} n_sims={n_sims}\n')
    print(late_df)

def main():
    K = [2, 5]
    n_obs = 1000
    n_sims = 16

    simulate_att_did(K, n_obs, n_sims)
    simulate_late(K, n_obs, n_sims)

if __name__ == '__main__':
    main()
