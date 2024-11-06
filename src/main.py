import numpy as np
import pandas as pd
import dml_estimators

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

#
# simulate_att_did
#
# Runs simulations of ATT-DID using both DML1 and DML2 for each given K
# and reports bias, MSE, and runtime as a printed table (for now)
#
def simulate_att_did(K: list, n_obs: int, n_sims: int):
    att_did_summary = []                                                        # Initialize array for reporting
    att_did = dml_estimators.ATTDID()                                           # Initialize ATTDID object

    #
    # Fit model with DML1 procedure
    #
    for k in K:
        simulation_results = np.zeros((n_sims, 2))                              # Allocate array of simulation results
        
        np.random.seed(123)                                                     # Set seed for generating seeds below
        seeds = np.random.randint(0, n_sims ** 2, n_sims)                       # Initialize array of seeds for simulations
        
        dml1_start = time.time()
        
        #
        # Run simulations in parallel
        #
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(att_did.run_simulation, i, k, 'dml1', n_obs, seeds[i]) for i in range(n_sims)]

            for future in as_completed(futures):
                i, result = future.result()
                simulation_results[i] = result                                  # Insert individual results at correct index as completed
                
        dml1_end = time.time()
        
        #
        # Append simulation results for this value of K to the data frame of reports
        #
        dml1_summary = ['dml1', k] + list(np.average(simulation_results, axis=0)) + [dml1_end - dml1_start]
        att_did_summary.append(dml1_summary)

    #
    # Fit model with DML2 procedure
    #
    for k in K:
        simulation_results = np.zeros((n_sims, 2))                              # Allocate array of simulation results
        
        np.random.seed(234)                                                     # Set seed for generating seeds below
        seeds = np.random.randint(0, n_sims ** 2, n_sims)                       # Initialize array of seeds for simulations
        
        dml2_start = time.time()
        
        #
        # Run simulations in parallel
        #
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(att_did.run_simulation, i, k, 'dml2', n_obs, seeds[i]) for i in range(n_sims)]

            for future in as_completed(futures):
                i, result = future.result()
                simulation_results[i] = result                                  # Insert individual results at correct index as completed
                
        dml2_end = time.time()
        
        #
        # Append simulation results for this value of K to the data frame of reports
        #
        dml2_summary = ['dml2', k] + list(np.average(simulation_results, axis=0)) + [dml2_end - dml2_start]
        att_did_summary.append(dml2_summary)

    #
    # Format results as pandas DataFrame
    #
    att_did_df = pd.DataFrame(att_did_summary, columns=['procedure', 'n_folds', 'bias', 'mse', 'runtime'])
    att_did_df.set_index(['procedure', 'n_folds'], inplace=True)

    #
    # Print results
    #
    print(f'\n[ATT-DID] n_obs={n_obs} n_sims={n_sims}\n')
    print(att_did_df)

#
# simulate_att_did
#
# Runs simulations of LATE using both DML1 and DML2 for each given K
# and reports bias, MSE, and runtime as a printed table (for now)
#
def simulate_late(K: list, n_obs: int, n_sims: int):
    late_summary = []
    late = dml_estimators.ATTDID()

    #
    # Fit model with DML1 procedure
    #
    for k in K:
        simulation_results = np.zeros((n_sims, 2))                              # Allocate array of simulation results
        
        np.random.seed(123)                                                     # Set seed for generating seeds below
        seeds = np.random.randint(0, n_sims ** 2, n_sims)                       # Initialize array of seeds for simulations
        
        dml1_start = time.time()
        
        #
        # Run simulations in parallel
        #
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(late.run_simulation, i, k, 'dml1', n_obs, seeds[i]) for i in range(n_sims)]

            for future in as_completed(futures):
                i, result = future.result()
                simulation_results[i] = result                                  # Insert individual results at correct index as completed
                
        dml1_end = time.time()
        
        dml1_summary = ['dml1', k] + list(np.average(simulation_results, axis=0)) + [dml1_end - dml1_start]
        late_summary.append(dml1_summary)

    #
    # Fit model with DML2 procedure
    #
    for k in K:
        simulation_results = np.zeros((n_sims, 2))                              # Allocate array of simulation results
        
        np.random.seed(456)                                                     # Set seed for generating seeds below
        seeds = np.random.randint(0, n_sims ** 2, n_sims)                       # Initialize array of seeds for simulations
        
        dml2_start = time.time()
        
        #
        # Run simulations in parallel
        #
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(late.run_simulation, i, k, 'dml2', n_obs, seeds[i]) for i in range(n_sims)]

            for future in as_completed(futures):
                i, result = future.result()
                simulation_results[i] = result                                  # Insert individual results at correct index as completed
                
        dml2_end = time.time()
        
        dml2_summary = ['dml2', k] + list(np.average(simulation_results, axis=0)) + [dml2_end - dml2_start]
        late_summary.append(dml2_summary)

    #
    # Format results as pandas DataFrame
    #
    late_df = pd.DataFrame(late_summary, columns=['procedure', 'n_folds', 'bias', 'mse', 'runtime'])
    late_df.set_index(['procedure', 'n_folds'], inplace=True)

    #
    # Print results
    #
    print(f'\n[LATE] n_obs={n_obs} n_sims={n_sims}\n')
    print(late_df)

def main():
    
    #
    # Define parameters
    #
    K = [2, 5]
    n_obs = 1000
    n_sims = 16

    #
    # Run simulations and print report
    #
    simulate_att_did(K, n_obs, n_sims)
    simulate_late(K, n_obs, n_sims)

if __name__ == '__main__':
    main()
