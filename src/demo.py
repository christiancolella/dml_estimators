import dml_estimators
import pandas as pd

def main():
    
    #
    # Define parameters
    #
    K = [2, 5]                                                                  # Simulate k=2 and k=5 in this demo
    n_obs = 1000                                                                # Generate 1000 observations using each model's DGP
    n_sims = 16                                                                 # Simulate 16 times for each procedure and each k-value
    
    #
    # Initialize model objects
    #
    att_did = dml_estimators.ATTDID()
    late = dml_estimators.LATE()

    #
    # Run simulations and report
    #
    att_did_results = att_did.simulate(K, n_obs, n_sims)
    
    print(f'\n[ATT-DID] n_obs={n_obs} n_sims={n_sims}\n')                       # Print ATT-DID results to console
    print(att_did_results)
    
    att_did_results.to_csv('results/att_did.csv')                               # Output ATT-DID results to csv
    
    late_results = late.simulate(K, n_obs, n_sims)
    
    print(f'\n[LATE] n_obs={n_obs} n_sims={n_sims}\n')                          # Print LATE results to console
    print(late_results)
    
    late_results.to_csv('results/late.csv')                                     # Output LATE results to csv

#
# This conditional statement is necessary for running parallel computing
#
if __name__ == '__main__':
    main()
