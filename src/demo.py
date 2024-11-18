import dml_estimators
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def demo_summarized():
    
    #
    # Define parameters
    #
    K = [2, 5]                                                                  # Simulate k=2 and k=5 in this demo
    n_obs = 1000                                                                # Generate 1000 observations using each model's DGP
    n_sims = 16                                                                 # Simulate 16 times for each procedure and each k-value
    
    #
    # Initialize model objects
    #
    att_did_regressor = RandomForestRegressor(n_estimators=300, max_depth=7, max_features=3, min_samples_leaf=3)
    att_did_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, max_features=4, min_samples_leaf=7)
    att_did = dml_estimators.ATTDID(regressor=att_did_regressor, classifier=att_did_classifier)    
    
    late_regressor = RandomForestRegressor(n_estimators=300, max_depth=7, max_features=3, min_samples_leaf=3)
    late_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, max_features=4, min_samples_leaf=7)
    late = dml_estimators.LATE(regressor=late_regressor, classifier=late_classifier)

    #
    # Run simulations and report
    #
    att_did_results = att_did.simulate(K, n_obs, n_sims, verbose=False)
    
    print(f'\n[ATT-DID] n_obs={n_obs} n_sims={n_sims}\n')                       # Print ATT-DID results to console
    print(att_did_results)
    
    att_did_results.to_csv('results/summarized_att_did.csv')                    # Output ATT-DID results to csv
    
    late_results = late.simulate(K, n_obs, n_sims, verbose=False)
    
    print(f'\n[LATE] n_obs={n_obs} n_sims={n_sims}\n')                          # Print LATE results to console
    print(late_results)
    
    late_results.to_csv('results/summarized_late.csv')                          # Output LATE results to csv


def demo_verbose():
    #
    # Define parameters
    #
    K = [2, 5]                                                                  # Simulate k=2 and k=5 in this demo
    n_obs = 1000                                                                # Generate 1000 observations using each model's DGP
    n_sims = 16                                                                 # Simulate 16 times for each procedure and each k-value
    
    #
    # Initialize model objects
    #
    att_did_regressor = RandomForestRegressor(n_estimators=300, max_depth=7, max_features=3, min_samples_leaf=3)
    att_did_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, max_features=4, min_samples_leaf=7)
    att_did = dml_estimators.ATTDID(regressor=att_did_regressor, classifier=att_did_classifier)
    
    late_regressor = RandomForestRegressor(n_estimators=300, max_depth=7, max_features=3, min_samples_leaf=3)
    late_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, max_features=4, min_samples_leaf=7)
    late = dml_estimators.LATE(regressor=late_regressor, classifier=late_classifier)

    #
    # Run simulations and report
    #
    att_did_results = att_did.simulate(K, n_obs, n_sims, verbose=True)
    
    print(f'\n[ATT-DID] n_obs={n_obs} n_sims={n_sims}\n')                       # Print ATT-DID results to console
    print(att_did_results)
    
    att_did_results.to_csv('results/verbose_att_did.csv')                       # Output ATT-DID results to csv
    
    late_results = late.simulate(K, n_obs, n_sims, verbose=True)
    
    print(f'\n[LATE] n_obs={n_obs} n_sims={n_sims}\n')                          # Print LATE results to console
    print(late_results)
    
    late_results.to_csv('results/verbose_late.csv')                             # Output LATE results to csv


def main():
    #
    # Run the demo with a summarized output
    #
    # demo_summarized()
    
    #
    # Run the demo with a verbose output
    #
    demo_verbose()
    

#
# This conditional statement is necessary for running parallel computing
#
if __name__ == '__main__':
    main()
