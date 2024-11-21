import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from dml_estimators import *

from typing import Union

def simulate_att_did(outfile: Union[None, str] = None, verbose: bool = False):
    """
    Demonstration of a simulation with the ATT-DID model.

    Args:
        outfile (Union[None, str], optional): A csv filepath to output
        simulation results to. Defaults to None.
        verbose (bool, optional): If true, reports the bias and MSE for
        each individual simulation. Otherwise, reports the averages of
        these statistics for each K and each procedure. Defaults to False.
    """
    
    #
    # Define parameters
    #
    K = [2, 5]                                                                  # Simulate k=2 and k=5 in this demo
    n_obs = 1000                                                                # Generate 1000 observations using each model's DGP
    n_sims = 16                                                                 # Simulate 16 times for each procedure and each k-value
    
    #
    # Initialize model objects
    #
    regressor = RandomForestRegressor(n_estimators=300, max_depth=7, max_features=3, min_samples_leaf=3)
    classifier = RandomForestClassifier(n_estimators=100, max_depth=5, max_features=4, min_samples_leaf=7)
    model = ATTDID(regressor=regressor, classifier=classifier)
    
    #
    # Run simulations and report
    #
    results = model.simulate(K, n_obs, n_sims, verbose=verbose)
    
    print(f'\n[ATT-DID] n_obs={n_obs} n_sims={n_sims}\n')                       # Print ATT-DID results to console
    print(results)
    print('\n')
    
    if outfile != None:
        results.to_csv(outfile)                                                 # Output ATT-DID results to csv

def simulate_late(outfile: Union[None, str] = None, verbose: bool = False):
    """
    Demonstration of a simulation with the LATE model.

    Args:
        outfile (Union[None, str], optional): A csv filepath to output
        simulation results to. Defaults to None.
        verbose (bool, optional): If true, reports the bias and MSE for
        each individual simulation. Otherwise, reports the averages of
        these statistics for each K and each procedure. Defaults to False.
    """
    
    #
    # Define parameters
    #
    K = [2, 5]                                                                  # Simulate k=2 and k=5 in this demo
    n_obs = 1000                                                                # Generate 1000 observations using each model's DGP
    n_sims = 16                                                                 # Simulate 16 times for each procedure and each k-value
    
    #
    # Initialize model objects
    #
    regressor = RandomForestRegressor(n_estimators=300, max_depth=7, max_features=3, min_samples_leaf=3)
    classifier = RandomForestClassifier(n_estimators=100, max_depth=5, max_features=4, min_samples_leaf=7)
    model = ATTDID(regressor=regressor, classifier=classifier)
    
    #
    # Run simulations and report
    #
    results = model.simulate(K, n_obs, n_sims, verbose=verbose)
    
    print(f'\n[ATT-DID] n_obs={n_obs} n_sims={n_sims}\n')                       # Print LATE results to console
    print(results)
    print('\n')
    
    if outfile != None:
        results.to_csv(outfile)                                                 # Output LATE results to csv
        