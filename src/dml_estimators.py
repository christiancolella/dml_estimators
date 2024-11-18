import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import clone

import doubleml as dml

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import Union


#
# ATTDID
#
# Provides methods that implement the DGP for ATT-DID, set up and fit the
# DID model provided by the DoubleML library using both DML1 and DML2
# procedures, and report on simulations for different k-values
#
class ATTDID:
    regressor = None
    classifier = None
    
    data = None
    dml1 = None
    dml2 = None
    
    def __init__(self, regressor, classifier):
        self.regressor = regressor
        self.classifier = classifier
    
    #
    # generate_data
    #
    # Generates n_obs of data according to the process defined in section 5.1
    # of Velez (2024) referenced from Sant'Anna and Zhao (2020), stores
    # dataset in the object's data property
    #
    def generate_data(self, n_obs: int, seed: Union[None, int]):
        assert n_obs > 0
        
        np.random.seed(seed)
        
        def f_ps(X: np.ndarray):
            assert np.shape(X)[1] == 4
            
            T = 0.25 * np.array([-1, 0.5, -0.25, -0.1])
            return np.matmul(X, T)
        
        def p(X: np.ndarray):
            assert np.shape(X)[1] == 4
            
            E = np.exp(f_ps(X))
            return np.divide(E, 1 + E)
        
        def f_reg(X: np.ndarray):
            assert np.shape(X)[1] == 4
            
            T = np.array([6.85, 3.425, 3.425, 3.425], dtype=np.float64)
            return 210 + np.matmul(X, T)
        
        def v(X: np.ndarray, A: np.ndarray):
            epsilon = np.random.normal(0, 1, n_obs)
            return np.dot(A, f_reg(X)) + epsilon
        
        X = np.random.uniform(0, 1, (n_obs, 4)).astype(np.float64)
        A = np.random.binomial(1, p(X)).astype(np.float64)
        
        Y_0 = f_reg(X) + v(X, A) + np.random.normal(0, 1, n_obs)
        
        Y_11 = 2 * f_reg(X) + v(X, A) + np.random.normal(0, 1, n_obs)
        Y_10 = 2 * f_reg(X) + v(X, A) + np.random.normal(0, 1, n_obs)
        Y_1 = np.multiply(A, Y_11) + np.multiply(1 - A, Y_10)
        
        Y = Y_1 - Y_0
        
        Y = np.reshape(Y, (n_obs, 1))
        A = np.reshape(A, (n_obs, 1))
        
        df = pd.DataFrame(np.concat([Y, A, X], axis=-1), columns=['Y', 'A', 'X1', 'X2', 'X3', 'X4'], dtype=np.float64)
        self.data = dml.DoubleMLData(df, y_col='Y', d_cols='A', x_cols=['X1', 'X2', 'X3', 'X4'])
        
        return self
    
    #
    # setup_dml1
    #
    # Initializes the DID model from DoubleML using the DML1 procedure and the
    # data stored in the object's data property
    #
    def setup_dml1(self, n_folds: int):
        assert self.data != None
        
        self.dml1 = dml.DoubleMLDID(self.data,
                                    ml_g=clone(self.regressor),
                                    ml_m=clone(self.classifier),
                                    n_folds=n_folds, 
                                    dml_procedure='dml1')
    
    #
    # setup_dml2
    #
    # Initializes the DID model from DoubleML using the DML2 procedure and the
    # data stored in the object's data property
    #
    def setup_dml2(self, n_folds: int):
        assert self.data != None
        
        self.dml2 = dml.DoubleMLDID(self.data,
                                    ml_g=clone(self.regressor),
                                    ml_m=clone(self.classifier),
                                    n_folds=n_folds, 
                                    dml_procedure='dml2')
        
        return self
    
    #
    # fit_dml1
    #
    # Fits the model stored in the object's dml1 property
    #
    def fit_dml1(self):
        assert self.dml1 != None
        
        self.dml1.fit()
        return self
    
    
    #
    # fit_dml2
    #
    # Fits the model stored in the object's dml2 property
    #
    def fit_dml2(self):
        assert self.dml2 != None
        
        self.dml2.fit()
        return self
    
    #
    # run_simulation
    #
    # Execute a single step of the simulation--generates data, fits the DID
    # model, and returns the bias and MSE of the individual simulation
    #
    def run_simulation(self, i: int, k: int, dml_procedure: str, n_obs: int, seed: Union[None, int]):
        if dml_procedure == 'dml1':
            self.generate_data(n_obs=n_obs, seed=seed)
            self.setup_dml1(n_folds=k)
            self.fit_dml1()
            
            bias = self.dml1.coef[0]
            mse = self.dml1.se[0] ** 2
            
            return i, (bias, mse)
    
        else:
            assert dml_procedure == 'dml2'
            
            self.generate_data(n_obs=n_obs, seed=seed)
            self.setup_dml2(n_folds=k)
            self.fit_dml2()
            
            bias = self.dml2.coef[0]
            mse = self.dml2.se[0] ** 2
            
            return i, (bias, mse)
        
    #
    # simulate
    #
    # Runs simulations of ATT-DID using both DML1 and DML2 for each given K
    # and returns a pd.DataFrame containing bias, MSE, and runtime
    #
    # If verbose is set to true, outputs the results for each individual
    # simulation with runtime omitted. Otherwise, outputs the average result
    # for each K and each procedure
    #
    def simulate(self, K: list, n_obs: int, n_sims: int, verbose: bool = False) -> pd.DataFrame:
        
        #
        # Initialize data frame of results with appropriate number of columns
        # and data types
        #
        n_cols = 2 * len(K) * (n_sims if verbose else 1)
        df = pd.DataFrame({
            'a': pd.Series(np.zeros(n_cols, dtype=np.str_)),
            'b': pd.Series(np.zeros(n_cols, dtype=np.int8)),
            'c': pd.Series(np.zeros(n_cols, dtype=np.int8 if verbose else np.float64)),
            'd': pd.Series(np.zeros(n_cols, dtype=np.float64)),
            'e': pd.Series(np.zeros(n_cols, dtype=np.float64))
            })

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
                futures = [executor.submit(self.run_simulation, i, k, 'dml1', n_obs, seeds[i]) for i in range(n_sims)]

                for future in as_completed(futures):
                    i, result = future.result()
                    simulation_results[i] = result                                  # Insert individual results at correct index as completed
                    
            dml1_end = time.time()
            
            #
            # Append simulation results for this value of K to the data frame of reports
            #
            start_row = K.index(k) * (n_sims if verbose else 1)
            end_row = start_row + n_sims
            
            if verbose:
                df.iloc[start_row:end_row, 0] = 'dml1'
                df.iloc[start_row:end_row, 1] = k
                df.iloc[start_row:end_row, 2] = np.arange(n_sims).astype(np.int8)
                df.iloc[start_row:end_row, 3:] = simulation_results
                    
            else:
                df.iloc[start_row, 0] = 'dml1'
                df.iloc[start_row, 1] = k
                df.iloc[start_row, 2:] = np.concat([np.average(simulation_results, axis=0), np.array([dml1_end - dml1_start])], axis=-1)

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
                futures = [executor.submit(self.run_simulation, i, k, 'dml2', n_obs, seeds[i]) for i in range(n_sims)]

                for future in as_completed(futures):
                    i, result = future.result()
                    simulation_results[i] = result                                  # Insert individual results at correct index as completed
                    
            dml2_end = time.time()
            
            #
            # Append simulation results for this value of K to the data frame of reports
            #
            start_row = (len(K) + K.index(k)) * (n_sims if verbose else 1)
            end_row = start_row + n_sims
            
            if verbose:
                df.iloc[start_row:end_row, 0] = 'dml2'
                df.iloc[start_row:end_row, 1] = k
                df.iloc[start_row:end_row, 2] = np.arange(n_sims).astype(np.int8)
                df.iloc[start_row:end_row, 3:] = simulation_results
                    
            else:
                df.iloc[start_row, 0] = 'dml2'
                df.iloc[start_row, 1] = k
                df.iloc[start_row, 2:] = np.concat([np.average(simulation_results, axis=0), np.array([dml2_end - dml2_start])], axis=-1)

        #
        # Format results as pandas DataFrame
        #
        if verbose:
            df.columns = ['procedure', 'n_folds', 'sim_index', 'bias', 'mse']
            df.set_index(['procedure', 'n_folds', 'sim_index'], inplace=True)
            
            return df
        else:
            df.iloc[:, 2] = np.sqrt(n_obs) * np.abs(df.iloc[:, 2])
            df.iloc[:, 3] = n_obs * df.iloc[:, 3]
            
            df.columns = ['procedure', 'n_folds', '√(n_obs)*|bias|', 'n_obs*mse', 'runtime']
            df.set_index(['procedure', 'n_folds'], inplace=True)
            
            return df


#
# LATE
#
# Provides methods that implement the DGP for LATE, set up and fit the
# IIVM model provided by the DoubleML library using both DML1 and DML2
# procedures, and report on simulations for different k-values
#
class LATE:
    regressor = None
    classifier = None
    
    data = None
    dml1 = None
    dml2 = None
    
    def __init__(self, regressor, classifier):
        self.regressor = regressor
        self.classifier = classifier
    
    #
    # generate_data
    #
    # Generates n_obs of data according to the process defined in section 5.2
    # of Velez (2024) referenced from Hong and Nekipelov (2010), stores
    # dataset in the object's data property
    #
    def generate_data(self, n_obs: int, seed: Union[None, int]):
        assert n_obs > 0
        
        np.random.seed(seed)
        
        X = np.random.uniform(0, 1, n_obs).astype(np.float64)
        V = np.random.normal(0, 1, n_obs).astype(np.float64)
        
        D_1 = np.where(X + 0.5 >= V, 1, 0)
        D_0 = np.where(X - 0.5 >= V, 1, 0)
        
        lambdas = np.array([np.exp(1 + X / 2), np.exp(X / 2), 2 * np.ones((n_obs)), np.ones(n_obs)]).T
        XI = np.random.poisson(lambdas).astype(np.float64)
        
        Y_1 = XI[:, 0] + np.where(D_1 * D_0 == 1, XI[:, 2], 0) + np.where(D_1 * D_0 == 0, XI[:, 3], 0)
        Y_0 = XI[:, 1] + np.where(D_1 * D_0 == 1, XI[:, 2], 0) + np.where(D_1 * D_0 == 0, XI[:, 3], 0)
        
        Z = np.random.binomial(1, norm.cdf(X - 0.5)).astype(np.float64)
        D = Z * D_1 + (1 - Z) * D_0
        
        Y = D * Y_1 + (1 - D) * Y_0
        
        Y = np.reshape(Y, (n_obs, 1))
        Z = np.reshape(Z, (n_obs, 1))
        X = np.reshape(X, (n_obs, 1))
        D = np.reshape(D, (n_obs, 1))
        
        df = pd.DataFrame(np.concat([Y, Z, D, X], axis=-1), columns=['Y', 'Z', 'D', 'X'], dtype=np.float64)
        self.data = dml.DoubleMLData(df, y_col='Y', d_cols='D', x_cols='X', z_cols='Z')
        
        return self
    
    #
    # setup_dml1
    #
    # Initializes the IIVM model from DoubleML using the DML1 procedure and the
    # data stored in the object's data property
    #
    def setup_dml1(self, n_folds: int):
        assert self.data != None
        
        self.dml1 = dml.DoubleMLIIVM(self.data,
                                     ml_g=clone(self.regressor),
                                     ml_m=clone(self.classifier),
                                     ml_r=clone(self.classifier),
                                     n_folds=n_folds,
                                     dml_procedure='dml1')
        
        return self
    
    #
    # setup_dml2
    #
    # Initializes the IIVM model from DoubleML using the DML2 procedure and the
    # data stored in the object's data property
    #
    def setup_dml2(self, n_folds: int):
        assert self.data != None
        
        self.dml2 = dml.DoubleMLIIVM(self.data,
                                     ml_g=clone(self.regressor),
                                     ml_m=clone(self.classifier),
                                     ml_r=clone(self.classifier),
                                     n_folds=n_folds,
                                     dml_procedure='dml2')
        
        return self
    
    #
    # fit_dml1
    #
    # Fits the model stored in the object's dml1 property
    #
    def fit_dml1(self):
        assert self.dml1 != None
        
        self.dml1.fit()
        return self
    
    #
    # fit_dml2
    #
    # Fits the model stored in the object's dml2 property
    #
    def fit_dml2(self):
        assert self.dml2 != None
        
        self.dml2.fit()
        return self
    
    
    #
    # run_simulation
    #
    # Execute a single step of the simulation--generates data, fits the IIVM
    # model, and returns the bias and MSE of the individual simulation
    #
    def run_simulation(self, i: int, k: int, dml_procedure: str, n_obs: int, seed: Union[None, int]):
        if dml_procedure == 'dml1':
            self.generate_data(n_obs=n_obs, seed=seed)
            self.setup_dml1(n_folds=k)
            self.fit_dml1()
            
            bias = self.dml1.coef[0]
            mse = self.dml1.se[0] ** 2
            
            return i, (bias, mse)
    
        else:
            assert dml_procedure == 'dml2'
            
            self.generate_data(n_obs=n_obs, seed=seed)
            self.setup_dml2(n_folds=k)
            self.fit_dml2()
            
            bias = self.dml2.coef[0]
            mse = self.dml2.se[0] ** 2
            
            return i, (bias, mse)
        
    #
    # simulate
    #
    # Runs simulations of LATE using both DML1 and DML2 for each given K
    # and returns a pd.DataFrame containing bias, MSE, and runtime
    #
    # If verbose is set to true, outputs the results for each individual
    # simulation with runtime omitted. Otherwise, outputs the average result
    # for each K and each procedure
    #
    def simulate(self, K: list, n_obs: int, n_sims: int, verbose: bool = False) -> pd.DataFrame:
        
        #
        # Initialize data frame of results with appropriate number of columns
        # and data types
        #
        n_cols = 2 * len(K) * (n_sims if verbose else 1)
        df = pd.DataFrame({
            'a': pd.Series(np.zeros(n_cols, dtype=np.str_)),
            'b': pd.Series(np.zeros(n_cols, dtype=np.int8)),
            'c': pd.Series(np.zeros(n_cols, dtype=np.int8 if verbose else np.float64)),
            'd': pd.Series(np.zeros(n_cols, dtype=np.float64)),
            'e': pd.Series(np.zeros(n_cols, dtype=np.float64))
            })

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
                futures = [executor.submit(self.run_simulation, i, k, 'dml1', n_obs, seeds[i]) for i in range(n_sims)]

                for future in as_completed(futures):
                    i, result = future.result()
                    simulation_results[i] = result                                  # Insert individual results at correct index as completed
                    
            dml1_end = time.time()
            
            #
            # Append simulation results for this value of K to the data frame of reports
            #
            start_row = K.index(k) * (n_sims if verbose else 1)
            end_row = start_row + n_sims
            
            if verbose:
                df.iloc[start_row:end_row, 0] = 'dml1'
                df.iloc[start_row:end_row, 1] = k
                df.iloc[start_row:end_row, 2] = np.arange(n_sims).astype(np.int8)
                df.iloc[start_row:end_row, 3:] = simulation_results
                    
            else:
                df.iloc[start_row, 0] = 'dml1'
                df.iloc[start_row, 1] = k
                df.iloc[start_row, 2:] = np.concat([np.average(simulation_results, axis=0), np.array([dml1_end - dml1_start])], axis=-1)

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
                futures = [executor.submit(self.run_simulation, i, k, 'dml2', n_obs, seeds[i]) for i in range(n_sims)]

                for future in as_completed(futures):
                    i, result = future.result()
                    simulation_results[i] = result                                  # Insert individual results at correct index as completed
                    
            dml2_end = time.time()
            
            #
            # Append simulation results for this value of K to the data frame of reports
            #
            start_row = (len(K) + K.index(k)) * (n_sims if verbose else 1)
            end_row = start_row + n_sims
            
            if verbose:
                df.iloc[start_row:end_row, 0] = 'dml2'
                df.iloc[start_row:end_row, 1] = k
                df.iloc[start_row:end_row, 2] = np.arange(n_sims).astype(np.int8)
                df.iloc[start_row:end_row, 3:] = simulation_results
                    
            else:
                df.iloc[start_row, 0] = 'dml2'
                df.iloc[start_row, 1] = k
                df.iloc[start_row, 2:] = np.concat([np.average(simulation_results, axis=0), np.array([dml2_end - dml2_start])], axis=-1)

        #
        # Format results as pandas DataFrame
        #
        if verbose:
            df.columns = ['procedure', 'n_folds', 'sim_index', 'bias', 'mse']
            df.set_index(['procedure', 'n_folds', 'sim_index'], inplace=True)
            
            return df
        else:
            df.iloc[:, 2] = np.sqrt(n_obs) * np.abs(df.iloc[:, 2])
            df.iloc[:, 3] = n_obs * df.iloc[:, 3]
            
            df.columns = ['procedure', 'n_folds', '√(n_obs)*|bias|', 'n_obs*mse', 'runtime']
            df.set_index(['procedure', 'n_folds'], inplace=True)
            
            return df
