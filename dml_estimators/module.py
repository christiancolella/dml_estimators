import numpy as np
import pandas as pd
from scipy.stats import norm

from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from abc import ABC, abstractmethod
from typing import Union

class Module:
    def __init__(self):
        self.data = None
        self.dml1 = None
        self.dml2 = None
        
    @abstractmethod
    def generate_data(self, n_obs: int, seed: Union[None, int]):
        """
        Abstract method that implements a data generating process. Must be
        overriden by a subclass.

        Args:
            n_obs (int): Number of observations to generate.
            seed (Union[None, int]): The random seed to use for RNG.
        Returns:
            self: Result is stored in this object's data property.
        """
        pass
    
    @abstractmethod
    def setup_dml1(self, n_folds: int):
        """
        Abstract method for defining the DML1 model. Must be overriden by a
        subclass.

        Args:
            n_folds (int): The number of folds to use for training.
        Returns:
            self: Result is stored in this object's dml1 property.
        """
        pass
    
    @abstractmethod
    def setup_dml2(self, n_folds: int):
        """
        Abstract method for defining the DML1 model. Must be overriden by a
        subclass.

        Args:
            n_folds (int): The number of folds to use for training.
        Returns:
            self: Result is stored in this object's dml1 property.
        """
        pass
    
    def fit_dml1(self):
        """
        Fits the model stored in the object's dml1 property

        Returns:
            self: The DML1 model is fit inplace.
        """
        assert self.dml1 != None
        
        self.dml1.fit()
        return self
    
    def fit_dml2(self):
        """
        Fits the model stored in the object's dml2 property

        Returns:
            self: The DML2 model is fit inplace.
        """
        assert self.dml2 != None
        
        self.dml2.fit()
        return self
    
    def run_one_simulation(self, i: int, k: int, dml_procedure: str, n_obs: int, seed: Union[None, int]):
        """
        Execute a single step of the simulation--generates data and fits the
        appropriate DoubleML model according to the method's parameters.

        Args:
            i (int): The index of the individual simulation.
            k (int): The number of folds for training.
            dml_procedure (str): Either 'dml1' or 'dml2', specifies which DML
            procedure to use.
            n_obs (int): The number of observations to generate in the DGP
            seed (Union[None, int]): The random seed to use for generating data
            for this iteration.

        Returns:
            tuple: The simulation index and a pair containing the bias and MSE
            of the estimate from the simulation.
        """
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
    
    def simulate(self, K: list, n_obs: int, n_sims: int, verbose: bool = False) -> pd.DataFrame:
        """
        Runs simulations of the DoubleML model using both DML1 and DML2 for each
        given K and returns a pd.DataFrame reporting bias and MSE.

        Args:
            K (list): A list of K-values to simulate.
            n_obs (int): The number of observations to generate in each DGP.
            n_sims (int): The number of simulations to run for each k and
            each procedure.
            verbose (bool, optional): If true, reports the bias and MSE for
            each individual simulation. Otherwise, reports the averages of
            these statistics for each K and each procedure. Defaults to False.

        Returns:
            pd.DataFrame: A data frame of the simulation results.
        """
        
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
                futures = [executor.submit(self.run_one_simulation, i, k, 'dml1', n_obs, seeds[i]) for i in range(n_sims)]

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
                futures = [executor.submit(self.run_one_simulation, i, k, 'dml2', n_obs, seeds[i]) for i in range(n_sims)]

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
