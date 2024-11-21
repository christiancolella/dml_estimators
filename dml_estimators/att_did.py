import numpy as np
import pandas as pd
import doubleml as dml

from dml_estimators.module import Module
from sklearn.base import clone

from typing import Union

class ATTDID(Module):
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
        """
        Generates n_obs of data according to the process defined in section 5.1
        of Velez (2024) referenced from Sant'Anna and Zhao (2020), stores
        dataset in the object's data property.
        """
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
    
    def setup_dml1(self, n_folds: int):
        assert self.data != None
        
        self.dml1 = dml.DoubleMLDID(self.data,
                                    ml_g=clone(self.regressor),
                                    ml_m=clone(self.classifier),
                                    n_folds=n_folds,
                                    dml_procedure='dml1')
    
    def setup_dml2(self, n_folds: int):
        assert self.data != None
        
        self.dml2 = dml.DoubleMLDID(self.data,
                                    ml_g=clone(self.regressor),
                                    ml_m=clone(self.classifier),
                                    n_folds=n_folds, 
                                    dml_procedure='dml2')
        
        return self
