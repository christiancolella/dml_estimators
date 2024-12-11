import numpy as np
import pandas as pd
import doubleml as dml
from scipy.stats import norm

from dml_estimators.module import Module
from sklearn.base import clone

from typing import Union

class LATE(Module):
    def __init__(self, regressor, classifier):
        self.regressor = regressor
        self.classifier = classifier
    
    def generate_data(self, n_obs: int, seed: Union[None, int]):
        """
        Generates n_obs of data according to the process defined in section 5.2
        of Velez (2024) referenced from Hong and Nekipelov (2010), stores
        dataset in the object's data property.
        """
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
    
    def setup_dml1(self, n_folds: int):
        assert self.data != None
        
        self.dml1 = dml.DoubleMLIIVM(self.data,
                                     ml_g=clone(self.regressor),
                                     ml_m=clone(self.classifier),
                                     ml_r=clone(self.classifier),
                                     n_folds=n_folds,
                                     dml_procedure='dml1')
        
        return self
    
    def setup_dml2(self, n_folds: int):
        assert self.data != None
        
        self.dml2 = dml.DoubleMLIIVM(self.data,
                                     ml_g=clone(self.regressor),
                                     ml_m=clone(self.classifier),
                                     ml_r=clone(self.classifier),
                                     n_folds=n_folds,
                                     dml_procedure='dml2')
        
        return self
