import numpy as np
import pandas as pd
import doubleml as dml

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import norm

from typing import Union
    
class ATTDID:
    data = None
    
    regressor = RandomForestRegressor(n_estimators=300, max_depth=7, max_features=3, min_samples_leaf=3)
    classifier = RandomForestClassifier(n_estimators=100, max_depth=5, max_features=4, min_samples_leaf=7)
    
    dml1 = None
    dml2 = None
    
    def generate_data(self, n_obs: int, seed: Union[None, int]):
        assert n_obs > 0
        
        df = pd.DataFrame([], columns=['Y', 'A', 'X1', 'X2', 'X3', 'X4'], dtype=np.float64)
        np.random.seed(seed)
        
        def f_ps(x: list):
            assert len(x) == 4
            return 0.25 * (-x[0] + 0.5 * x[1] - 0.25 * x[2] - 0.1 * x[3])
        
        def p(x: list):
            assert len(x) == 4
            return np.exp(f_ps(x)) / (1 + np.exp(f_ps(x)))
        
        def f_reg(x: list):
            assert len(x) == 4
            return 210 + 6.85 * x[0] + 3.425 * (x[1] + x[2] + x[3])
        
        def v(x: list, a: np.float64):
            assert len(x) == 4
            return a * f_reg(x) + np.random.normal(0, 1)
        
        X = np.random.uniform(0, 1, (n_obs, 4)).astype(np.float64)
            
        for x in X:
            a = np.random.binomial(1, p(x))
            
            y0 = f_reg(x) + v(x, a) + np.random.normal(0, 1)
            y1 = 2 * f_reg(x) + v(x, a) + np.random.normal(0, 1)
            
            y = y1 - y0
                        
            df.loc[len(df.index)] = np.array([y, a, x[0], x[1], x[2], x[3]])
        
        self.data = dml.DoubleMLData(df, y_col='Y', d_cols='A', x_cols=['X1', 'X2', 'X3', 'X4'])
        
        return self
    
    def setup_dml1(self, n_folds: int):
        assert self.data != None
        
        self.dml1 = dml.DoubleMLDID(self.data,
                                    ml_g=self.regressor,
                                    ml_m=self.classifier,
                                    n_folds=n_folds, 
                                    dml_procedure='dml1')
    
    def setup_dml2(self, n_folds: int):
        assert self.data != None
        
        self.dml2 = dml.DoubleMLDID(self.data,
                                    ml_g=self.regressor,
                                    ml_m=self.classifier,
                                    n_folds=n_folds, 
                                    dml_procedure='dml2')
        
        return self
    
    def fit_dml1(self):
        assert self.dml1 != None
        
        self.dml1.fit()
        return self
    
    def fit_dml2(self):
        assert self.dml2 != None
        
        self.dml2.fit()
        return self
    
    def run_simulation(self, i: int, k: int, dml_procedure: str, n_obs: int, seed: Union[None, int]):
        if dml_procedure == 'dml1':
            self.generate_data(n_obs=n_obs, seed=seed)
            self.setup_dml1(n_folds=k)
            self.fit_dml1()
            
            bias = self.dml1.coef[0]
            mse = (self.dml1.se[0] ** 2) * (n_obs - 1)
            
            return i, (bias, mse)
    
        else:
            assert dml_procedure == 'dml2'
            
            self.generate_data(n_obs=n_obs, seed=seed)
            self.setup_dml2(n_folds=k)
            self.fit_dml2()
            
            bias = self.dml2.coef[0]
            mse = (self.dml2.se[0] ** 2) * (n_obs - 1)
            
            return i, (bias, mse)
    
class LATE:
    data = None
    
    regressor = RandomForestRegressor(n_estimators=300, max_depth=7, max_features=3, min_samples_leaf=3)
    classifier = RandomForestClassifier(n_estimators=100, max_depth=5, max_features=4, min_samples_leaf=7)
    
    dml1 = None
    dml2 = None
    
    def generate_data(self, n_obs: int, seed: Union[None, int]):
        assert n_obs > 0
        
        df = pd.DataFrame([], columns=['Y', 'Z', 'D', 'X'], dtype=np.float64)
        np.random.seed(seed)
        
        x = np.random.uniform(0, 1, n_obs).astype(np.float64)
        v = np.random.normal(0, 1, n_obs).astype(np.float64)
        
        for i in range(n_obs):
            d_1 = 1 if x[i] + 0.5 >= v[i] else 0
            d_0 = 1 if x[i] - 0.5 >= v[i] else 0
            
            xi_1 = np.random.poisson(np.exp(x[i] / 2))
            xi_2 = np.random.poisson(np.exp(x[i] / 2))
            xi_3 = np.random.poisson(2)
            xi_4 = np.random.poisson(1)
            
            y_1 = xi_1 + (xi_3 if d_1 == 1 and d_0 == 1 else 0) + (xi_4 if d_1 == 0 and d_0 == 0 else 0)
            y_0 = xi_2 + (xi_3 if d_1 == 1 and d_0 == 1 else 0) + (xi_4 if d_1 == 0 and d_0 == 0 else 0)
            
            z = np.random.binomial(1, norm.cdf(x[i] - 0.5))
            d = z * d_1 + (1 - z) * d_0
            
            y = d * y_1 + (1 - d) * y_0
            
            df.loc[len(df.index)] = np.array([y, z, d, x[i]])
        
        self.data = dml.DoubleMLData(df,
                                     y_col='Y',
                                     d_cols='D',
                                     x_cols='X',
                                     z_cols='Z')
        
        return self
    
    def setup_dml1(self, n_folds: int):
        assert self.data != None
        
        self.dml1 = dml.DoubleMLIIVM(self.data,
                                     ml_g=self.regressor,
                                     ml_m=self.classifier,
                                     ml_r=self.classifier,
                                     n_folds=n_folds,
                                     dml_procedure='dml1')
        
        return self
    
    def setup_dml2(self, n_folds: int):
        assert self.data != None
        
        self.dml2 = dml.DoubleMLIIVM(self.data,
                                     ml_g=self.regressor,
                                     ml_m=self.classifier,
                                     ml_r=self.classifier,
                                     n_folds=n_folds,
                                     dml_procedure='dml2')
        
        return self
    
    def fit_dml1(self):
        assert self.dml1 != None
        
        self.dml1.fit()
        return self
    
    def fit_dml2(self):
        assert self.dml2 != None
        
        self.dml2.fit()
        return self
    
    def run_simulation(self, i: int, k: int, dml_procedure: str, n_obs: int, seed: Union[None, int]):
        if dml_procedure == 'dml1':
            self.generate_data(n_obs=n_obs, seed=seed)
            self.setup_dml1(n_folds=k)
            self.fit_dml1()
            
            bias = self.dml1.coef[0]
            mse = (self.dml1.se[0] ** 2) * (n_obs - 1)
            
            return i, (bias, mse)
    
        else:
            assert dml_procedure == 'dml2'
            
            self.generate_data(n_obs=n_obs, seed=seed)
            self.setup_dml2(n_folds=k)
            self.fit_dml2()
            
            bias = self.dml2.coef[0]
            mse = (self.dml2.se[0] ** 2) * (n_obs - 1)
            
            return i, (bias, mse)
