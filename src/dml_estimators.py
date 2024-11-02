import numpy as np
import pandas as pd
import doubleml as dml

from scipy.stats import norm

from typing import Union
    
class ATTDID:
    data = None
    model = None
    
    def generate_data(self, n_obs: int, seed: Union[None, int]):
        assert n_obs > 0
        
        #
        # Population treatment effect for this DGP is 218.5625
        #
        
        np.random.seed(seed)
        
        df = pd.DataFrame([], columns=['y', 'a', 'x1', 'x2', 'x3', 'x4', 't'], dtype=np.float64)
        
        def fps(x: list):
            assert len(x) == 4
            return 0.25 * (-x[0] + 0.5 * x[1] - 0.25 * x[2] - 0.1 * x[3])
        
        def p(x: list):
            assert len(x) == 4
            return np.exp(fps(x)) / (1 + np.exp(fps(x)))
        
        def freg(x: list):
            assert len(x) == 4
            return 210 + 6.85 * x[0] + 3.425 * (x[1] + x[2] + x[3])
        
        def v(x: list, a: np.float64):
            assert len(x) == 4
            return a * freg(x) + np.random.normal(0, 1)
        
        X = np.random.uniform(0, 1, (n_obs, 4)).astype(np.float64)
        
        for x in X:
            a = np.random.binomial(1, p(x))
            
            for i in range(3):
                t = 0 if i == 0 else 1
                y = (1 if i == 0 else 2) * freg(x) + v(x, a) + np.random.normal(0, 1)
                
                df.loc[len(df.index)] = np.array([y, a, x[0], x[1], x[2], x[3], t])
        
        self.data = dml.DoubleMLData(df, y_col='y', d_cols='a', x_cols=['x1', 'x2', 'x3', 'x4'], t_col='t')
        
        return self
    
    def setup_model(self, regressor, classifier, n_folds: int):
        assert self.data != None
        
        self.model = dml.DoubleMLDID(self.data,
                                     ml_g=regressor,
                                     ml_m=classifier,
                                     n_folds=n_folds)
        
        return self
    
    def fit_model(self):
        assert self.model != None
        
        self.model.fit()
        return self
    
class LATE:
    data = None
    model = None
    
    def generate_data(self, n_obs: int, seed: Union[None, int]):
        assert n_obs > 0
        
        np.random.seed(seed)
        
        df = pd.DataFrame([], columns=['y', 'z', 'd', 'x'], dtype=np.float64)
        
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
                                     y_col='y',
                                     d_cols='d',
                                     x_cols='x',
                                     z_cols='z')
        
        return self
    
    def setup_model(self, regressor, classifier, n_folds: int):
        assert self.data != None
        
        self.model = dml.DoubleMLIIVM(self.data,
                                      ml_g=regressor,
                                      ml_m=classifier,
                                      ml_r=classifier,
                                      n_folds=n_folds)
        
        return self
    
    def fit_model(self):
        assert self.model != None
        
        self.model.fit()
        return self
