import numpy as np
import pandas as pd
import scipy

from typing import Union

class ATTDID:
    seed = None
    
    def set_seed(self, seed: Union[None, int]):
        self.seed = seed
        return self
    
    def generate(self, n_obs: int):
        assert n_obs > 0
        
        np.random.seed(self.seed)
        
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
                
        return df

class LATE:
    seed = None
    
    def set_seed(self, seed: Union[None, int]):
        self.seed = seed
        return self
    
    def generate(self, n_obs: int):
        assert n_obs > 0
        
        np.random.seed(self.seed)
        
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
            
            z = np.random.binomial(1, scipy.stats.norm.cdf(x[i] - 0.5))
            d = z * d_1 + (1 - z) * d_0
            
            y = d * y_1 + (1 - d) * y_0
            
            df.loc[len(df.index)] = np.array([y, z, d, x[i]])
            
        print(df.head(5))
        return df
    
LATE().set_seed(123).generate(10)
