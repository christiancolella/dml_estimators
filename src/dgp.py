import numpy as np
import pandas as pd

class ATTDID:
    seed = None
    
    def set_seed(self, seed: int):
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
                
        print(df.head(5))
        return df

class LATE:
    seed = None
    
ATTDID().set_seed(123).generate(10)
