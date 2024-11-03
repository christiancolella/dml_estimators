import dml_estimators
import pandas as pd

import time

from concurrent.futures import ProcessPoolExecutor

df = pd.DataFrame([], columns=['n_folds', 'procedure', 'coef', 'std err', 'runtime'])

K = [2, 5, 10]

att_did = dml_estimators.ATTDID()
att_did.generate_data(n_obs=3000, seed=123)

for k in K:
    att_did.setup_dml1(n_folds=k)
    att_did.setup_dml2(n_folds=k)
    
    dml1_start = time.time()
    
    att_did.fit_dml1()
    
    dml1_end = time.time()
        
    dml1_summary = pd.DataFrame([[k, 'dml1', att_did.dml1.coef[0], att_did.dml1.se[0], dml1_end - dml1_start]], columns=['n_folds', 'procedure', 'coef', 'std err', 'runtime'])
    
    dml2_start = time.time()
    
    att_did.fit_dml2()
    
    dml2_end = time.time()
        
    dml2_summary = pd.DataFrame([[k, 'dml2', att_did.dml2.coef[0], att_did.dml2.se[0], dml2_end - dml2_start]], columns=['n_folds', 'procedure', 'coef', 'std err', 'runtime'])
        
    df = pd.concat((df, dml1_summary, dml2_summary))

df.reset_index()
df.set_index(['n_folds', 'procedure'])
print(df)
