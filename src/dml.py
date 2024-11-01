import numpy as np
import pandas as pd
import doubleml as dml

from typing import Union
    
class ATTDID:
    def __init__(data: pd.DataFrame, y_col: str, a_col: str, x_cols: Union[None, str, list], t_col: str):
        self.obj_dml_data = dml.DoubleMLData(data, y_col=y_col, d_cols=a_col, x_cols=x_cols, t_col=t_col)
    
class LATE:
    def __init__(data: pd.DataFrame, y_col: str, z_col: Union[None, str], d_col: str, x_cols: Union[None, str, list]):
        self.obj_dml_data = dml.DoubleMLData(data, y_col=y_col, d_cols=a_col, x_cols=x_cols, z_cols=z_col)
