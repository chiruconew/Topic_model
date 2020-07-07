import pandas as pd
from pandas import DataFrame


def read_sample() -> DataFrame:
    df = pd.read_json('C:/Users/chiruco/Desktop/python/ProyPython/Topic_Model/Topic_model_py/data/raw/newsgroups.json')
    return df