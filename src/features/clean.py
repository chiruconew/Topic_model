import spacy  
from typing import List

import re
import pandas as pd
import gensim
from typing import Generator, List
import spacy


def clean_up_text(df:pd.DataFrame) -> List[str]:
    data = df.content.values.tolist() # convertir a lista
    data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data] # Quitar e-mail
    data = [re.sub(r'\s+', ' ', sent) for sent in data] # quitar enters (new line)
    data = [re.sub(r"\'", "", sent) for sent in data] # quitar comillas
    return(data)
    