import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

df = sns.load_dataset('titanic')


text_params = df.select_dtypes('string')
print(text_params)

