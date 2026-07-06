import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

df = sns.load_dataset('titanic')

print(df.head)
print(df.info)
print(df.axis[0], df.axis[1])

