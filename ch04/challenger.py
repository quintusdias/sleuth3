import pathlib

import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

pd.options.display.float_format = "{:.1f}".format

path = pathlib.Path.home() / 'data' / 'sleuth3' / 'case0401.csv'
df = pd.read_csv(path)

