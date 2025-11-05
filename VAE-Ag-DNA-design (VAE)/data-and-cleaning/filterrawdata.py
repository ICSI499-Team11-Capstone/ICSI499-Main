import pandas as pd
import numpy as np

df = pd.read_csv('211112_training_data.csv')
df['WaveLength'] = df.apply(lambda df: df['NIR bright peak'] if not np.isnan(df['NIR bright peak']) else df['Peak 1 [nm]'] if not np.isnan(df['Peak 1 [nm]']) else df['Peak 2 [nm]'] if not np.isnan(df['Peak 2 [nm]']) else 0, axis=1)
df['LII'] = df.apply(lambda df: df['NIR Normalized I_int'] * 20 if not np.isnan(df['NIR Normalized I_int']) else df['Peak 1 area'] if not np.isnan(df['Peak 1 area']) else df['Peak 2 area'] if not np.isnan(df['Peak 2 area']) else 0, axis=1)
df.to_csv("datawithLII.csv")
nondark = df.loc[df['Normalized I_int'] >= 1.0] # This seems to be where we could introduce 0.8 (Matthew)
cleandata = nondark.loc[nondark['LII'] >= 0.2] # This line is getting rid of darks (based on normalized LII)
header = ['Sequence', 'WaveLength', 'LII']
cleandata.to_csv("cleandata.csv",index = False, columns = header)