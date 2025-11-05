import pandas as pd
import numpy as np

df = pd.read_excel('all_14.xlsx')
# From Peter: We want to include values with a normalized intensity greater than 3. If a sequence has both a NIR and Peak 1 norm value we should use whichever value is larger and its corresponding wavelength.
df['WaveLength'] = df.apply(
    lambda df: df['NIR wavelength'] 
        if (not np.isnan(df['NIR wavelength']) and (np.isnan(df['Peak 1 [nm]']) or df['NIR Norm INT']>df['Peak 1 Norm area'])) else df['Peak 1 [nm]'] if not np.isnan(df['Peak 1 [nm]']) else df['Norm Peak 2 [nm]'] if not np.isnan(df['Norm Peak 2 [nm]']) else df['Norm Peak 3 [nm]'] if not np.isnan(df['Norm Peak 3 [nm]']) else 0, axis=1)
df['LII'] = df.apply(
    lambda df: df['NIR INT'] if (not np.isnan(df['NIR wavelength']) and (np.isnan(df['Peak 1 [nm]']) or df['NIR Norm INT']>df['Peak 1 Norm area']))
    else df['Peak 1 Norm area'] if not np.isnan(df['Peak 1 Norm area']) else df['Peak 2 Norm area'] if not np.isnan(df['Peak 2 Norm area']) else df['Peak 3 Norm area'] if not np.isnan(df['Peak 3 Norm area']) else 0, axis=1)
#df.to_csv("211112_datawithLII.csv")

# For now skip filtering based on Integrated intensity (threshold=0)
nondark = df.loc[df['I_int (450 - 800 nm)'] >= 0.0]
# Keep norm LII >= 0.5
cleandata = nondark.loc[nondark['LII'] >= 1] # This line is getting rid of darks (based on normalized LII)
header = ['Sequence', 'WaveLength', 'LII', 'Name']
cleandata.to_csv("cleandata_14.csv",index = False, columns = header)