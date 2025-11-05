
# """
# Created on Thu Jan 20 20:03:10 2022

# @author: farihamoomtaheen
# """

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from matplotlib import path
import shapleyGaussian as seq;

df = pd.read_csv("data-for-sampling/past-samples-with-info/samples-model-2/detailed-sequences-model-2", index_col = 0)


def Linechart(x, y1, y2, dimName):

# compare z0,z1 before and after re-encoding. Linechart
    plt.figure()
    plt.plot(x, y1, label= dimName + ' Generated')  
    plt.plot(x, y2, label= dimName + ' Encoded')
    
    plt.legend(loc="upper left")
    plt.xlabel("No of Samples")
    plt.ylabel(dimName)
    plt.title(dimName + " Comparison-Linechart")
    
    plt.show()
    
def Scatter(x, y1, y2, dimName):
    
# compare z0,z1 before and after re-encoding. Scatter
    plt.figure()    
    plt.scatter(x,y1,c='blue', label= dimName + ' Generated')
    plt.scatter(x,y2,c='red', label= dimName + ' Encoded')
    
    plt.legend(loc="upper left")
    plt.xlabel("No of Samples")
    plt.ylabel(dimName)
    plt.title(dimName + " Comparison-Scatterplot")
    
    plt.show()
    
# Wavelength plots
x = np.linspace(1, len(df), len(df))
y1 = df['Wavelen Generated']
y2 = df['Wavelen Encoded']
dimension = 'Wavelength'

Linechart(x,y1,y2, dimension)
Scatter(x, y1, y2, dimension)


# LII plots
y1 = df['LII Generated']
y2 = df['LII Encoded']
dimension = 'LII'

Linechart(x, y1, y2, dimension)
Scatter(x, y1, y2, dimension)

# fig = px.scatter(x, y1, color_discrete_sequence=['red'])
# fig = px.scatter(x, y2, color_discrete_sequence=['blue'])
# fig.show()

#compare difference in z0 before and after re-encoding. 
y = np.subtract(y1,y2)
plt.plot(x,y)
# plt.show()

plt.scatter(x,y, c='red')
plt.show()


def process_sequences_wavelength_lii(base_data_path: str):
    """Function used for writing original data to file"""
    df = pd.read_csv(base_data_path)
    wavelength_arr = df['Wavelen'].to_numpy()
    lii_arr = df['LII'].to_numpy()
    sequence_arr = df['Sequence'].to_numpy()

    return wavelength_arr, lii_arr, sequence_arr


def write_clean_data(base_data_csv: str, processed_data_file: str, model: str, generated_file: str):
    """This function is used to write a csv file that has sequence, wavelength, lii, z-wavelength, z-lii information
    for a given model and dataset"""
    processed_data_file, model = seq.unpack_and_load_data(processed_data_file, model)

    wavelength_array = processed_data_file['Wavelen']
    local_ii_array = processed_data_file['LII']
    ohe_sequences_tensor = processed_data_file['ohe']

    latent_dist = seq.encode_data(ohe_sequences_tensor, model)

    mean = latent_dist.mean.detach().numpy()
    mean = np.array(mean)

    wavelength_data, lii_data, sequence_data = process_sequences_wavelength_lii(base_data_csv)

    with open(generated_file, 'w+') as f:
        f.write("Sequence,Wavelength,LII,Z-Wavelength,Z-LII\n")
        for i, matrix in enumerate(mean):
            f.write(f"{sequence_data[i]},{wavelength_data[i]},{lii_data[i]},{matrix[0]},{matrix[1]}\n")

write_clean_data('cleandata.csv', 'data-for-sampling/processed-data-files/clean-data-base.npz', 'all-results/1-18-22-res/models/a0.005lds19b0.008g1.0d1.0h13.pt', 'data-info-model-4.csv')




