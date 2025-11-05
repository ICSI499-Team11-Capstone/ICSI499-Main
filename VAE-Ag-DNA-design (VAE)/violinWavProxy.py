#wav proxy violin

import math
import time
# import sequenceDataset as sd
import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns



def process_data_file(path_to_dataset: str, sequence_length=10):
    """Takes in a filepath to desired dataset and the sequence length of the sequences for that dataset,
    saves .npz file with arrays for one hot encoded sequences, array of wavelengths and array of local
    integrated intensities"""
    data = sd.SequenceDataset(path_to_dataset, sequence_length)
    ohe_sequences = data.transform_sequences(data.dataset['Sequence'].apply(lambda x: pd.Series([c for c in x])).
                                             to_numpy())
    Wavelen = np.array(data.dataset['Wavelen'])
    LII = np.array(data.dataset['LII'])
    np.savez(f"data-for-sampling/processed-data-files/pca-data-{time.time()}", Wavelen=Wavelen, LII=LII,
             ohe=ohe_sequences)


# plot based on lii
def pca_visualize_lii(principalComponents, principalDf, lii, labels):
    """Takes in principal components, principal degrees of freedom, array of data points from processed data based
    upon local integrated intensity and labels for principal component analysis, all of which are needed to visualize
    principal component analysis based upon local integrated intensity"""

    lii_color = []
    for i in range(len(lii)):
        if math.isnan(lii[i]):
            lii_color.append('Null wavelength (new sequences)')
        elif lii[i] <= 1:
            lii_color.append('<= 1')
        else:
            lii_color.append('> 1')

    fig = px.scatter_matrix(
        principalComponents,
        labels=labels,
        dimensions=range(4),
        color = lii_color

    )
    fig.update_traces(diagonal_visible=False)
    fig.show()

    # uncomment this if you want to save
    # fig.write_image("PCA_lii/pca.png")


# plot based on wl
def pca_visualize_wavelength(principalComponents, principalDf, wavelength, labels):
    """Takes in principal components, principal degrees of freedom, array of data points from processed data based
    upon wavelength and labels for principal component analysis, all of which are needed to visualize
    principal component analysis based upon wavelength"""

    wl_color = []
    for i in range(len(wavelength)):
        if math.isnan(wavelength[i]):
            wl_color.append('Null wavelength (new sequences)')
        if wavelength[i] <= 590:
            wl_color.append('<= 590 nm')
        elif wavelength[i] > 590 and wavelength[i] <= 660:
            wl_color.append('590-660 nm')
        elif wavelength[i] > 660 and wavelength[i] <= 800:
            wl_color.append('660-800 nm')
        elif wavelength[i] > 800:
            wl_color.append('> 800 nm(Near IR)')

    fig = px.scatter_matrix(
        principalComponents,
        labels=labels,
        dimensions=range(4),
        color = wl_color,
        # color_discrete_sequence=["green", "silver", "gray", "red", "yellow"]
        color_discrete_map={
                "<= 590 nm": "green",
                "590-660 nm": "red",
                "660-800 nm": "indianred",
                "> 800 nm(Near IR)": "yellow"},

    )
    fig.update_traces(diagonal_visible=False)

    fig.show()

    # uncomment this if you want to save
    # fig.write_image("PCA_wl/pca.png")


def violinWavProxy(wavelength, z):
    """A 3D version of pca_visualize_wavelength(), where z provides a third dimension in the visualization of
    principal component analysis"""

    wl_color = []

    color_class = []
    symbol_class = [0] * len(wavelength)

    for i in range(len(wavelength)):
        if math.isnan(wavelength[i]):
            wl_color.append('Null wavelength (new sequences)')
            color_class.append('nan')
        if wavelength[i] <= 590:
            wl_color.append('<= 590 nm')
            color_class.append('Green')
        elif wavelength[i] > 590 and wavelength[i] <= 660:
            wl_color.append('590-660 nm')
            color_class.append('Red')
        elif wavelength[i] > 660 and wavelength[i] <= 800:
            wl_color.append('660-800 nm')
            color_class.append('Far Red')
        elif wavelength[i] > 800:
            wl_color.append('> 800 nm(Near IR)')
            color_class.append('NIR')


    # scatterDf = pd.DataFrame()
    # scatterDf["WAV Proxy"] = z[:,0].tolist()
    # scatterDf["Classes"] = color_class
    # scatterDf["Symbols"] = symbol_class

    # plt.rcParams['font.family'] = 'Arial'
    # sns.set(style="whitegrid", font_scale = 1)
    # # df = sns.load_dataset('scatterDf')
    # my_pal = {"Green": "#25DC05", "Red": "#F80326", "Far Red": "#632124", "NIR": "#2F9EFE"}
    # # plot

    # # sns.violinplot( x=scatterDf["Classes"], y=scatterDf["WAV Proxy"],
    # #                palette=my_pal, order=[ "Green", "Red", "Very Red", "NIR"],
    # #                linewidth=(3))
    
    # sns.violinplot( x=scatterDf["WAV Proxy"], y=scatterDf["Classes"],
    #                palette=my_pal, order=[ "NIR", "Far Red", "Red", "Green"],
    #                linewidth=(3))


    # plt.savefig("wavProxy.eps", format = 'eps', dpi=1200)
    # plt.show()

 

    scatterDf = pd.DataFrame()
    scatterDf["WAV Proxy"] = z[:, 0].tolist()
    scatterDf["Classes"] = color_class
    scatterDf["Symbols"] = symbol_class

    plt.rcParams['font.family'] = 'Arial'
    sns.set(style="whitegrid", font_scale=2)
    my_pal = {"Green": "#25DC05", "Red": "#F80326", "Far Red": "#632124", "NIR": "#2F9EFE"}

    # Create the violin plot
    sns.violinplot(x=scatterDf["WAV Proxy"], y=scatterDf["Classes"],
               palette=my_pal, order=["NIR", "Far Red", "Red", "Green"],
               linewidth=3)

    # Increase the size of text for the x-axis and y-axis
    plt.xlabel("WAV Proxy", fontsize=25, color='black', fontname='Arial')  # You can adjust the fontsize as needed
    plt.ylabel("Classes", fontsize=25, color='black', fontname='Arial')     # You can adjust the fontsize as needed


# Change the color of the box and grids to black
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    plt.gcf().set_size_inches(12, 7)  # Adjust the figure size as needed
    plt.tight_layout()

# Save the plot
    plt.savefig("ViolinPlot1.pdf", format='pdf', dpi=1200)

# Show the plot
    plt.show()





def process_data(path_to_dataset):
    """Basic wrapper for loading the given dataset using numpy"""
    data = np.load(path_to_dataset)
    return data


def get_wavelength_arr(data):
    """Basic wrapper for accessing wavelength array from data array"""
    wavelength = data['Wavelen']
    return wavelength


def get_lii_arr(data):
    """Basic wrapper for accessing local integrated intensity array from data array"""
    lii = data["LII"]
    return lii


def get_ohe_data(data):
    """Wrapper function that accesses one hot encoded array from data array and returns it as a pytorch tensor"""
    ohe_data = torch.from_numpy(data['ohe'])
    return ohe_data


def process_model(path_to_model):
    """Basic wrapper for loading the archived .pt pytorch model"""
    model = torch.load(path_to_model)
    return model


def get_z_from_latent_distribution(model, ohe_data):
    """Returns the array of data points in latent space z from the parametrized latent distribution latent_dist"""
    latent_dist = model.encode(ohe_data)
    z = latent_dist.loc.detach().numpy()
    return z


def preprocess_pca(z):
    """Preprocessing before pca visualization can occur (returns needed objects such as principal components, principal
    degress of freedom and labels"""

    # PCA
    pca = PCA(n_components=4)
    principalComponents = pca.fit_transform(z)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    # plt.scatter(principalComponents[:,0], principalComponents[:,1])
    labels = {
        str(i): f"PC {i + 1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }
    return principalComponents, principalDf, labels


def calculate_wavelength_means_and_standard_deviations(wavelength_array, z_wv_arr):
    """This is used to calculate the means and standard deviations for the different classes of z values
    corresponding to wavelength (new sequences, green, red, very red, near-IR). """
    ret_val = []
    for i in range(5):
        ret_val.append([])  # Hardcoded 5 for wavelength classes

    for i in range(len(wavelength_array)):
        if math.isnan(wavelength_array[i]):
            ret_val[0] = np.append(ret_val[0], z_wv_arr[i])
        elif wavelength_array[i] <= 590:
            ret_val[1] = np.append(ret_val[1], z_wv_arr[i])
        elif 590 < wavelength_array[i] <= 660:
            ret_val[2] = np.append(ret_val[2], z_wv_arr[i])
        elif 660 < wavelength_array[i] <= 800:
            ret_val[3] = np.append(ret_val[3], z_wv_arr[i])
        elif wavelength_array[i] > 800:
            ret_val[4] = np.append(ret_val[4], z_wv_arr[i])


    ret_val_mean = np.array(ret_val, dtype=object)
    ret_val_std = np.array(ret_val, dtype=object)
    for i, arr in enumerate(ret_val):
        if len(arr) == 0:
            ret_val_mean[i] = None
            ret_val_std[i] = None
        else:
            ret_val_mean[i] = np.mean(arr)
            ret_val_std[i] = np.std(arr)

    return ret_val_mean, ret_val_std


def add_wavelength_annotations_for_3d(fig, wavelength_array, z_wv_arr):
    """Function used to add annotations to wavelength plots that indicate mean and standard deviations for different
    clouds of points"""

    wv_means, wv_std = calculate_wavelength_means_and_standard_deviations(wavelength_array, z_wv_arr)

    fig.add_annotation(dict(font=dict(color='black', size=15), x=1, y=.8, xanchor='right', showarrow=False,
                                text=f"New Mean: {None if wv_means[0] is None else '{:.3f}'.format(wv_means[0])}"))
    fig.add_annotation(dict(font=dict(color='black', size=15), x=1, y=.75, xanchor='right', showarrow=False,
                            text=f"New Std: {None if wv_std[0] is None else '{:.3f}'.format(wv_std[0])}"))
    fig.add_annotation(dict(font=dict(color='black', size=15), x=1, y=.70, xanchor='right', showarrow=False,
                            text=f"Green Mean: {None if wv_means[1] is None else '{:.3f}'.format(wv_means[1])}"))
    fig.add_annotation(dict(font=dict(color='black', size=15), x=1, y=.65, xanchor='right', showarrow=False,
                            text=f"Green Std: {None if wv_std[1] is None else '{:.3f}'.format(wv_std[1])}"))
    fig.add_annotation(dict(font=dict(color='black', size=15), x=1, y=.60, xanchor='right', showarrow=False,
                            text=f"Red Mean: {None if wv_means[2] is None else '{:.3f}'.format(wv_means[2])}"))
    fig.add_annotation(dict(font=dict(color='black', size=15), x=1, y=.55, xanchor='right', showarrow=False,
                            text=f"Red Std: {None if wv_std[2] is None else '{:.3f}'.format(wv_std[2])}"))
    fig.add_annotation(dict(font=dict(color='black', size=15), x=1, y=.50, xanchor='right', showarrow=False,
                            text=f"Very Red Mean: {None if wv_means[3] is None else '{:.3f}'.format(wv_means[3])}"))
    fig.add_annotation(dict(font=dict(color='black', size=15), x=1, y=.45, xanchor='right', showarrow=False,
                            text=f"Very Red Std: {None if wv_std[3] is None else '{:.3f}'.format(wv_std[3])}"))
    fig.add_annotation(dict(font=dict(color='black', size=15), x=1, y=.40, xanchor='right', showarrow=False,
                            text=f"Near-IR Mean: {None if wv_means[4] is None else '{:.3f}'.format(wv_means[4])}"))
    fig.add_annotation(dict(font=dict(color='black', size=15), x=1, y=.35, xanchor='right', showarrow=False,
                            text=f"Near-IR Std: {None if wv_std[4] is None else '{:.3f}'.format(wv_std[4])}"))


def conduct_visualizations(path_to_dataset: str, path_to_model, pca_tuple: tuple):
    """Function for conducting visualizations of pca. Needs the path to the processed dataset you want to use (must
    use a dataset that is returned by process_data_file()), the path to the model you wish to use (has .pt extension)
    and boolean tuple representing whether you want to visualize lii, wavelength 2d or wavelength 3d. Type True if
    you want to produce the respective visualization, False otherwise."""
    data = process_data(path_to_dataset)
    model = process_model(path_to_model)

    ohe_data = get_ohe_data(data)
    z = get_z_from_latent_distribution(model, ohe_data)

    wavelength, lii = None, None
    if pca_tuple[0] is True:
        lii = get_lii_arr(data)
    if pca_tuple[1] is True or pca_tuple[2] is True:
        wavelength = get_wavelength_arr(data)


    principalComponents, principalDf, labels = preprocess_pca(z)

    if pca_tuple[0] is True:
        pca_visualize_lii(principalComponents, principalDf, lii, labels)
    if pca_tuple[1] is True:
        pca_visualize_wavelength(principalComponents, principalDf, wavelength, labels)
    if pca_tuple[2] is True:
        violinWavProxy(wavelength, z)




# process_data_file("data-for-sampling/merged-data-files/merged-cleandata-data-file-1642535499.239726")


# conduct_visualizations('clean-data-base.npz',
#               '1-20-22-models-info/a19lds19b0.007g1.0d1.0h13.pt', (False, False, True))

# conduct_visualizations('clean-data-base.npz',
#               'models/prunedNir/nir0a0.01lds19b0.007g1.0d1.0h13.pt', (False, False, True))

# conduct_visualizations('data-for-sampling/processed-data-files/clean-data-base.npz',
#               'all-results/results-2-9-22/models/a0.01lds19b0.007g0.0d0.0h13.pt', (False, False, True))

conduct_visualizations('./data-for-sampling/processed-data-files/processed-1675019876.8287058.npz', 
'./models/weighted/a0.003lds17b0.007g2d1h15.pt', (True, True, True))



              


