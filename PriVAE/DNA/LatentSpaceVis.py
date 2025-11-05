import math
import time
import sequenceDataset as sd
import pandas as pd
import numpy as np
import torch
from umap import UMAP  # Import UMAP
from sklearn.decomposition import PCA  # Import PCA
from sklearn.metrics import pairwise_distances
import plotly.graph_objects as go
import random
import os
import numpy as np
import torch

SEED = 14
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_data_file(path_to_dataset: str, sequence_length=10):
    """Takes in a filepath to desired dataset and the sequence length of the sequences for that dataset,
    saves .npz file with arrays for one hot encoded sequences, array of wavelengths and array of local
    integrated intensities"""
    data = sd.SequenceDataset(path_to_dataset, sequence_length)
    ohe_sequences = data.transform_sequences(data.dataset['Sequence'].apply(lambda x: pd.Series([c for c in x])).
                                             to_numpy())
    Wavelen = np.array(data.dataset['Wavelen'])
    LII = np.array(data.dataset['LII'])
    np.savez(f"GCN-modifiedVAE/data-for-sampling/processed-data-files/pca-data-{time.time()}", Wavelen=Wavelen, LII=LII,
             ohe=ohe_sequences)


############################################################
def intra_cluster_distance(df, label_col):
    clusters = df[label_col].unique()
    intra_distances = {}
    for cluster in clusters:
        cluster_data = df[df[label_col] == cluster].drop(columns=[label_col])
        distances = pairwise_distances(cluster_data)
        intra_distances[cluster] = np.mean(distances)
    return intra_distances

def inter_cluster_distance(df, label_col):
    clusters = df[label_col].unique()
    inter_distances = {}
    for i, cluster1 in enumerate(clusters):
        for cluster2 in clusters[i+1:]:
            data1 = df[df[label_col] == cluster1].drop(columns=[label_col])
            data2 = df[df[label_col] == cluster2].drop(columns=[label_col])
            distances = pairwise_distances(data1, data2)
            inter_distances[(cluster1, cluster2)] = np.mean(distances)
    return inter_distances
#############################################################

base_colors = {"G": "forestgreen", "R": "red", "F": "lightskyblue", "N": "black"}


# Function to extract colors for mixed labels
def get_mixed_marker_colors(label):
    label = label.replace("clean-", "").replace("mixed-", "")

    parts = label.split("-")
    if len(parts) == 2:  
        return base_colors[parts[0]], base_colors[parts[1]] 
    
    return base_colors[label], None 

def get_marker_size(count):
    if count < 100:
        return 20
    elif 200 <= count <= 350:
        return 24
    elif count >= 390:
        return 30
    else:
        return 6

def get_inner_marker_size(outer_size):
    return max(2, outer_size - 10)



import plotly.graph_objects as go
import pandas as pd

def umap_visualize_3D_means(embeddingComponents, embeddingDf, Labels, variance_text=None):
    base_colors = {"G": "forestgreen", "R": "red", "F": "lightskyblue", "N": "black"}

    # Define cluster labels (original mappings)
    clusterscolor = {
        0: 'clean-G', 1: 'clean-R', 2: 'clean-F', 3: 'clean-N',
        4: 'mixed-G-R', 5: 'mixed-G-F', 6: 'mixed-G-N', 
        7: 'mixed-R-G', 8: 'mixed-R-F', 9: 'mixed-R-N',
        10: 'mixed-F-G', 11: 'mixed-F-R', 12: 'mixed-F-N', 
        13: 'mixed-N-G', 14: 'mixed-N-R', 15: 'mixed-N-F'
    }

    # Clean label names for plotting and legend (e.g., "mixed-R-G" → "RG")
    Labels_with_color = [clusterscolor[i].replace("clean-", "").replace("mixed-", "").replace("-", "") for i in Labels]
    embeddingDf['Labels'] = Labels_with_color

    means = embeddingDf.groupby('Labels', as_index=False).mean()
    counts = embeddingDf.groupby('Labels').size().reset_index(name='count')

    fig = go.Figure()

    def get_marker_size(count):
        if count < 100:
            return 17
        elif 200 <= count <= 350:
            return 24
        elif count >= 390:
            return 30
        else:
            return 16

    def get_inner_marker_size(outer_size):
        return max(2, outer_size - 9)

    legend_entries = {}
    means = means.sort_values(by='Labels')

    for i, row in means.iterrows():
        label = row['Labels']
        label_count = counts.loc[counts['Labels'] == label, 'count'].values[0]
        marker_size = get_marker_size(label_count)
        inner_size = get_inner_marker_size(marker_size)
        hover_text = f"{label} (Count: {label_count})"

        if len(label) == 2: 
            inner_color = base_colors[label[0]]
            outer_color = base_colors[label[1]]

            fig.add_trace(go.Scatter3d(
                x=[row.iloc[1]], y=[row.iloc[2]], z=[row.iloc[3]],
                mode='markers',
                marker=dict(size=marker_size, color=inner_color, symbol="circle"),
                name=label, hoverinfo="text", text=hover_text, showlegend=False
            ))

            fig.add_trace(go.Scatter3d(
                x=[row.iloc[1]], y=[row.iloc[2]], z=[row.iloc[3]],
                mode='markers',
                marker=dict(size=inner_size, color=outer_color, symbol="circle"),
                hoverinfo="text", text=hover_text, showlegend=False
            ))

            if label not in legend_entries:
                legend_entries[label] = go.Scatter3d(
                    x=[None], y=[None], z=[None], mode="markers",
                    marker=dict(size=60, symbol="circle", color=outer_color,
                                line=dict(color=inner_color, width=4)),
                    name=label
                )

        else: 
            marker_color = base_colors[label]
            fig.add_trace(go.Scatter3d(
                x=[row.iloc[1]], y=[row.iloc[2]], z=[row.iloc[3]],
                mode="markers",
                marker=dict(size=marker_size, symbol="circle", color=marker_color),
                name=label, hoverinfo="text", text=hover_text, showlegend=False
            ))

            if label not in legend_entries:
                legend_entries[label] = go.Scatter3d(
                    x=[None], y=[None], z=[None], mode="markers",
                    marker=dict(size=60, symbol="circle", color=marker_color),
                    name=label
                )

    ordered_labels = ['G', 'R', 'F', 'N', 'GR', 'GF', 'GN', 'RG', 'RF', 'RN', 'FG', 'FR', 'FN', 'NG', 'NR', 'NF']
    for label in ordered_labels:
        if label in legend_entries:
            fig.add_trace(legend_entries[label])

    # Layout setup
    fig.update_layout(
        width=1000,
        height=800,
        scene=dict(
            xaxis=dict(title='PC1', showticklabels=False, gridcolor='lightgray', backgroundcolor='white', tickfont=dict(size=35)),
            yaxis=dict(title='PC2', showticklabels=False, gridcolor='lightgray', backgroundcolor='white', tickfont=dict(size=35)),
            zaxis=dict(title='PC3', showticklabels=False, gridcolor='lightgray', backgroundcolor='white', tickfont=dict(size=35)),
            camera=dict(eye=dict(x=1.15, y=0.9, z=0.75)),
            bgcolor='white'
        ),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.08,
            font=dict(size=30, family="Arial", color="black"),
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='gray',
            borderwidth=1,
            itemsizing='trace',
            itemwidth=60
        ),
        margin=dict(l=10, r=10, b=10, t=10),
        paper_bgcolor='white'
    )

    # Save or show
    fig.write_image("./visualizations/2DNA-3D-Mean-PCA.png", scale=4, width=1000, height=800)
    fig.write_image("./visualizations/2DNA-3D-Mean-PCA.pdf", format='pdf')
    fig.write_image("./visualizations/2DNA-3D-Mean-PCA.eps", format='eps')
    fig.write_html("./visualizations/2DNA-3D-Mean-PCA.html")

    fig.show()




def process_data(path_to_dataset):
    """Basic wrapper for loading the given dataset using numpy"""
    data = np.load(path_to_dataset)
    return data

def get_label_arr(data):
    """Basic wrapper for accessing local integrated intensity array from data array"""
    label = data["Labels"]
    return label

def get_ohe_data(data):
    """Wrapper function that accesses one hot encoded array from data array and returns it as a pytorch tensor"""
    ohe_data = torch.from_numpy(data['ohe'])
    return ohe_data

def process_model(path_to_model):
    """Basic wrapper for loading the archived .pt pytorch model"""
    model = torch.load(path_to_model)#, map_location=torch.device('cpu'))
    return model

def get_z_from_latent_distribution(model, ohe_data, edges):
    """Returns the array of data points in latent space z from the parametrized latent distribution latent_dist"""
    latent_dist, z_mean, z_log_std = model.encode(ohe_data, edges)
    z = z_mean.detach().cpu().numpy()
    return z


def preprocess_pca(z, n_components=10):
    """Preprocessing before PCA visualization can occur, returns embedding components and the PCA variance explained."""

    pca_model = PCA(n_components=n_components, random_state=SEED)
    embeddingComponents = pca_model.fit_transform(z)
    embeddingDf = pd.DataFrame(data=embeddingComponents, columns=[f'PCA{i+1}' for i in range(n_components)])
    explained_variance = pca_model.explained_variance_ratio_

    print("Explained Variance by PC:", explained_variance) 

    return embeddingComponents, embeddingDf, explained_variance


def find_knn(vals, k=17):
    ret = []
    weights = []
    adj = np.zeros_like(vals)
    for i in range(len(vals)):
        inds = np.argpartition(vals[i,:], k+1, axis=None)[:k+1].tolist()
        inds.remove(i)
        ret.append(inds)
        temp = []
        for j in range(len(inds)):
            temp.append(np.exp(- vals[i, inds[j]] ** 2 / (2. * 1 ** 2)))
        weights.append(temp)
        adj[i,inds] = 1
    ret = np.array(ret)
    weights = np.array(weights)
    weights /= np.max(weights)
    return ret, weights, adj

def extract_connections(indices, raw_dataset, X_batch, label_batch, adj_inds, adj_weights):
    new_adj_inds = []
    for i in range(len(indices)):
        row = int(indices[i])
        for j in range(len(adj_inds[row])):
            start = int(indices[i])
            end = adj_inds[row][j]
            wg = adj_weights[row][j]
            new_adj_inds.append([end, start, wg])
    return new_adj_inds

def conduct_visualizations(path_to_dataset: str, path_to_model):
    """Function for conducting visualizations using PCA."""
    
    data = process_data(path_to_dataset)
    model = process_model(path_to_model)
    
    df2 = pd.read_excel('./data-and-cleaning/CSdistance-250116-shuffled-final-labeled-data-log10.xlsx')
    ds = df2.values
    adj_inds, adj_weights, adj_matrix = find_knn(ds, k=17)
    new_adj_inds = extract_connections(indices=np.arange(len(ds)), raw_dataset=ds, X_batch=ds, label_batch=[None for i in ds], adj_inds=adj_inds, adj_weights=adj_weights)

    ohe_data = get_ohe_data(data)
    model=model.to(device)
    ohe_data = ohe_data.to(device)
    z = get_z_from_latent_distribution(model, ohe_data, new_adj_inds)

    Labels = get_label_arr(data)
    embeddingComponents, embeddingDf, explained_variance = preprocess_pca(z, n_components=10)

    # Format variance text correctly
    cumulative_variance = np.cumsum(explained_variance)
    num_pcs_to_90 = np.argmax(cumulative_variance >= 0.90) + 1
    variance_text = f"Top {num_pcs_to_90} PCs preserve 90% variance<br>"
    variance_text += "<br>".join([f"PC {i+1} variance: {explained_variance[i]*100:.2f}%" for i in range(num_pcs_to_90)])

    print(variance_text)
    umap_visualize_3D_means(embeddingComponents, embeddingDf, Labels, variance_text)


 
#conduct_visualizations('./data-for-sampling/processed-data-files/processed-1740440934.750912.npz', './models/optuna/a22lds22b0.007g0.9132648047995245d1.0h110.007_0.9132648047995245_1.0_22_1_0.0_11_17.pt')
