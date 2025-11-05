import math
import time
import sequenceDatasetPep as sd
import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA  # Import PCA
from sklearn.metrics import pairwise_distances
import plotly.graph_objects as go
import random
import os
import numpy as np
import torch
import plotly.graph_objects as go
import os

SEED = 42  # Set a seed for reproducibility
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

# Define colors for each base label
base_colors = {
    "G": "green",
    "R": "red",
    "F": "blue",
    "N": "black"
}

# Function to extract colors for mixed labels
def get_mixed_marker_colors(label):
    """
    Returns the colors for mixed labels.
    - Outer (square) = First label color
    - Inner (circle) = Second label color
    """
    # Remove "clean-" or "mixed-" to extract the base labels
    label = label.replace("clean-", "").replace("mixed-", "")

    parts = label.split("-")
    if len(parts) == 2:  # Mixed case
        return base_colors[parts[0]], base_colors[parts[1]]  # (Square color, Circle color)
    
    return base_colors[label], None  # Single label case

def get_marker_size(count):
    if count < 100:
        return 10
    elif 100 <= count <= 500:
        return 17
    elif count >= 500:
        return 24
    else:
        return 16

def get_inner_marker_size(outer_size):
    return max(5, outer_size - 7)


def umap_visualize_3D_means(embeddingComponents, embeddingDf, Labels, variance_text=None):

    base_colors = {
        "E": "green",
        "S": "red",
        "P": "blue",
        "N": "orange",
        "ESP": "black"  # <- Add this line
    }


    label_display_map = {
        "E": "E",
        "S": "S",
        "P": "P",
        "N": "N",
        "ES": "ES",
        "EP": "EP",
        "SP": "SP",
        "ESP": "ESP"
    }

    embeddingDf['Labels'] = Labels
    means = embeddingDf.groupby('Labels', as_index=False).mean()
    counts = embeddingDf.groupby('Labels').size().reset_index(name='count')

    fig = go.Figure()

    legend_entries = {}
    means = means.sort_values(by='Labels')

    for i, row in means.iterrows():
        label = row['Labels']
        display_label = label_display_map.get(label, label)
        label_count = counts.loc[counts['Labels'] == label, 'count'].values[0]
        print("\nLabel Counts:\n", counts)

        outer_marker_size = get_marker_size(label_count)
        inner_marker_size = get_inner_marker_size(outer_marker_size)
        hover_text = f"{display_label} (Count: {label_count})"

        if len(label) == 2:  # Mixed labels like EP, ES, SP
            inner_color = base_colors[label[1]]
            outer_color = base_colors[label[0]]

            # Outer circle
            fig.add_trace(go.Scatter3d(
                x=[row.iloc[1]], y=[row.iloc[2]], z=[row.iloc[3]],
                mode='markers',
                marker=dict(size=outer_marker_size, color=outer_color, symbol="circle"),
                name=display_label, hoverinfo="text", text=hover_text, showlegend=False
            ))

            # Inner circle
            fig.add_trace(go.Scatter3d(
                x=[row.iloc[1]], y=[row.iloc[2]], z=[row.iloc[3]],
                mode='markers',
                marker=dict(size=inner_marker_size, color=inner_color, symbol="circle"),
                hoverinfo="text", text=hover_text, showlegend=False
            ))

            if display_label not in legend_entries:
                legend_name_with_count = f"{display_label} ({label_count})"
                legend_entries[display_label] = go.Scatter3d(
                    x=[None], y=[None], z=[None], mode="markers",
                    marker=dict(size=outer_marker_size, symbol="circle", color=inner_color,
                                line=dict(color=outer_color, width=4)),
                    name=legend_name_with_count
                )


        else:  # Pure or 3-letter group like ESP
            marker_color = base_colors.get(label, base_colors.get(label[0], "black"))


            fig.add_trace(go.Scatter3d(
                x=[row.iloc[1]], y=[row.iloc[2]], z=[row.iloc[3]],
                mode="markers",
                marker=dict(size=outer_marker_size, symbol="circle", color=marker_color),
                name=display_label, hoverinfo="text", text=hover_text, showlegend=False
            ))

            if display_label not in legend_entries:
                legend_name_with_count = f"{display_label} ({label_count})"
                legend_entries[display_label] = go.Scatter3d(
                    x=[None], y=[None], z=[None], mode="markers",
                    marker=dict(size=outer_marker_size, symbol="circle", color=marker_color),
                    name=legend_name_with_count
                )


    # Add legend in your preferred order
    ordered_labels = ["N", "E", "S", "P", "ES", "EP", "SP", "ESP"]
    for label in ordered_labels:
        if label in legend_entries:
            fig.add_trace(legend_entries[label])

    # Layout and export
    fig.update_layout(
        width=900,
        height=700,
        scene=dict(
            xaxis=dict(title='', showticklabels=False, backgroundcolor='white', gridcolor='lightgray'),
            yaxis=dict(title='', showticklabels=False, backgroundcolor='white', gridcolor='lightgray'),
            zaxis=dict(title='', showticklabels=False, backgroundcolor='white', gridcolor='lightgray'),
            bgcolor='white',
            camera=dict(eye=dict(x=-2.1983323078256807, y=0.1335662280422116, z=0.09744294277367208)), #camera=dict(eye=dict(x=0.98, y=0.85, z=0.4)),
        ),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=21, family="Arial", color="black"),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='gray',
            borderwidth=1,
            itemwidth=50,
            itemsizing='trace'
        ),
        margin=dict(l=10, r=10, b=10, t=10),
        paper_bgcolor='white'
    )

    fig.write_html(
        "./visualizations/Pep-3D-Mean-PCA.html",
        include_plotlyjs="cdn",
        full_html=True,
        post_script="""
        <script>
        var plotDiv = document.querySelector('.js-plotly-plot');
        plotDiv.on('plotly_relayout', function(eventdata){
            if (eventdata['scene.camera.eye']) {
                const eye = eventdata['scene.camera.eye'];
                alert('Camera eye position:\\n' +
                    'x: ' + eye.x.toFixed(2) + '\\n' +
                    'y: ' + eye.y.toFixed(2) + '\\n' +
                    'z: ' + eye.z.toFixed(2));
            }
        });
        </script>
        """
    )


    # Save the visualization
    if not os.path.exists("./visualizations"):
        os.makedirs("./visualizations")
    fig.write_image("./visualizations/Pep-3D-Mean-PCA.png", scale=4, width=900, height=700)
    fig.write_image("./visualizations/Pep-3D-Mean-PCA.eps", format='eps')
    #fig.write_image("./visualizations/Pep-3D-Mean-PCA.pdf", format='pdf')
    fig.write_html("./visualizations/Pep-3D-Mean-PCA.html")

    fig.show()



def process_data(path_to_dataset):
    """Basic wrapper for loading the given dataset using numpy"""
    data = np.load(path_to_dataset, allow_pickle=True)
    return data

def get_label_arr(data):
    """Basic wrapper for accessing local integrated intensity array from data array"""
    # label = data["Labels"]
    # label = data["label"]
    label = data["c4_label"]
    print("lbl 1     ", data["label"])
    print("lbl 2     ", data["c4_label"])
    # kkk += 6
    return label

def get_ohe_data(data):
    """Wrapper function that accesses one hot encoded array from data array and returns it as a pytorch tensor"""
    ohe_data = torch.from_numpy(data['ohe'])
    ohe_emb = torch.from_numpy(data['ohe_emb'])
    return ohe_data, ohe_emb

def process_model(path_to_model):
    """Basic wrapper for loading the archived .pt pytorch model"""
    model = torch.load(path_to_model, map_location=torch.device('cpu'))
    return model

def get_z_from_latent_distribution(model, ohe_data, edges):
    """Returns the array of data points in latent space z from the parametrized latent distribution latent_dist"""
    print("ohe data    ", ohe_data.shape)
    latent_dist, z_mean, z_log_std = model.encode(ohe_data, edges)
    z = z_mean.detach().cpu().numpy()
    return z


def preprocess_pca(z, n_components=10):
    """Preprocessing before PCA visualization can occur, returns embedding components and the PCA variance explained."""

    pca_model = PCA(n_components=n_components, random_state=SEED)
    embeddingComponents = pca_model.fit_transform(z)
    embeddingDf = pd.DataFrame(data=embeddingComponents, columns=[f'PCA{i+1}' for i in range(n_components)])
    explained_variance = pca_model.explained_variance_ratio_

    print("\nExplained variance per PC:")
    for i, var in enumerate(explained_variance, 1):
        print(f"  PC{i}: {var*100:.2f}%")

    cumulative = np.cumsum(explained_variance)
    for i, cum_var in enumerate(cumulative, 1):
        if cum_var >= 0.90:
            print(f"\n✅ Total of {i} PCs are needed to preserve at least 90% of the variance.\n")
            break


    return embeddingComponents, embeddingDf, explained_variance


def find_knn(vals, k=15):
    ret = []
    weights = []
    adj = np.zeros_like(vals)
    
    for i in range(len(vals)):
        inds = np.argpartition(vals[i, :], k + 1, axis=None)[:k + 1].tolist()
        if i in inds:
            inds.remove(i)
        ret.append(inds)

        temp = []
        for j in inds:
            temp.append(np.exp(- vals[i, j] ** 2 / (2. * 1 ** 2)))
        weights.append(temp)

        adj[i, inds] = 1

    # Normalize each weight list manually
    max_weight = max([max(w) for w in weights if len(w) > 0])
    weights = [[w / max_weight for w in row] for row in weights]

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

    print("data    ", data)
    
    # df2 = pd.read_excel('./data-and-cleaning/CSdistance-250116-shuffled-final-labeled-data-log10.xlsx')
    # df2 = pd.read_excel('./data-and-cleaning/CSdistance-250116-shuffled-final-labeled-data-log10.xlsx')
    # ds = df2.values
    
    ds = pd.read_excel('./data-and-cleaning/L1distance_peptide_with_activity_labels-May6.xlsx').values[:,1:]

    adj_inds, adj_weights, adj_matrix = find_knn(ds, k=25)
    new_adj_inds = extract_connections(indices=np.arange(len(ds)), raw_dataset=ds, X_batch=ds, label_batch=[None for i in ds], adj_inds=adj_inds, adj_weights=adj_weights)

    ohe_data, ohe_emb = get_ohe_data(data)
    model=model.to(device)
    ohe_data = ohe_data.to(device)
    ohe_emb = ohe_emb.to(device)
    z = get_z_from_latent_distribution(model, ohe_emb, new_adj_inds)

    Labels = get_label_arr(data)
    embeddingComponents, embeddingDf, explained_variance = preprocess_pca(z, n_components=10)

    # Format variance text correctly
    cumulative_variance = np.cumsum(explained_variance)
    num_pcs_to_90 = np.argmax(cumulative_variance >= 0.90) + 1
    variance_text = f"Top {num_pcs_to_90} PCs preserve 90% variance<br>"
    variance_text += "<br>".join([f"PC {i+1} variance: {explained_variance[i]*100:.2f}%" for i in range(num_pcs_to_90)])

    print(variance_text)
    umap_visualize_3D_means(embeddingComponents, embeddingDf, Labels, variance_text)


    

conduct_visualizations('./Peptide-paper/clean-data-base-1747608586.5291824.npz', './Peptide-paper/a32lds32b0.007g0.5890989775345152d1.0h160.007_0.5890989775345152_1.0_32_1_0.0_16_25.pt')


