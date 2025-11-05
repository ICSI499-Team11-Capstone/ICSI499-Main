
from matplotlib.pyplot import axis
import torch
import numpy as np
import pandas as pd
from sequenceModelPep import SequenceModel
import sequenceDatasetPep as sd
import time
import os
import sys
import json
import re
import filter_sampled_sequences as filt
from sklearn.metrics import pairwise_distances
from collections import Counter
from embedder import Embedder


def find_knn(vals, k=15):
    ret = []
    weights = []
    adj = np.zeros_like(vals)
    for i in range(len(vals)):
        inds = np.argpartition(vals[i,:], k+1, axis=None)[:k+1].tolist()
        # print("inds    ", inds)
        if i in inds:
            inds.remove(i)
        else:
            inds.pop(-1)
        ret.append(inds)
        ##
        temp = []
        for j in range(len(inds)):
            #temp.append(1/(vals[i,inds[j]]+0.001))
            temp.append(np.exp(- vals[i, inds[j]] ** 2 / (2. * 1 ** 2))) 
        weights.append(temp)
        ##
        adj[i,inds] = 1
        # for j in range(len(inds)):
        #     adj[i,inds[j]] = vals[i,inds[j]]
    # print("ret    ", ret)

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

def process_data_file(path_to_dataset: str, sequence_length=10, prepended_name="processed", path_to_put=None, return_path=False, embedder=None):
    """Takes in a filepath to desired dataset and the sequence length of the sequences for that dataset,
    saves .npz file with arrays for one hot encoded sequences, array of wavelengths and array of local
    integrated intensities"""
    # print("path to dataset    ", path_to_dataset)
    # data = sd.SequenceDataset(datafile=path_to_dataset, seqlen=sequence_length, embedder=embedder)
    data = sd.SequenceDataset(datafile=path_to_dataset, embedder=embedder)
    # print("seq data   ", data.dataset['Sequence'])
    # kkkk += 3 
    # ohe_sequences = data.transform_sequences(data.dataset['Sequence'].apply(lambda x: pd.Series([c for c in x])).
    #                                          to_numpy())  #One hot encodings in the form ['A', 'C', 'G', 'T']
    ohe_sequences, ohe_emb, lengths = data.transform_sequences(data.dataset['Sequence'].to_numpy().tolist())
    # print("ohe sequences    ", ohe_sequences)
    # ret, ret_emb, np.array(lengths)
    # print("data    ", data.dataset)
    label = np.array(data.dataset['Label'])
    c4_label = np.array(data.dataset['class'])
    

    if path_to_put is not None:
        file_path = f"{path_to_put}/{prepended_name}-{time.time()}.npz"
    else:
        file_path = f"./data-for-sampling/processed-data-files/{prepended_name}-{time.time()}.npz"

    np.savez(file_path, label=label, c4_label=c4_label, ohe=ohe_sequences, ohe_emb=ohe_emb)
    # np.savez(file_path, ohe=ohe_sequences)
    if return_path:
        return file_path
    


def encode_data(ohe_sequences: object, model: object, edges: object):
    """This is a wrapper function for the encode() function that can be found in sequenceModel.py. This simply calls
    that function and returns the latent distribution that is produced in the latent space."""
    ohe_sequences = torch.from_numpy(ohe_sequences)

    # df2 = pd.read_excel('./data-and-cleaning/CSdistance_Sep18-shuffled_supercleanGMMFilteredClusterd.xlsx')
    # ds = df2.values
    # adj_inds, adj_weights, adj_matrix = find_knn(ds,k=10)
    # new_adj_inds = extract_connections(indices=np.arange(len(ds)), raw_dataset=ds, X_batch=ds, label_batch=[None for i in ds], adj_inds=adj_inds, adj_weights=adj_weights)
    if len(edges) > 0:
        # edges = torch.from_numpy(np.array(edges)[:,:2].T)
        edges = torch.from_numpy(np.array(edges))
    else:
        edges = torch.from_numpy(np.array(edges))
    # print("encode data    ", edges.shape)
    if torch.cuda.is_available():
        # edges = edges.cuda()
        ohe_sequences = ohe_sequences.cuda()

    # print("ohe final    ", ohe_sequences.shape)
    latent_dist, z_mean, z_log_std = model.encode(ohe_sequences, edges)
    # print(" ---   model is ok")
    # print("encode zmean     ", z_mean.shape)
    return latent_dist, z_mean, z_log_std

def encode_data(ohe_sequences: object, model: object, edges: object):
    """This is a wrapper function for the encode() function that can be found in sequenceModel.py. This simply calls
    that function and returns the latent distribution that is produced in the latent space, without changing shape."""
    ohe_sequences = torch.from_numpy(ohe_sequences)

    if len(edges) > 0:
        edges = torch.from_numpy(np.array(edges))
    else:
        edges = torch.from_numpy(np.array(edges))

    if torch.cuda.is_available():
        ohe_sequences = ohe_sequences.cuda()

    with torch.no_grad():
        latent_dist, z_mean, z_log_std = model.encode(ohe_sequences, edges)

    torch.cuda.empty_cache()

    return latent_dist, z_mean, z_log_std


def calculate_mean(mean_matrix: object):
    """This is a wrapper function for calculating the mean of the mean_matrix that characterizes the
    latent_distribution, which simply takes the mean of each dimension of the latent space using np.mean()."""
    dimension_means = []
    for i, col in enumerate(range(mean_matrix.shape[1])):
        mean = np.mean(mean_matrix[:, i])
        dimension_means.append(mean)
    mean_vector = np.array(dimension_means)
    return mean_vector


def calculate_covariance(mean_matrix: object):
    """This is a wrapper function for calculating the covariance matrix of the mean_matrix, which describes the variance
    between latent dimensions. This simply uses the numpy.cov() function to do so and returns the covariance_matrix."""
    mean_matrix_transpose = np.transpose(mean_matrix)
    covariance_matrix = np.cov(mean_matrix_transpose)
    return covariance_matrix



def calculate_z_sample(latent_dist, z_mean, z_log_std, Label_matrix, num_of_samples):
    """This is used to calculate the sample(s) z, the random sample from the latent distribution. This calculates the
    mean_vector, covariance_matrix, lower bound vector, calls R script and returns calculated z_sample."""
    # print("calc zmean   ", z_mean.shape)
    # print("calc std    ", z_log_std.shape)
    mean_matrix = latent_dist.mean.detach().cpu().numpy()
    # print(" calc z    ", mean_matrix.shape)
    # # mean_matrix = mean_matrix[inds]
    # # print(" calc z inds   ", mean_matrix.shape)
    # mean_vector = calculate_mean(mean_matrix)
    # covariance_matrix = calculate_covariance(mean_matrix)
    ###
    # print("zmean   ", z_mean)
    mean_vector = calculate_mean(z_mean)
    covariance_matrix = calculate_covariance(z_mean)
    ###
    # dist = np.random.normal(loc=mean_vector, scale=covariance_matrix, size=10)
    # print("covariance matrix    ", covariance_matrix)
    ###
    # print("mean vector    ", mean_vector)
    # print("cov    ", covariance_matrix)
    z_samples = np.random.multivariate_normal(mean_vector, covariance_matrix, size=num_of_samples)
    z_samples = np.asarray(z_samples)
    return z_samples


def decode_data(z_sample, model):
    """This is a wrapper function for the decode() function found in sequenceModel.py. This takes as input the
    calculated z_sample and returns the decoded sample in the form of a 10 x 4 matrix, where each 4-element array
    represents the numerical estimates for each base in the DNA sequence"""
    if torch.cuda.is_available():
        z_sample = z_sample.cuda()
    decoded_sample = model.decode(z_sample)
    return decoded_sample


def convert_sample(decoded_sample):
    sequence_alphabet = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L'
                         , 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y'}
    sequence_length = decoded_sample.shape[0]
    result = ""
    for i in range(sequence_length):
        max_index = np.argmax(decoded_sample[i, :])
        # print("max index    ", max_index)
        # print("cv sample    ", sequence_alphabet.get(max_index))
        result += sequence_alphabet.get(max_index)
    # print("result     ", result)
    return result


def convert_and_write_sample(decoded_sample, f: str):
    """This function takes in the decoded sample returned from decode_data() and takes the maximum value for each base,
    wherein the maximum estimate is replaced by the corresponding base in the DNA sequence. This is written to a csv."""
    sequence_alphabet = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L'
                         , 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y'}
    sequence_length = decoded_sample.shape[0]
    for i in range(sequence_length):
        max_index = np.argmax(decoded_sample[i, :])
        f.write(sequence_alphabet.get(max_index))
    # f.write("  ----- nnnnnnnnnnnn")
    f.write("\n")


def unpack_and_load_data(path_to_file: str, path_to_model: str):
    """This function is used as a wrapper function to load the .npz file that stores the wavelength, LII and one
    hot encoded arrays and load the trained model used for sampling. Both the data file and model are returned
    as objects."""
    data_file = np.load(path_to_file, allow_pickle=True)
    #model = torch.load(path_to_model)
    model = torch.load(path_to_model, map_location=torch.device('cpu'))
    return data_file, model


def compare_sequences(sequence_a, sequence_b):
    """Compares two sequences base by base and returns the ratio of how many they have in common"""
    length = max(len(sequence_a), len(sequence_b))
    num_match = 0
    for i in range(length):
        num_match = num_match + 1 if sequence_a[i] == sequence_b[i] else num_match
    return eval(f"{num_match}/{length}")


def write_detailed_sequences(path_to_put_folder, path_to_sequences, z_label, generated_purity, neighbor_labels_list, neighbor_labels_10, neighbor_labels_20):
    detailed_path = f"{path_to_put_folder}/detailed-sequences"
    print(" **   detailed path    ", detailed_path)
    
    with open(detailed_path, 'w') as f:
        with open(path_to_sequences, 'r') as original:
            file_contents = original.readlines()

            newline = "\n".join("")

            initial_line = file_contents[0]
            f.write(initial_line)
            file_contents = file_contents[1:]

            for i, line in enumerate(file_contents):
                if i < np.shape(z_label)[0]:
                    t = z_label[i, :].tolist()
                    vals = '_'.join([str(val) for val in t])

                    # Keep neighbor_labels_list for k=15 and add new columns for k=10 and k=20
                    f.write(f"{line[:line.rindex(newline)-1]},{vals},{generated_purity[i]},{neighbor_labels_list[i]},{neighbor_labels_10[i]},{neighbor_labels_20[i]}\n")

    return detailed_path


def write_encoded_sequence_wavelength_lii(path_to_generated: str, path_to_data_file: str, model, edge_index):
    data_file = np.load(path_to_data_file)

    # label_array = data_file['label']
    ohe_sequences_tensor = data_file['ohe_emb']

    # print("ohe tensor    ", ohe_sequences_tensor)
    # print("ohe tensor    ", ohe_sequences_tensor.shape)

    # print("write encoded     ", ohe_sequences_tensor.shape)
    ##
    sa,sb,sc = ohe_sequences_tensor.shape
    edge_index = []
    for i in range(sa):
        for j in range(sa):
            edge_index.append([i,j,1])
    #########
    # for i in range(sa):
    #     edge_index.append([i,i])
    ##
    # edge_index = torch.from_numpy(np.array(edge_index))
    # print("edges    ", edge_index.shape)
    latent_dist, z_mean, z_log_std = encode_data(ohe_sequences_tensor, model, edge_index)

    # mean_matrix = latent_dist.mean.detach().cpu().numpy()
    # z_value = mean_matrix[:,0]
    # print("write after    ", z_value.shape)

    z_value = z_mean.detach().cpu().numpy()

    random_sample, _, _ = SequenceModel.reparametrize(model, latent_dist)

    decoded = decode_data(random_sample, model)

    newline = "\n".join("")

    with open(path_to_generated, 'r+') as f:
        file_contents = f.readlines()
        file_contents = file_contents[1:]
        f.truncate(0)

    # print("file contents    ", file_contents)

    with open(path_to_generated, 'r+') as f:
        cols = "Sequence Generated,Value Generated,Purity,Top 15 Neighbors,Top 10 Neighbors,Top 20 Neighbors, Sequence Encoded/Decoded,Value Encoded,Ratio\n"
        
        f.write(cols)

        decoded = decoded.detach().cpu().numpy()
        for i, line in enumerate(file_contents):
            if i < np.shape(z_value)[0]:
                sequence_original = line.split(',')[0]
                sequence_generated = convert_sample(decoded[i, :, :])
                ratio = compare_sequences(sequence_original, sequence_generated)
                ######
                t = z_value[i,:].tolist()
                # print("t    ", t)
                t = [str(i) for i in t]
                # print("t2    ", t)
                vals = '_'.join(t)
                # vals = vals.replace('\\n','')
                # print("vals    ", vals)
                ######
                # f.write(f"{line[:line.rindex(newline)-1]},{sequence_generated},{z_value[i]},{ratio}\n")
                f.write(f"{line[:line.rindex(newline)-1]},{sequence_generated},{vals},{ratio}\n")



def calculate_purity_jaccard(latent_code, class4_labels, base="B", k=15):
    """
    Calculate the Jaccard-based purity of each sample considering only the specified base.
    """
    distances = pairwise_distances(latent_code)

    neighbor_purity_scores = []
    neighbor_labels_15 = [] 

    for i in range(latent_code.shape[0]):
        neighbor_indices = np.argsort(distances[i])[1:k+1]  
        
        
        neighbor_labels = [class4_labels[idx] for idx in neighbor_indices]
        neighbor_labels_15.append("_".join(map(str, neighbor_labels)))

       
        current_label = set(c for c in str(class4_labels[i]) if c == base)
        neighbor_sets = [set(c for c in str(lbl) if c == base) for lbl in neighbor_labels]
        full_sets = [set(str(lbl)) for lbl in neighbor_labels] 

        jaccard_similarities = [
            len(current_label.intersection(neighbor)) / len(current_label.union(full))
            if len(current_label.union(full)) > 0 else 0  # Avoid division by zero
            for neighbor, full in zip(neighbor_sets, full_sets)
        ]

        neighbor_purity_scores.append(np.mean(jaccard_similarities))

    return neighbor_purity_scores, neighbor_labels_15



def calculate_generated_purity_jaccard(generated_z, training_z, training_labels, generated_label, base="B"):
    """
    Calculate the Jaccard-based purity for each generated sequence, considering only the specified base (e.g., "N").
    - Uses k=15 for purity calculation.
    - Stores neighbor labels for k=10, k=15, and k=20.
    """
    distances = pairwise_distances(generated_z, training_z)

    generated_purity = []
    neighbor_labels_list = []  
    neighbor_labels_10 = []
    neighbor_labels_20 = []

    for i in range(generated_z.shape[0]):
        
        neighbor_indices_10 = np.argsort(distances[i])[:10]
        neighbor_indices_15 = np.argsort(distances[i])[:15] 
        neighbor_indices_20 = np.argsort(distances[i])[:20]

        # Get neighbor labels
        neighbor_labels_10.append("_".join(map(str, [training_labels[idx] for idx in neighbor_indices_10])))
        neighbor_labels_list.append("_".join(map(str, [training_labels[idx] for idx in neighbor_indices_15])))  
        neighbor_labels_20.append("_".join(map(str, [training_labels[idx] for idx in neighbor_indices_20])))

        current_label = set(c for c in str(generated_label) if c == base)
        neighbor_sets_15 = [set(c for c in str(training_labels[idx]) if c == base) for idx in neighbor_indices_15]
        full_sets_15 = [set(str(training_labels[idx])) for idx in neighbor_indices_15]  

        jaccard_similarities = [
            len(current_label.intersection(neighbor)) / len(current_label.union(full))
            if len(current_label.union(full)) > 0 else 0  
            for neighbor, full in zip(neighbor_sets_15, full_sets_15)
        ]

        generated_purity.append(np.mean(jaccard_similarities))

    return generated_purity, neighbor_labels_list, neighbor_labels_10, neighbor_labels_20


def calculate_purity(latent_code, labels, k=15):
    distances = pairwise_distances(latent_code)

    neighbor_purity_scores = []

    for i in range(latent_code.shape[0]):
        # Find the k-nearest neighbors for each sequence
        neighbor_indices = np.argsort(distances[i])[1:k+1]
        neighbor_labels = labels[neighbor_indices]

        # Compute the neighbor purity for sequence i
        # print("t1    ", type(neighbor_labels))
        # print("t2    ", type(labels[i]))
        purity = np.sum(neighbor_labels == labels[i]) / k
        neighbor_purity_scores.append(purity)

    # print("neighbot scores len   ", len(neighbor_purity_scores))
    # print("neighbot scores    ", neighbor_purity_scores)

    return neighbor_purity_scores

def save_to_excel(sequences, labels, z_values, purities, neighbor_labels_15, file_name="output.xlsx"):
    data = {
        "Sequence": sequences,
        "Label": labels,
        "Z Value": [z.tolist() for z in z_values],  # Convert array to list
        "Purity": purities,
        "Top 15 Neighbors": neighbor_labels_15  # Now added
    }

    df = pd.DataFrame(data)
    df.to_excel(file_name, index=False, engine="openpyxl")
    print(f"Data saved to {file_name}")



def sampling(path_to_data_file: str, path_to_model: str, path_to_put: str, path_to_distance_file: str, num_of_samples: int, c_label: int, embedder) -> np.ndarray:
    """This function serves as a main function for the sampling process, taking in the path to the data file with the
    .npz extension, the path to the trained model used for sampling and a path to write the resulting sequences.
    Set the boolean value to True to only return the randomly sampled vectors from the truncated distribution,
    otherwise it decodes samples and writes to file path specified in arguments."""

    print("sampliiiiiing  ...")

    path_to_put_folder = f"{path_to_put}/samples-{time.time()}"
    os.mkdir(path_to_put_folder)

    data_file, model = unpack_and_load_data(path_to_data_file, path_to_model)

    if torch.cuda.is_available():
        model = model.cuda()

    if isinstance(c_label, int):
        label_array = data_file['label']
        purity_func = calculate_purity
        # print(" ---- label array 11   ", label_array)
    elif isinstance(c_label, str):
        label_array = data_file['c4_label']
        purity_func = calculate_purity_jaccard
        # print(" ---- label array 22   ", label_array)
    else:
        raise ValueError('Wrong type for class.')
    ohe_sequences_tensor = data_file['ohe_emb']

    # print("ohe shape    ", ohe_sequences_tensor.shape)

    #####################
    df2 = pd.read_excel(path_to_distance_file)
    ds = df2.values[:,1:]
    # print("ds    ", ds)
    adj_inds, adj_weights, adj_matrix = find_knn(ds,k=15)
    new_adj_inds = extract_connections(indices=np.arange(len(ds)), raw_dataset=ds, X_batch=ds, label_batch=[None for i in ds], adj_inds=adj_inds, adj_weights=adj_weights)
    # np.save('./temp_dist.npy', new_adj_inds)
    # ##
    # new_adj_inds = np.load('./temp_dist.npy')
    #####################
    latent_dist, z_mean, z_log_std = encode_data(ohe_sequences_tensor, model, new_adj_inds)

    # purity = purity_func(z_mean.detach().cpu().numpy(), label_array)
    # purity = np.array(purity)
    # actual_sequences = [convert_sample(ohe_seq) for ohe_seq in data_file['ohe']]

    # Calculate purity and retrieve top 15 neighbors
    purity, neighbor_labels_15 = calculate_purity_jaccard(z_mean.detach().cpu().numpy(), label_array)
    purity = np.array(purity)
    # print("datafile      ", data_file['ohe'])
    # print("datafile shape     ", data_file['ohe'].shape)
    actual_sequences = [convert_sample(ohe_seq) for ohe_seq in data_file['ohe']]


    # Save everything in Excel
    save_to_excel(
        sequences=actual_sequences,
        labels=label_array.tolist(),
        z_values=z_mean.detach().cpu().numpy(),
        purities=purity,
        neighbor_labels_15=neighbor_labels_15,  # Now storing the neighbors
        file_name="./data-and-cleaning/z-value/N-Purity-a22lds22b0.007g0.9132648047995245d1.0h11k17.xlsx"
    )

    z_mean_all = z_mean.detach().cpu().numpy()  
    
    ##################################################################################
    def is_nx_format(label):
        label = str(label).strip().upper()
        return label.startswith("N") and len(label) > 1  # Must start with N and have other characters


    def classify_label(label):
        # print(" -----   label    ", label)
        label = re.sub(r'[^ABCE]', '', str(label).upper()) 
        if not label:
            return "Unknown"

        ordered_letters = list(label)  
        if len(ordered_letters) >= 2 and ordered_letters[0] == ordered_letters[1]:
            return f"clean-{ordered_letters[0]}"

        seen_letters = set()
        unique_letters = []
        
        for letter in ordered_letters:
            if letter not in seen_letters:
                unique_letters.append(letter)
                seen_letters.add(letter)
            if len(unique_letters) == 2:  
                break
        
        if len(unique_letters) == 1:  
            return f"clean-{unique_letters[0]}"  
        return f"mixed-{unique_letters[0]}-{unique_letters[1]}"
    # Filter high-purity training samples

    print("zmean    ", z_mean.shape)
    print("zstd    ", z_log_std.shape)
    print("purity    ", purity.shape)
    print("label    ", label_array.shape)

    new_df = pd.DataFrame(data=np.hstack((label_array.reshape(-1,1), purity.reshape(-1,1))), columns=['Label', "Purity"])
    new_df["Classified_Label"] = new_df["Label"].apply(classify_label)
    label_purity_avg = new_df.groupby("Classified_Label")["Purity"].mean()
    label_counts = Counter(new_df["Classified_Label"])
    # print("label counts    ", label_counts)

    # print("all labels    ", new_df["Classified_Label"])
    # print("group label    ", group_label)
    
    # label_inds = new_df["Classified_Label"].values == group_label #"mixed-N-G"

    label_inds = label_array == c_label
    z_mean = z_mean[label_inds]
    z_std = z_log_std[label_inds]
    purity = purity[label_inds]

    print("after 1   ", z_mean.shape)
    print("after 2   ", z_std.shape)
    print("after 3   ", purity.shape)


    ############################################################################################################
    ############################################################################################################
    inds = np.argsort(purity)[-10:]
    #inds = np.argsort(purity)[:]
    z_mean = z_mean[inds].detach().cpu().numpy()
    z_std = z_std[inds].detach().cpu().numpy()
    z_samples = calculate_z_sample(latent_dist, z_mean, z_std, label_array, num_of_samples)
    # generated_purity, neighbor_labels_list = calculate_generated_purity_jaccard(
    #     z_samples, z_mean_all, label_array, c_label
    # )

    # print("-------")
    # print("zs   ", z_samples.shape)
    # print("zmean   ", z_mean_all.shape)
    # print("lal   ", label_array.shape)
    generated_purity, neighbor_labels_list, neighbor_labels_10, neighbor_labels_20 = calculate_generated_purity_jaccard(
        z_samples, z_mean_all, label_array, c_label
    )

    # print("generated purity    ", generated_purity)

    # print("z samples    ", z_samples.shape)
    # purity = calculate_purity(z_samples, label_array)
    # purity = np.array(purity)
    # print("purity len   ", len(purity))
    # print("purity    ", purity)
    # #####
    # z_samples = z_samples[label_inds]
    # purity = purity[label_inds]
    # print("label selection z_sam    ", z_samples.shape)
    # #####
    # inds = np.argsort(purity)[:num_of_samples]
    # print("sort inds    ", inds)
    # z_samples = z_samples[inds]
    # print("selected z samples     ", z_samples.shape)

    path_to_sequences = f"{path_to_put_folder}/generated-sequences"
    # print("path to seq    ", path_to_sequences)
    # print("z_samples    ", z_samples.shape)
    with open(path_to_sequences, 'a', newline='') as f:
        f.write("Sequence,Wavelen,LII\n")
        for i, sample in enumerate(z_samples):
            sample = np.array(sample, dtype='float32')
            sample = torch.tensor(sample)
            decoded_sample = decode_data(sample, model)
            decoded_sample = decoded_sample.detach().cpu().numpy()
            decoded_sample = np.reshape(decoded_sample, (decoded_sample.shape[1], decoded_sample.shape[2]))
            convert_and_write_sample(decoded_sample, f)

    print(" 000  path to sequences    ", path_to_sequences)
    post_processing(path_to_sequences, path_to_put_folder, z_samples, model, new_adj_inds, generated_purity, neighbor_labels_list, neighbor_labels_10, neighbor_labels_20, embedder)
    # post_processing(path_to_sequences, path_to_put_folder, z_samples, model, new_adj_inds, generated_purity, neighbor_labels_list)

def post_processing(path_to_sequences, path_to_put_folder, z_samples, model, edge_index, 
                    generated_purity, neighbor_labels_list, neighbor_labels_10, neighbor_labels_20, embedder):
    """Handles post-processing of generated sequences, filtering, and storing sequence details."""

    filt.write_unique(path_to_sequences)
    data_set_dict = filt.fill_training_data_dict(sampling_params["Original Data Path"])
    filt.remove_duplicate(data_set_dict, path_to_sequences)

    # Now, we include neighbors 10 and 20 in the detailed sequences file
    detailed_data_path = write_detailed_sequences(path_to_put_folder, path_to_sequences, z_samples, 
                                                  generated_purity, neighbor_labels_list, neighbor_labels_10, neighbor_labels_20)

    generated_data_path = process_data_file(path_to_sequences, prepended_name="generated-sequences-", 
                                            path_to_put=path_to_put_folder, return_path=True, embedder=embedder)
    # print(" *********   generated path  ", generated_data_path)
    write_encoded_sequence_wavelength_lii(detailed_data_path, generated_data_path, model, edge_index)



if __name__ == "__main__":
    print("--  sampling")
    with open("sampling-parameters.json", 'r') as f:
        try:
            data = json.load(f)
            sampling_params = data['Parameters']
        except:
            print("Cannot process parameter file, please make sure sampling-parameters.json is correctly configured.")
            sys.exit(1)

    print("setting loaded...")
    embedder = Embedder()
    path_to_data_npz = process_data_file(sampling_params['Original Data Path'], prepended_name="clean-data-base", return_path=True, embedder=embedder)
    sampling(path_to_data_npz, sampling_params['Model Path'], "./data-for-sampling", sampling_params['Distance Data Path'], sampling_params["Number of Samples"], sampling_params['Corresponding Label'], embedder=embedder)
    #os.remove(path_to_data_npz)