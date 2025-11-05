from matplotlib.pyplot import axis
import torch
import numpy as np
import pandas as pd
from sequenceModel import SequenceModel
import sequenceDataset as sd
import time
import os
import sys
import json
import filter_sampled_sequences as filt
import rpy2.robjects.pandas2ri as pandas2ri
import rpy2.robjects.pandas2ri as pandas2ri
from rpy2.robjects.conversion import localconverter


# import sys
# sys.path.append('/Users/matthewkilleen/miniconda3/envs/kdd-sub/bin')


def process_data_file(path_to_dataset: str, sequence_length=10, prepended_name="processed", path_to_put=None, return_path=False):
    """Takes in a filepath to desired dataset and the sequence length of the sequences for that dataset,
    saves .npz file with arrays for one hot encoded sequences, array of wavelengths and array of local
    integrated intensities"""
    # Convert to absolute path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path_to_dataset = os.path.join(script_dir, path_to_dataset)
    
    data = sd.SequenceDataset(abs_path_to_dataset, sequence_length)
    ohe_sequences = data.transform_sequences(data.dataset['Sequence'].apply(lambda x: pd.Series([c for c in x])).
                                             to_numpy())  #One hot encodings in the form ['A', 'C', 'G', 'T']
    Wavelen = np.array(data.dataset['Wavelen'])
    LII = np.array(data.dataset['LII'])

    if path_to_put is not None:
        file_path = os.path.join(script_dir, path_to_put, f"{prepended_name}-{time.time()}.npz")
    else:
        file_path = os.path.join(script_dir, "data-for-sampling/processed-data-files", f"{prepended_name}-{time.time()}.npz")

    np.savez(file_path, Wavelen=Wavelen, LII=LII,ohe=ohe_sequences)
    if return_path:
        return file_path
    


def encode_data(ohe_sequences: object, model: object):
    """This is a wrapper function for the encode() function that can be found in sequenceModel.py. This simply calls
    that function and returns the latent distribution that is produced in the latent space."""
    ohe_sequences = torch.from_numpy(ohe_sequences)
    latent_dist = model.encode(ohe_sequences)
    return latent_dist


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


# NOTE: FOR FUTURE USE, ONE WOULD NEED TO BASE THIS CALCULATION UPON THE Z PROXY DIMENSIONS FOR WAVELENGTH
# AND LOCAL INTEGRATED INTENSITY, HERE WE ARE COMPARING AGAINST THRESHOLDS FOR THE ACTUAL VALUES
def calculate_lower_bound_vector(mean_matrix, wavelength_matrix, lii_matrix, wv_thresh, lii_thresh):
    """This is used to calculate the lower-bound vector for random sampling from the truncated
    normal distribution. It sets the 0th dimension and 1st dimensions of z space to certain values.
    The rest of the dimensions are the minimum values observed in the latent sample for the dimensions."""
    wav_arr = np.array([])
    for i, elem in enumerate(mean_matrix[:, 0]): # column of values of wavelength dim
        if wavelength_matrix[i] < 590: # signifies all data points with near-IR wavelength criteria
            wav_arr = np.append(wav_arr, elem)
    wav_mean = np.mean(wav_arr)
    wav_std = np.std(wav_arr) # Calculating mean and standard deviation for wavelength dimension for near-IR class

    lii_arr = np.array([])
    for j, elem in enumerate(mean_matrix[:,1]): # column of values of LII dim
        if lii_matrix[j] < 1: # signifies all data points with near-IR LII criteria
            lii_arr = np.append(lii_arr, elem)
    lii_mean = np.mean(lii_arr)
    lii_std = np.std(lii_arr) # Calculating mean and standard deviation for LII dimension for near-IR class
    

    mean_matrix_dims = mean_matrix.shape
    lower_bound_vector = np.array([float('inf') for i in range(mean_matrix_dims[1])])

    # Feel free to change these values for 0th and 1st dimensions of the latent space, respectively
    lower_bound_vector[0] = wav_mean #+ wav_std
    lower_bound_vector[1] = lii_mean #+ lii_std # TODO: Uncomment this if you want to constrain wavelength and LII proxy dimensions

    wv_filter = list(filter(lambda x: x >= np.percentile(wavelength_matrix, wv_thresh), wavelength_matrix))
    lii_filter = list(filter(lambda x: x >= np.percentile(lii_matrix, lii_thresh), lii_matrix))

    wv_test = list(map(lambda x: x > np.percentile(wavelength_matrix, wv_thresh), wv_filter))
    lii_test = list(map(lambda x: x > np.percentile(lii_matrix, lii_thresh), lii_filter))

    # lower_bound_vector[0] = np.percentile(wavelength_matrix, wv_thresh)
    # lower_bound_vector[1] = np.percentile(lii_matrix, lii_thresh)

    return lower_bound_vector


def execute_truncated_sampling_r(mean_vector, covariance_matrix, lower_bound_vector):
    """This function calls truncatedSampling.R to sample from the truncated normal distribution. Returns
    a numpy array with the resulting sampled vectors."""
    print('1')

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    os.environ['/usr/local/lib/R'] = sampling_params['Path to R'] # Note: this will differ based on install location and operating system
    print('2')
    
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri

    # Defining the R script and loading the instance in Python
    r = robjects.r
    print('3')

    r['source']('truncatedSampling.R')
    # Loading the function we have defined in R.
    truncated_sampling_r = robjects.globalenv['truncated_sampling']

    # Reading and processing data
    mean = pd.DataFrame(mean_vector, dtype='string')
    covariance = pd.DataFrame(covariance_matrix, dtype='string')
    lower = pd.DataFrame(lower_bound_vector, dtype='string')
    print('4')
    
    # pandas2ri.activate()

    # # converting it into r object for passing into r function
    # r_mean = pandas2ri.py2rpy(mean)
    # r_covariance = pandas2ri.py2rpy(covariance)
    # r_lower = pandas2ri.py2rpy(lower)
    
    with localconverter(pandas2ri.converter) as cv:
        r_mean = pandas2ri.py2rpy(mean)
        r_covariance = pandas2ri.py2rpy(covariance)
        r_lower = pandas2ri.py2rpy(lower)
    print('5')

    # Invoking the R function and getting the result
    samples = truncated_sampling_r(r_mean, r_covariance, r_lower, sampling_params["Number of Samples"])
    print('6')

    #print(samples)
    print('7')

    # Converting it back to a pandas dataframe.
    samples = pandas2ri.rpy2py(samples)
    print('8')

    return samples


def calculate_z_sample(latent_dist, wavelength_matrix, lii_matrix):
    """This is used to calculate the sample(s) z, the random sample from the latent distribution. This calculates the
    mean_vector, covariance_matrix, lower bound vector, calls R script and returns calculated z_sample."""
    mean_matrix = latent_dist.mean.detach().numpy()
    mean_vector = calculate_mean(mean_matrix)
    covariance_matrix = calculate_covariance(mean_matrix)
    lower_bound_vector = calculate_lower_bound_vector(mean_matrix, wavelength_matrix, lii_matrix,
    sampling_params['Wavelength Proxy Threshold'], sampling_params['LII Proxy Threshold'])
    z_samples = execute_truncated_sampling_r(mean_vector, covariance_matrix, lower_bound_vector)
    z_samples = np.asarray(z_samples)
    return z_samples


def decode_data(z_sample, model):
    """This is a wrapper function for the decode() function found in sequenceModel.py. This takes as input the
    calculated z_sample and returns the decoded sample in the form of a 10 x 4 matrix, where each 4-element array
    represents the numerical estimates for each base in the DNA sequence"""
    decoded_sample = model.decode(z_sample)
    return decoded_sample


def convert_sample(decoded_sample):
    sequence_alphabet = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    sequence_length = decoded_sample.shape[0]
    result = ""
    for i in range(sequence_length):
        max_index = np.argmax(decoded_sample[i, :])
        result += sequence_alphabet.get(max_index)
    return result


def convert_and_write_sample(decoded_sample, f: str):
    """This function takes in the decoded sample returned from decode_data() and takes the maximum value for each base,
    wherein the maximum estimate is replaced by the corresponding base in the DNA sequence. This is written to a csv."""
    sequence_alphabet = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    sequence_length = decoded_sample.shape[0]
    for i in range(sequence_length):
        max_index = np.argmax(decoded_sample[i, :])
        f.write(sequence_alphabet.get(max_index))
    f.write("\n")


def unpack_and_load_data(path_to_file: str, path_to_model: str):
    """This function is used as a wrapper function to load the .npz file that stores the wavelength, LII and one
    hot encoded arrays and load the trained model used for sampling. Both the data file and model are returned
    as objects."""
    data_file = np.load(path_to_file)
    model = torch.load(path_to_model, weights_only=False)
    return data_file, model


def compare_sequences(sequence_a, sequence_b):
    """Compares two sequences base by base and returns the ratio of how many they have in common"""
    length = max(len(sequence_a), len(sequence_b))
    num_match = 0
    for i in range(length):
        num_match = num_match + 1 if sequence_a[i] == sequence_b[i] else num_match
    return eval(f"{num_match}/{length}")


def write_detailed_sequences(path_to_put_folder, path_to_sequences, z_wav, z_lii):
    detailed_path = f"{path_to_put_folder}/detailed-sequences"
    with open(detailed_path, 'w') as f:
        with open(path_to_sequences, 'r') as original:
            file_contents = original.readlines()

            newline = "\n".join("")

            initial_line = file_contents[0]
            f.write(initial_line)
            file_contents = file_contents[1:]

            for i, line in enumerate(file_contents):
                if i < np.shape(z_wav)[0]:
                    f.write(f"{line[:line.rindex(newline)-1]},{z_wav[i]},{z_lii[i]}\n")

    return detailed_path


def write_encoded_sequence_wavelength_lii(path_to_generated: str, path_to_data_file: str, model):
    data_file = np.load(path_to_data_file)

    wavelength_array = data_file['Wavelen']
    local_ii_array = data_file['LII']
    ohe_sequences_tensor = data_file['ohe']

    latent_dist = encode_data(ohe_sequences_tensor, model)

    mean_matrix = latent_dist.mean.detach().numpy()

    z_wav = mean_matrix[:,0]
    z_lii = mean_matrix[:,1]

    random_sample, _, _ = SequenceModel.reparametrize(model, latent_dist)

    decoded = decode_data(random_sample, model)

    newline = "\n".join("")

    with open(path_to_generated, 'r+') as f:
        file_contents = f.readlines()
        file_contents = file_contents[1:]
        f.truncate(0)

    with open(path_to_generated, 'r+') as f:
        f.write("Sequence Generated,Wavelen Generated,LII Generated,Sequence Encoded/Decoded,Wavelen Encoded,LII Encoded,Ratio\n")

        decoded = decoded.detach().numpy()
        for i, line in enumerate(file_contents):
            if i < np.shape(z_wav)[0]:
                sequence_original = line.split(',')[0]
                sequence_generated = convert_sample(decoded[i, :, :])
                ratio = compare_sequences(sequence_original, sequence_generated)
                f.write(f"{line[:line.rindex(newline)-1]},{sequence_generated},{z_wav[i]},{z_lii[i]},{ratio}\n")


def write_merged_dataset(path_to_base_dataset: str, path_to_generated_samples: str, path_to_put_folder: str):
    generated_file = f"{path_to_put_folder}/merged-data-set"
    #generated_file = f"./data-for-sampling/merged-data-files/{time.time()}.csv"
    with open(generated_file, 'w') as new:
        with open(path_to_base_dataset) as base:
            contents = base.readlines()
            new.writelines(contents)
        with open(path_to_generated_samples) as gen:
            gen.readline()
            contents = gen.readlines()
            new.writelines(contents)

    return generated_file


def sampling(path_to_data_file: str, path_to_model: str, path_to_put: str) -> np.ndarray:
    """This function serves as a main function for the sampling process, taking in the path to the data file with the
    .npz extension, the path to the trained model used for sampling and a path to write the resulting sequences.
    Set the boolean value to True to only return the randomly sampled vectors from the truncated distribution,
    otherwise it decodes samples and writes to file path specified in arguments."""

    path_to_put_folder = f"{path_to_put}/samples-{time.time()}"
    os.mkdir(path_to_put_folder)

    data_file, model = unpack_and_load_data(path_to_data_file, path_to_model)

    wavelength_array = data_file['Wavelen']
    local_ii_array = data_file['LII']
    ohe_sequences_tensor = data_file['ohe']

    latent_dist = encode_data(ohe_sequences_tensor, model)

    z_samples = calculate_z_sample(latent_dist, wavelength_array, local_ii_array)

    path_to_sequences = f"{path_to_put_folder}/generated-sequences"
    with open(path_to_sequences, 'a', newline='') as f:
        f.write("Sequence,Wavelen,LII\n")
        for i, sample in enumerate(z_samples):
            sample = np.array(sample, dtype='float32')
            sample = torch.tensor(sample)
            decoded_sample = decode_data(sample, model)
            decoded_sample = decoded_sample.detach().numpy()
            decoded_sample = np.reshape(decoded_sample, (decoded_sample.shape[1], decoded_sample.shape[2]))
            convert_and_write_sample(decoded_sample, f)

    post_processing(path_to_sequences, path_to_put_folder, z_samples, model)


def post_processing(path_to_sequences, path_to_put_folder, z_samples, model):
    """This is a function that deals with all of the post processing needed in order to filter out repeated sequences that 
    were generated, create annotated files that list out important values in z space and create .npz files necessary to use
    PCA later on"""
    filt.write_unique(path_to_sequences)
    data_set_dict = filt.fill_training_data_dict(sampling_params["Original Data Path"])
    filt.remove_duplicate(data_set_dict, path_to_sequences)

    detailed_data_path = write_detailed_sequences(path_to_put_folder, path_to_sequences, z_samples[:,0], z_samples[:,1])
    generated_data_path = process_data_file(path_to_sequences, prepended_name="generated-sequences-", path_to_put=path_to_put_folder, return_path=True)
    write_encoded_sequence_wavelength_lii(detailed_data_path, generated_data_path, model)

    #merged_path = write_merged_dataset("cleandata.csv", path_to_sequences, path_to_put_folder)
    #process_data_file(merged_path, prepended_name="pca-merged-", path_to_put=path_to_put_folder)
    


with open("sampling-parameters.json", 'r') as f:
    try:
        data = json.load(f)
        sampling_params = data['Parameters']
    except:
        print("Cannot process parameter file, please make sure sampling-parameters.json is correctly configured.")
        sys.exit(1)
process_data_file(sampling_params['Original Data Path'], prepended_name='clean-data-base')  # Use this to process the data you wish to use into .npz

path_to_data_npz = process_data_file(sampling_params['Original Data Path'], prepended_name="clean-data-base", return_path=True)
sampling(path_to_data_npz, sampling_params['Model Path'], "./data-for-sampling/past-samples-with-info")
os.remove(path_to_data_npz)