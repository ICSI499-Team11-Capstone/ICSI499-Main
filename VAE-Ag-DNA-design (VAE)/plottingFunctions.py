from ast import Mod
from email import header
import math
from operator import index
from sys import path
from tkinter import font
from turtle import back, color, shape, width
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from pyparsing import col
import torch
import seaborn as sns
import time
import collections
from sklearn.preprocessing import OneHotEncoder
import random
import torch
from matplotlib.ticker import MaxNLocator
from sequenceModel import SequenceModel
from matplotlib import rcParams
sns.set(style="white", font_scale=2)


def gather_hyperparameter_spreads(directory_to_be_searched: str, param_arr: list, parameter_dictionary: dict):
    """Function that traverses specified directory and recursively logs hyperparameters it hasn't seen"""
    for element in os.listdir(directory_to_be_searched):
        element_path = os.path.join(directory_to_be_searched, element)
        if os.path.isfile(element_path) and ".npz" in element:
            current_npz = np.load(element_path)
            params = current_npz.f.par
            temp_param_arr = param_arr
            if len(params) != len(param_arr):
                temp_param_arr = param_arr[1:]
            for i, param in enumerate(params):
                param_list = parameter_dictionary.get(temp_param_arr[i])
                if param not in param_list:
                    param_list.append(param)
                    parameter_dictionary[temp_param_arr[i]] = param_list
        elif os.path.isdir(element_path):
            gather_hyperparameter_spreads(element_path, param_arr, parameter_dictionary)
    
    keys = parameter_dictionary.keys()
    for key in keys:
        parameter_dictionary.get(key).sort()

    return parameter_dictionary


def gather_training_data_info(path_to_base_npz: str, path_to_model):
    npz_data = np.load(path_to_base_npz)
    model = torch.load(path_to_model)

    ohe = npz_data['ohe']

    ohe = torch.from_numpy(ohe)
    latent_dist = model.encode(ohe)

    mean_matrix = latent_dist.mean.detach().numpy()

    z_wav = mean_matrix[:,0]
    z_lii = mean_matrix[:,1]

    return z_wav, z_lii


def gather_sampled_metrics(path_to_detailed_sequences_file: str):
    with open(path_to_detailed_sequences_file, 'r+') as f:
        df = pd.read_csv(f)
    col_names = [i for i in df.columns]
    wv_gen, lii_gen, wv_enc, lii_enc = df.loc[:,col_names[1]], df.loc[:,col_names[2]], df.loc[:,col_names[4]], df.loc[:,col_names[5]]
    wv_gen, lii_gen, wv_enc, lii_enc = wv_gen.to_numpy(dtype='float32'), lii_gen.to_numpy(dtype='float32'), wv_enc.to_numpy(dtype='float32'), lii_enc.to_numpy(dtype='float32')
    return wv_gen, lii_gen, wv_enc, lii_enc


def construct_wavelength_plot(train_wv, wv_gen, wv_enc):
    """Constructs the wavelength plot needed for z space representations"""
    train_wv_x = [i for i in range( math.floor(min(train_wv)), math.ceil(max(train_wv)) + 1, 1)]
    wv_gen_x = [i for i in range( math.floor(min(wv_gen)), math.ceil(max(wv_gen)) + 1, 1)]
    wv_enc_x = [i for i in range( math.floor(min(wv_enc)), math.ceil(max(wv_enc)) + 1, 1)]

    train_size = len(train_wv)
    train_wv_y = []
    # The three chunks of code below make a histogram by adding points less than arr[x] into y and deleting until empty
    for i in range(len(train_wv_x)):
        train_wv_y.append(np.size(train_wv[np.where(train_wv <= train_wv_x[i])]) / train_size)
        train_wv = np.delete(train_wv, np.where(train_wv <= train_wv_x[i]))

    gen_size = len(wv_gen)
    wv_gen_y = []
    for i in range(len(wv_gen_x)):
        wv_gen_y.append(np.size(wv_gen[np.where(wv_gen <= wv_gen_x[i])]) / gen_size)
        wv_gen = np.delete(wv_gen, np.where(wv_gen <= wv_gen_x[i]))

    enc_size = len(wv_enc)
    wv_enc_y = []
    for i in range(len(wv_enc_x)):
        wv_enc_y.append(np.size(wv_enc[np.where(wv_enc <= wv_enc_x[i])]) / enc_size)
        wv_enc = np.delete(wv_enc, np.where(wv_enc <= wv_enc_x[i]))

    min_in_train, min_in_gen, min_in_enc = min(train_wv_x), min(wv_gen_x), min(wv_enc_x)
    min_in_x = min(min_in_train, min_in_gen, min_in_enc)
    max_in_train, max_in_gen, max_in_enc = max(train_wv_x), max(wv_gen_x), max(wv_enc_x)
    max_in_x = max(max_in_train, max_in_gen, max_in_enc)

    true_x = [i for i in range(min_in_x, max_in_x + 1, 1)]
    max_len = len(true_x)

    train_wv_y_pad = []
    wv_gen_y_pad = []
    wv_enc_y_pad = []

    for i in true_x: # Padding code so that all arrays are the same length
        if i in train_wv_x:
            train_wv_y_pad.append(train_wv_y[train_wv_x.index(i)])
        else:
            train_wv_y_pad.append(0)
        if i in wv_gen_x:
            wv_gen_y_pad.append(wv_gen_y[wv_gen_x.index(i)])
        else:
            wv_gen_y_pad.append(0)
        if i in wv_enc_x:
            wv_enc_y_pad.append(wv_enc_y[wv_enc_x.index(i)])
        else:
            wv_enc_y_pad.append(0)

    fontsize = 30

    true_x = np.array(true_x)
    train_wv_y_pad = np.array(train_wv_y_pad)
    wv_gen_y_pad = np.array(wv_gen_y_pad)
    wv_enc_y_pad = np.array(wv_enc_y_pad)

    fig, ax = plt.subplots(figsize=(44,20))
    x_labels = [str(i) if i % 5 == 0 else ' ' for i in true_x]
    plt.xticks(ticks=true_x, labels=x_labels, fontsize=5*fontsize)
    plt.yticks(fontsize=5*fontsize)

    first_width = 1
    second_width = .4
    ax.bar(true_x, train_wv_y_pad, color='black', width=first_width, label='Training Data')
    ax.bar(true_x, wv_gen_y_pad, color='blue', width=second_width, label='Generated Samples', edgecolor='blue')


    ax.set_xlabel('WAV Proxy', fontsize=5*fontsize)
    ax.set_ylabel('Probability Density', fontsize=4.5*fontsize)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.legend(['Training', 'Sampled'], fontsize=3.75*fontsize, loc='upper right')

    #plt.show()

    fig.savefig('distribution-plot-wavelength-sampled.png', bbox_inches='tight')
    fig.savefig('distribution-plot-wavelength-sampled.eps', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(44,20))

    train_wv_y_pad = list(train_wv_y_pad)
    train_wv_y_pad = train_wv_y_pad[1:18]
    wv_enc_y_pad = list(wv_enc_y_pad)
    wv_enc_y_pad = wv_enc_y_pad[1:18]

    true_x = list(true_x)
    true_x = true_x[1:len(train_wv_y_pad) + 1]
    x_labels = [str(i) if i % 5 == 0 else ' ' for i in true_x]
    plt.xticks(ticks=true_x, labels=x_labels, fontsize=5*fontsize)
    yticks = plt.yticks(fontsize=5*fontsize)

    ax.bar(true_x, train_wv_y_pad, color='black', width=first_width, label='Training Data')
    ax.bar(true_x, wv_enc_y_pad, color='red', width=second_width, label='Generated Samples', edgecolor='red')


    ax.set_xlabel('WAV Proxy', fontsize=5*fontsize)
    ax.set_ylabel('Probability Density', fontsize=4.5*fontsize)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.legend(['Training', 'Re-encoded'], fontsize=3.75*fontsize, loc='upper right')

    #plt.show()

    fig.savefig('distribution-plot-wavelength-encoded.png', bbox_inches='tight')
    fig.savefig('distribution-plot-wavelength-encoded.eps', bbox_inches='tight')

    
def construct_lii_plot(train_lii, lii_gen, lii_enc):
    """Constructs the Local Integrated Intensity plot needed for z space representations"""
    train_lii_x = [i for i in range( math.floor(min(train_lii)), math.ceil(max(train_lii)) + 1, 1)]
    lii_gen_x = [i for i in range( math.floor(min(lii_gen)), math.ceil(max(lii_gen)) + 1, 1)]
    lii_enc_x = [i for i in range( math.floor(min(lii_enc)), math.ceil(max(lii_enc)) + 1, 1)]

    train_size = len(train_lii)
    train_lii_y = []
    # The three chunks of code below make a histogram by adding points less than arr[x] into y and deleting until empty
    for i in range(len(train_lii_x)):
        train_lii_y.append(np.size(train_lii[np.where(train_lii <= train_lii_x[i])]) / train_size)
        train_lii = np.delete(train_lii, np.where(train_lii <= train_lii_x[i]))

    gen_size = len(lii_gen)
    lii_gen_y = []
    for i in range(len(lii_gen_x)):
        lii_gen_y.append(np.size(lii_gen[np.where(lii_gen <= lii_gen_x[i])]) / gen_size)
        lii_gen = np.delete(lii_gen, np.where(lii_gen <= lii_gen_x[i]))

    enc_size = len(lii_enc)
    lii_enc_y = []
    for i in range(len(lii_enc_x)):
        lii_enc_y.append(np.size(lii_enc[np.where(lii_enc <= lii_enc_x[i])]) / enc_size)
        lii_enc = np.delete(lii_enc, np.where(lii_enc <= lii_enc_x[i]))

    min_in_train, min_in_gen, min_in_enc = min(train_lii_x), min(lii_gen_x), min(lii_enc_x)
    min_in_x = min(min_in_train, min_in_gen, min_in_enc)
    max_in_train, max_in_gen, max_in_enc = max(train_lii_x), max(lii_gen_x), max(lii_enc_x)
    max_in_x = max(max_in_train, max_in_gen, max_in_enc)

    true_x = [i for i in range(min_in_x, max_in_x + 1, 1)]
    max_len = len(true_x)

    train_lii_y_pad = []
    lii_gen_y_pad = []
    lii_enc_y_pad = []

    for i in true_x: # Padding code so that all arrays are the same length
        if i in train_lii_x:
            train_lii_y_pad.append(train_lii_y[train_lii_x.index(i)])
        else:
            train_lii_y_pad.append(0)
        if i in lii_gen_x:
            lii_gen_y_pad.append(lii_gen_y[lii_gen_x.index(i)])
        else:
            lii_gen_y_pad.append(0)
        if i in lii_enc_x:
            lii_enc_y_pad.append(lii_enc_y[lii_enc_x.index(i)])
        else:
            lii_enc_y_pad.append(0)

    true_x = np.array(true_x)
    train_lii_y_pad = np.array(train_lii_y_pad)
    lii_gen_y_pad = np.array(lii_gen_y_pad)
    lii_enc_y_pad = np.array(lii_enc_y_pad)

    fontsize = 30

    fig, ax = plt.subplots(figsize=(44,20))
    x_labels = [str(i) if i % 5 == 0 else ' ' for i in true_x]
    plt.xticks(ticks=true_x, labels=x_labels, fontsize=5*fontsize)
    plt.yticks(fontsize=5*fontsize)

    first_width = 1
    second_width = .4
    ax.bar(true_x, train_lii_y_pad, color='black', width=first_width, label='Training Data')
    ax.bar(true_x, lii_gen_y_pad, color='blue', width=second_width, label='Generated Samples', edgecolor='blue')

    ax.set_xlabel('LII Proxy', fontsize=5*fontsize)
    ax.set_ylabel('Probability Density', fontsize=4.5*fontsize)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.legend(['Training', 'Sampled'], fontsize=3.75*fontsize, loc='upper right')

    fig.savefig('distribution-plot-lii-generated.png', bbox_inches='tight')
    fig.savefig('distribution-plot-lii-generated.eps', bbox_inches='tight')

    fontsize = 30

    fig, ax = plt.subplots(figsize=(44,20))
    x_labels = [str(i) if i % 5 == 0 else ' ' for i in true_x]
    plt.xticks(ticks=true_x, labels=x_labels, fontsize=5*fontsize)
    plt.yticks(fontsize=5*fontsize)

    ax.bar(true_x, train_lii_y_pad, color='black', width=first_width, label='Training Data')
    ax.bar(true_x, lii_enc_y_pad, color='red', width=second_width, label='Generated Samples', edgecolor='red')

    ax.set_xlabel('LII Proxy', fontsize=5*fontsize)
    ax.set_ylabel('Probability Density', fontsize=4.5*fontsize)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.legend(['Training', 'Re-encoded'], fontsize=3.75*fontsize, loc='upper right')

    fig.savefig('distribution-plot-lii-encoded.png', bbox_inches='tight')
    fig.savefig('distribution-plot-lii-encoded.eps', bbox_inches='tight')


def create_wavelength_lii_dist_plots(path_to_base_npz: str, path_to_model, path_to_detailed_sequences_file: str):
    """This function constructs plots of training and generated data in z space, specifically analyzing the
    dimensions correlated with Wavelength and Local Integrated Intensity"""
    train_wav, train_lii = gather_training_data_info(path_to_base_npz, path_to_model)
    wv_gen, lii_gen, wv_enc, lii_enc = gather_sampled_metrics(path_to_detailed_sequences_file)
    construct_wavelength_plot(train_wav, wv_gen, wv_enc)
    construct_lii_plot(train_lii, lii_gen, lii_enc)


def generate_acc_vs_param_plot_lin_dims(npz_files_to_compare: tuple, param_idx=8, param_name='lin_dim', lin_dim_arr=[12,16,20]):
    """Generates a plot for the specified hyperparameter vs. reconstruction accuracy"""
    optimal_param_dict = {0: 0.01, 1: 0.007, 2: 1.0, 3: 1.0, 4: 19.0, 5: 1.0, 6: 0.0, 7: 13.0, 8: 16}
    param_acc_dict = {}
    for i, npz in enumerate(npz_files_to_compare):
        npz = np.load(npz)
        data = npz["tl"]
        param_value = lin_dim_arr[i]
        train_accuracy = data[:,5][-1]
        valid_accuracy = npz["vl"][:,5][-1]
        wl_correlation = npz["wldims"][-1,0]
        lii_correlation = npz["liidims"][-1,1]

        param_acc_dict[param_value] = [0,0,0,0]

        train_accuracy = data[:,5][-1]
        valid_accuracy = npz["vl"][:,5][-1]
        wl_correlation = npz["wldims"][-1,0]
        lii_correlation = npz["liidims"][-1,1]
        param_acc_dict[param_value] = [train_accuracy, valid_accuracy, wl_correlation, lii_correlation]

    param_acc_dict = collections.OrderedDict(sorted(param_acc_dict.items()))
    param_vals = [float(i) for i in param_acc_dict.keys()]
    train_accuracy = [param_acc_dict.get(i)[0] for i in param_vals]
    valid_accuracy = [param_acc_dict.get(i)[1] for i in param_vals]
    wl_correlation_vals = [param_acc_dict.get(i)[2] for i in param_vals]
    lii_correlation_vals = [param_acc_dict.get(i)[3] for i in param_vals]

    fontsize = 25

    fig, ax = plt.subplots(figsize=(30,22))
    par_vs_acc_line, = ax.plot(param_vals, train_accuracy, linewidth=20, zorder=1)
    par_vs_acc_scatter = ax.scatter(param_vals, train_accuracy, s=3000, zorder=1)
    par_vs_val_acc_line = ax.plot(param_vals, valid_accuracy, linewidth=20, zorder=1, linestyle='dashed', dashes=(5,10))
    par_vs_val_acc_scatter = ax.scatter(param_vals, valid_accuracy, s=3000, zorder=2)
    special_var = ax.scatter(optimal_param_dict.get(param_idx), 
    train_accuracy[param_vals.index(optimal_param_dict.get(param_idx))], color='red', marker='s', s=3000, zorder=2)
    val_special_var = ax.scatter(optimal_param_dict.get(param_idx), 
    valid_accuracy[param_vals.index(optimal_param_dict.get(param_idx))], color='red', marker='s', s=3000, zorder=2)


    x_label_name = "w"
    ax.set_xlabel(x_label_name, fontsize=10*fontsize)

    ax.set_ylabel("Accuracy", fontsize=10*fontsize)

    # ax.legend([par_vs_acc_line, par_vs_val_acc_line], ['Training', 'Validation'], loc=0)

    plt.xticks(fontsize= 8*fontsize)
    plt.yticks(fontsize= 8*fontsize)

    # min_in_y_range = min(train_accuracy) - .10 * min(train_accuracy) if min(train_accuracy) - .10 * min(train_accuracy) >= 0 else 0
    # max_in_y_range = max(train_accuracy) + .10 * max(train_accuracy) if max(train_accuracy) + .10 * max(train_accuracy) < 1 else 1

    # plt.xticks(ticks=np.arange(min(param_vals), max(param_vals) + .01, 2))

    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))

    fig.savefig(f"paper-materials/hyper-param-experiments/final-associated-files/{param_name}-vs-accuracy-{time.time()}.png", bbox_inches='tight')
    fig.savefig(f"paper-materials/hyper-param-experiments/final-associated-files/{param_name}-vs-accuracy-{time.time()}.eps", bbox_inches='tight')

    fontsize = 25

    fig, ax = plt.subplots(figsize=(30,22))
    par_vs_wl_line, = ax.plot(param_vals, wl_correlation_vals, color='black', linewidth=20, zorder=1)
    par_vs_wl_scatter = ax.scatter(param_vals, wl_correlation_vals, color='black', s=3000, zorder=2)
    special_var = ax.scatter(optimal_param_dict.get(param_idx), 
    wl_correlation_vals[param_vals.index(optimal_param_dict.get(param_idx))], color='red', marker='s', s=3000, zorder=2)

    
    par_vs_lii_line, = ax.plot(param_vals, lii_correlation_vals, color = 'black', linewidth=20, linestyle='dashed', dashes=(5,10), zorder=1)
    par_vs_lii_scatter = ax.scatter(param_vals, lii_correlation_vals, color = 'black', s=3000, zorder=2)
    ax.scatter(optimal_param_dict.get(param_idx), 
    lii_correlation_vals[param_vals.index(optimal_param_dict.get(param_idx))], color='red', marker='s', s=3000, zorder=2)


    x_label_name = "w"
    ax.set_xlabel(x_label_name, fontsize=10*fontsize)
        
    ax.set_ylabel("Correlation", fontsize=10*fontsize)

    plt.xticks(fontsize= 6*fontsize)
    plt.yticks(fontsize= 6*fontsize)
    ax.legend([par_vs_wl_line, par_vs_lii_line, special_var], ['WAV',
    'LII'],  fontsize=6*fontsize, bbox_to_anchor=(0,0,1,1), loc=7)

    # plt.xticks(ticks=np.arange(min(param_vals), max(param_vals) + .01, 2))

    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))

    fig.savefig(f"paper-materials/hyper-param-experiments/final-associated-files/{param_name}-vs-level-of-correlation-{time.time()}.png", bbox_inches='tight')
    fig.savefig(f"paper-materials/hyper-param-experiments/final-associated-files/{param_name}-vs-level-of-correlation-{time.time()}.eps", bbox_inches='tight')


def generate_acc_vs_param_plot(npz_files_to_compare: tuple, param_idx: int, param_name: str):
    """Generates a plot for the specified hyperparameter vs. reconstruction accuracy"""
    optimal_param_dict = {0: 0.01, 1: 0.007, 2: 1.0, 3: 1.0, 4: 19.0, 5: 1.0, 6: 0.0, 7: 13.0}
    param_acc_dict = {}
    for npz in npz_files_to_compare:
        npz = np.load(npz)
        data = npz["tl"]
        if np.shape(data[:,5])[0] != 2000:
            continue
        if len(npz["par"]) != 8: # this is the number of all parameters including alpha
            if param_name == "alpha":
                continue
            param_value = npz["par"][param_idx - 1]
        else:
            param_value = npz["par"][param_idx]
        train_accuracy = data[:,5][-1]
        valid_accuracy = npz["vl"][:,5][-1]
        wl_correlation = npz["wldims"][-1,0]
        lii_correlation = npz["liidims"][-1,1]

        if param_acc_dict.get(param_value, None) == None:
            param_acc_dict[param_value] = [0,0,0,0]
        if len(param_acc_dict.get(param_value)) <= 0 or train_accuracy > param_acc_dict.get(param_value, None)[0]:
            current_list = param_acc_dict[param_value]
            current_list[0] = train_accuracy
            param_acc_dict[param_value] = current_list
        if len(param_acc_dict.get(param_value)) <= 1 or valid_accuracy > param_acc_dict.get(param_value, None)[1]:
            current_list = param_acc_dict[param_value]
            current_list[1] = valid_accuracy
            param_acc_dict[param_value] = current_list
        if len(param_acc_dict.get(param_value)) <= 2 or wl_correlation > param_acc_dict.get(param_value, None)[2]:
            current_list = param_acc_dict[param_value]
            current_list[2] = wl_correlation
            param_acc_dict[param_value] = current_list
        if len(param_acc_dict.get(param_value)) <= 3 or lii_correlation > param_acc_dict.get(param_value, None)[3]:
            current_list = param_acc_dict[param_value]
            current_list[3] = lii_correlation
            param_acc_dict[param_value] = current_list

    npz = np.load('all-results/results-2-3-22/runs/a0.01lds19b0.007g1.0d1.0h13.npz')
    data = npz["tl"]
    param_value = npz["par"][param_idx]
    train_accuracy = data[:,5][-1]
    valid_accuracy = npz["vl"][:,5][-1]
    wl_correlation = npz["wldims"][-1,0]
    lii_correlation = npz["liidims"][-1,1]
    param_acc_dict[optimal_param_dict.get(param_idx)] = [train_accuracy, valid_accuracy, wl_correlation, lii_correlation]

    param_acc_dict = collections.OrderedDict(sorted(param_acc_dict.items()))
    param_vals = [float(i) for i in param_acc_dict.keys()]
    train_accuracy = [param_acc_dict.get(i)[0] for i in param_vals]
    valid_accuracy = [param_acc_dict.get(i)[1] for i in param_vals]
    wl_correlation_vals = [param_acc_dict.get(i)[2] for i in param_vals]
    lii_correlation_vals = [param_acc_dict.get(i)[3] for i in param_vals]

    fontsize = 25

    fig, ax = plt.subplots(figsize=(30,22))
    par_vs_acc_line, = ax.plot(param_vals, train_accuracy, linewidth=20, zorder=1)
    par_vs_acc_scatter = ax.scatter(param_vals, train_accuracy, s=3000, zorder=2)
    par_vs_vacc_line, = ax.plot(param_vals, valid_accuracy, linewidth=20, zorder=1, linestyle='dashed', dashes=(5,10))
    par_vs_vacc_scatter = ax.scatter(param_vals, valid_accuracy, s=3000, zorder=2)
    special_var = ax.scatter(optimal_param_dict.get(param_idx), 
    train_accuracy[param_vals.index(optimal_param_dict.get(param_idx))], color='red', marker='s', s=3000, zorder=2)
    val_special_var = ax.scatter(optimal_param_dict.get(param_idx), 
    valid_accuracy[param_vals.index(optimal_param_dict.get(param_idx))], color='red', marker='s', s=3000, zorder=2)

    x_label_name = None
    if param_name == 'latentDims':
        x_label_name = '|z|'
    elif param_name == 'lstmLayers':
        x_label_name = 'Number LSTM Layers'
    elif param_name == 'hiddenSize':
        x_label_name = 'LSTM Info (h/2)'

    else:
        x_label_name = param_name[0].upper() + param_name[1:]

    if param_name == 'alpha':
        ax.set_xlabel('\u03B1', fontsize=10*fontsize)
    elif param_name == 'gamma':
        ax.set_xlabel('\u03B3', fontsize=10*fontsize)
    elif param_name == 'delta':
        ax.set_xlabel('\u03B4', fontsize=10*fontsize)
    elif param_name == 'beta':
        ax.set_xlabel('\u03B2', fontsize=10*fontsize)
    elif param_name == 'lstmLayers' or param_name == 'hiddenSize':
        ax.set_xlabel(x_label_name, fontsize=8*fontsize)
    else:
        ax.set_xlabel(x_label_name, fontsize=10*fontsize)

    ax.set_ylabel("Accuracy", fontsize=10*fontsize)

    if param_name == 'beta':
        plt.xticks(ticks=np.arange(0.006, 0.009, 0.001))
    elif param_name == 'alpha':
        plt.xticks(ticks=np.arange(0.00, .21, .1))
    elif param_name == 'latentDims':
        plt.xticks(ticks=np.arange(17.0,21.01,1))
    elif param_name == 'hiddenSize':
        plt.xticks(ticks=np.arange(min(param_vals), max(param_vals) + 1, 2.0))
    elif param_name == 'gamma':
        plt.xticks(ticks=np.arange(0.5, 2.01, .5))
    elif param_name == 'delta':
        plt.xticks(np.arange(min(param_vals), max(param_vals) + 0.01, 1.5))
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))

    plt.xticks(fontsize= 8*fontsize)
    plt.yticks(fontsize= 8*fontsize)

    min_in_y_range = min(train_accuracy) - .10 * min(train_accuracy) if min(train_accuracy) - .10 * min(train_accuracy) >= 0 else 0
    max_in_y_range = max(train_accuracy) + .10 * max(train_accuracy) if max(train_accuracy) + .10 * max(train_accuracy) < 1 else 1

    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    
    if x_label_name == '|z|':
        plt.legend(handles=[par_vs_acc_line, par_vs_vacc_line], labels=['Training', 'Validation'],  fontsize=5.5*fontsize, loc=0)
    # ax.legend([par_vs_wl_line, par_vs_lii_line, special_var], ['WAV',
    # 'LII'],  fontsize=6*fontsize, bbox_to_anchor=(0,0,1,1), loc=7)


    fig.savefig(f"paper-materials/hyper-param-experiments/final-associated-files/{param_name}-vs-accuracy-{time.time()}.png", bbox_inches='tight')
    fig.savefig(f"paper-materials/hyper-param-experiments/final-associated-files/{param_name}-vs-accuracy-{time.time()}.eps", bbox_inches='tight')

    fontsize = 25

    fig, ax = plt.subplots(figsize=(30,22))
    par_vs_wl_line, = ax.plot(param_vals, wl_correlation_vals, color='black', linewidth=20, zorder=1)
    par_vs_wl_scatter = ax.scatter(param_vals, wl_correlation_vals, color='black', s=3000, zorder=2)
    special_var = ax.scatter(optimal_param_dict.get(param_idx), 
    wl_correlation_vals[param_vals.index(optimal_param_dict.get(param_idx))], color='red', marker='s', s=3000, zorder=2)

    
    par_vs_lii_line, = ax.plot(param_vals, lii_correlation_vals, color = 'black', linewidth=20, linestyle='dashed', dashes=(2.5,5), zorder=1)
    par_vs_lii_scatter = ax.scatter(param_vals, lii_correlation_vals, color = 'black', s=3000, zorder=2)
    ax.scatter(optimal_param_dict.get(param_idx), 
    lii_correlation_vals[param_vals.index(optimal_param_dict.get(param_idx))], color='red', marker='s', s=3000, zorder=2)

    x_label_name = None
    if param_name == 'latentDims':
        x_label_name = '|z|'
    elif param_name == 'lstmLayers':
        x_label_name = 'Number LSTM Layers'
    elif param_name == 'hiddenSize':
        x_label_name = 'LSTM Info (h/2)'
    else:
        x_label_name = param_name[0].upper() + param_name[1:]

    if param_name == 'alpha':
        ax.set_xlabel('\u03B1', fontsize=10*fontsize)
    elif param_name == 'gamma':
        ax.set_xlabel('\u03B3', fontsize=10*fontsize)
    elif param_name == 'delta':
        ax.set_xlabel('\u03B4', fontsize=10*fontsize)
    elif param_name == 'beta':
        ax.set_xlabel('\u03B2', fontsize=10*fontsize)
    elif param_name == 'lstmLayers' or param_name == 'hiddenSize':
        ax.set_xlabel(x_label_name, fontsize=8*fontsize)
    else:
        ax.set_xlabel(x_label_name, fontsize=10*fontsize)
        
    ax.set_ylabel("Correlation", fontsize=10*fontsize)

    if param_name == 'beta':
        plt.xticks(ticks=np.arange(0.006, 0.009, 0.001))
    elif param_name == 'alpha':
        plt.xticks(ticks=np.arange(0.00, .21, .1))
    elif param_name == 'latentDims':
        plt.xticks(ticks=np.arange(16.0,21.01,1))
    elif param_name == 'hiddenSize':
        plt.xticks(ticks=np.arange(min(param_vals), max(param_vals) + 1, 2.0))
    elif param_name == 'gamma':
        plt.xticks(ticks=np.arange(0.5, 2.01, .5))
    elif param_name == 'delta':
        plt.xticks(np.arange(min(param_vals), max(param_vals) + 0.01, 1.5))
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))

    plt.xticks(fontsize= 8*fontsize)
    plt.yticks(fontsize= 8*fontsize)
    ax.legend([par_vs_wl_line, par_vs_lii_line, special_var], ['WAV',
    'LII'],  fontsize=6*fontsize, bbox_to_anchor=(0,0,1,1), loc=7)

    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    fig.savefig(f"paper-materials/hyper-param-experiments/final-associated-files/{param_name}-vs-level-of-correlation-{time.time()}.png", bbox_inches='tight')
    fig.savefig(f"paper-materials/hyper-param-experiments/final-associated-files/{param_name}-vs-level-of-correlation-{time.time()}.eps", bbox_inches='tight')


def retrieve_needed_files_from_directory(directory_to_search: str, file_extension: str) -> tuple:
    """This function gathers all of files in a directory with a given extension into a returned tuple."""
    retrieved_files = []
    for file_path in os.listdir(directory_to_search):
        file_path = os.path.join(directory_to_search, file_path)
        if file_extension in file_path:
            retrieved_files.append(file_path)
        elif os.path.isdir(file_path):
            returned_tuple = retrieve_needed_files_from_directory(file_path, file_extension)
            for elem in returned_tuple:
                retrieved_files.append(elem)
    return tuple(retrieved_files)


def generate_acc_vs_param_plots(directory_to_search: str):
    """This function is used to generate hyperparameter vs. accuracy plots for all identified hyperparamters present
    in the given directory"""
    param_dict = {"alpha":0, "beta": 1, "gamma": 2, "delta": 3, "latentDims": 4, "lstmLayers": 5, 
    "dropout":6, "hiddenSize":7}
    param_list = param_dict.keys()
    for param in param_list:
        file_list = retrieve_needed_files_from_directory(directory_to_search, ".npz")
        generate_acc_vs_param_plot(file_list, param_dict.get(param), param)


def npz_testing(path: str) -> None:
    """Just a testing function to see the contents of an npz file"""
    npz_contents = np.load(path)
    print(npz_contents)


def pt_testing(path: str) -> None:
    model_contents = torch.load(path)
    print(model_contents)


# def write_excel_data_base_files_to_csv(origin_file_path: str, keyword: str) -> None:
#     path = "data-and-cleaning/excel-data-files"
#     if not os.path.isdir(path):
#         os.mkdir(path)  
#     file_contents = pd.read_excel(origin_file_path, engine='openpyxl', sheet_name=None)
#     sheet_names = file_contents.keys()
#     sheet_names = [i for i in sheet_names]
#     for sheet_name in file_contents:
#         if keyword in sheet_name:
#             current_df = file_contents.get(sheet_name)
#             column_name_list = ['Sequence', 'Peak 1 [nm]', 'Peak 1 area']
#             seq_col = current_df.loc[:,column_name_list[0]].to_numpy()
#             wv_col = current_df.loc[:, column_name_list[1]].to_numpy()
#             lii_col = current_df.loc[:, column_name_list[2]].to_numpy()
#             current_df = pd.DataFrame(data=np.array([seq_col, wv_col, lii_col]).transpose())
#             current_df = current_df.rename({i: column_name_list[i] for i in range(len(column_name_list))}, axis='columns')
#             current_df = current_df.dropna(axis='rows')
#             current_df.to_csv(f"{path}/{sheet_name.replace(' ', '-')}-base.csv", index=False)


# def write_padded_8_base_files(path_to_8_base_norm: str, *paths_to_write_new_files: str):
#     df = pd.read_csv(path_to_8_base_norm)
#     padding_character = "0"
#     front_padding = f"{padding_character}{padding_character}"
#     back_padding = ''
#     column_name_list = ['Sequence', 'Wavelen', 'LII']
#     for path in paths_to_write_new_files:
#         column_to_pad = np.array(df.loc[:,'Sequence'])
#         for i, row in enumerate(column_to_pad):
#             column_to_pad[i] = front_padding + row + back_padding
#         front_padding = front_padding[1:]
#         back_padding += padding_character
#         wv_arr = df.loc[:,'Peak 1 [nm]']
#         lii_arr = df.loc[:, 'Peak 1 area']
#         new_df_arr = np.array([column_to_pad, wv_arr, lii_arr]).transpose()
#         current_df = pd.DataFrame(new_df_arr)
#         current_df = current_df.rename({i: column_name_list[i] for i in range(len(column_name_list))}, axis='columns')
#         current_df.to_csv(f"data-and-cleaning/excel-data-files/{path}", index=False)


# def write_sliding_window_files(path_to_above_10_norm: str, path_to_write_new_file: str):
#     df = pd.read_csv(path_to_above_10_norm)
#     sequence_column = np.array(df.loc[:,'Sequence'])
#     new_col = np.array([])
#     column_name_list = ['Sequence', 'Wavelen', 'LII']
#     wv_arr = np.array(df.loc[:,'Peak 1 [nm]'])
#     lii_arr = np.array(df.loc[:, 'Peak 1 area'])
#     new_wv_arr = np.array([])
#     new_lii_arr = np.array([])
#     max_idx = len(sequence_column[0])
#     for i, sequence in enumerate(sequence_column):
#         start = 0
#         end = 10
#         while end <= max_idx:
#             new_col = np.append(new_col, sequence[start:end])
#             new_wv_arr = np.append(new_wv_arr, wv_arr[i])
#             new_lii_arr = np.append(new_lii_arr, lii_arr[i])
#             start += 1
#             end += 1
#     new_df_arr = np.array([new_col, new_wv_arr, new_lii_arr]).transpose()
#     current_df = pd.DataFrame(new_df_arr)
#     current_df = current_df.rename({i: column_name_list[i] for i in range(len(column_name_list))}, axis='columns')
#     current_df.to_csv(path_to_write_new_file, index=False)

    
# def transform_sequences(seq_arr):
#     alphabet = ['A', 'C', 'G', 'T']
#     enc = OneHotEncoder()
#     enc.fit(np.array(alphabet).reshape(-1,1))
#     return enc.transform(seq_arr.reshape(-1,1)).toarray().reshape(
#         -1, 10, len(alphabet))



# def process_editted_data_file(path_to_dataset: str):
#     df = pd.read_csv(path_to_dataset)
#     seq_arr = df.loc[:,'Sequence']
    
#     ohe_sequences = transform_sequences(seq_arr.apply(lambda x: pd.Series
#     ([c if c != '0' else random.choice(['A', 'C', 'G', 'T']) for c in x])).to_numpy())
#     Wavelen = np.array(df.loc[:,'Wavelen'])
#     LII = np.array(df.loc[:,'LII'])

#     np.savez(f"data-for-sampling/processed-data-files/{path_to_dataset[path_to_dataset.rindex('/'):path_to_dataset.rindex('.')]}", Wavelen=Wavelen, LII=LII, ohe=ohe_sequences)


# def process_data(path_to_dataset):
#     """Basic wrapper for loading the given dataset using numpy"""
#     data = np.load(path_to_dataset)
#     return data


# def process_model(path_to_model):
#     """Basic wrapper for loading the archived .pt pytorch model"""
#     model = torch.load(path_to_model)
#     return model

# def classify_sequences(z_arr: np.ndarray, wavelength_arr: np.ndarray) -> np.ndarray:
#     class_array = [[], [], [], []] # in the order green, red, very red, near ir
#     for i, wavelength in enumerate(wavelength_arr):
#         if wavelength <= 590:
#             class_array[0].append(z_arr[i])
#         elif 590 < wavelength <= 660:
#             class_array[1].append(z_arr[i])
#         elif 660 < wavelength <= 800:
#             class_array[2].append(z_arr[i])
#         elif wavelength > 800:
#             class_array[3].append(z_arr[i])
#     return np.array(class_array)


# def generate_violin_plot():
#     pass


# def generate_violin_plots(npz_a: str, npz_b: str, path_to_model: str, identifier: str, isWavelengthPlot: bool):
#     npz_a = process_data(npz_a)
#     npz_b = process_data(npz_b)
#     model = process_model(path_to_model)

#     latent_dist_a = model.encode(torch.from_numpy(npz_a['ohe']))
#     latent_dist_b = model.encode(torch.from_numpy(npz_b['ohe']))

#     fig, ax = plt.subplots(figsize=(20,20))

#     if (isWavelengthPlot):
#         z_wav_dim_a = latent_dist_a.mean.detach().numpy()[:,0]
#         z_wav_dim_b = latent_dist_b.mean.detach().numpy()[:,0]

#         wavelength_class_a = classify_sequences(z_wav_dim_a, npz_a['Wavelen'])
#         wavelength_class_b = classify_sequences(z_wav_dim_b, npz_b['Wavelen'])

#         for i, arr in enumerate(wavelength_class_a):
#             for j, element in enumerate(arr):
#                 wavelength_class_a[i][j] = float(element)
#         for i, arr in enumerate(wavelength_class_b):
#             for j, element in enumerate(arr):
#                 wavelength_class_b[i][j] = float(element)

    

#         data_arr = np.array([wavelength_class_a[0], wavelength_class_a[1], wavelength_class_a[2], wavelength_class_a[3], 
#         wavelength_class_b[0], wavelength_class_b[1], wavelength_class_b[2], wavelength_class_b[3]], dtype=object)

#         # ax.set_xticklabels(['Green T', 'Red T', 'Very Red T', 'Near IR T', 
#         # 'Green Var', 'Red Var', 'Very Red Var'])
#         ax.set_ylabel("Proxy for Wavelength", fontsize=50)
#         ax.set_xlabel('Wavelength Class', fontsize=50)

#     else:
#         z_lii_dim_a = latent_dist_a.mean.detach().numpy()[:,1]
#         z_lii_dim_b = latent_dist_b.mean.detach().numpy()[:,1]


#     initial = 0.7
#     step = 0.7
#     if data_arr[0]:
#         p1 = ax.violinplot(data_arr[0], showmedians=True, positions=[initial], showmeans=True, showextrema=True)
#         for elem in p1['bodies']:
#             p1['cbars'].set_color(['black'])
#             p1['cmins'].set_color(['black'])
#             p1['cmaxes'].set_color(['black'])
#             elem.set_facecolor('green')
#             elem.set_edgecolor('black')
#             elem.set_alpha(1)
#     if data_arr[1]:
#         p2 = ax.violinplot(data_arr[1], showmedians=True, positions=[initial + step])
#         for elem in p2['bodies']:
#             p2['cbars'].set_color(['black'])
#             p2['cmins'].set_color(['black'])
#             p2['cmaxes'].set_color(['black'])
#             elem.set_facecolor('#DE3163')
#             elem.set_edgecolor('black')
#             elem.set_alpha(1)
#     if data_arr[2]:
#         p3 = ax.violinplot(data_arr[2], showmedians=True, positions=[initial + 2*step])
#         for elem in p3['bodies']:
#             p1['cbars'].set_color(['black'])
#             p1['cmins'].set_color(['black'])
#             p1['cmaxes'].set_color(['black'])
#             elem.set_facecolor('#D2042D')
#             elem.set_edgecolor('black')
#             elem.set_alpha(1)
#     if data_arr[3]:
#         p4 = ax.violinplot(data_arr[3], showmedians=True, positions=[initial + 3*step])
#         for elem in p4['bodies']:
#             p1['cbars'].set_color(['black'])
#             p1['cmins'].set_color(['black'])
#             p1['cmaxes'].set_color(['black'])
#             elem.set_facecolor('blue')
#             elem.set_edgecolor('black')
#             elem.set_alpha(1)
#     if data_arr[4]:
#         p5 = ax.violinplot(data_arr[4], showmedians=True, positions=[initial + 4*step])
#         for elem in p5['bodies']:
#             p1['cbars'].set_color(['black'])
#             p1['cmins'].set_color(['black'])
#             p1['cmaxes'].set_color(['black'])
#             elem.set_facecolor('green')
#             elem.set_edgecolor('black')
#             elem.set_alpha(1)
#     if data_arr[5]:
#         p6 = ax.violinplot(data_arr[5], showmedians=True, positions=[initial + 5*step])
#         for elem in p6['bodies']:
#             p1['cbars'].set_color(['black'])
#             p1['cmins'].set_color(['black'])
#             p1['cmaxes'].set_color(['black'])
#             elem.set_facecolor('#DE3163')
#             elem.set_edgecolor('black')
#             elem.set_alpha(1)
#     if data_arr[6]:
#         p7 = ax.violinplot(data_arr[6], showmedians=True, positions=[initial + 6*step])
#         for elem in p7['bodies']:
#             p1['cbars'].set_color(['black'])
#             p1['cmins'].set_color(['black'])
#             p1['cmaxes'].set_color(['black'])
#             elem.set_facecolor('#D2042D')
#             elem.set_edgecolor('black')
#             elem.set_alpha(1)
#     if data_arr[7]:
#         p8 = ax.violinplot(data_arr[7], showmedians=True, positions=[initial + 7*step])
#         for elem in p8['bodies']:
#             p1['cbars'].set_color(['black'])
#             p1['cmins'].set_color(['black'])
#             p1['cmaxes'].set_color(['black'])
#             elem.set_facecolor('blue')
#             elem.set_edgecolor('black')
#             elem.set_alpha(1)

#     # ax.xaxis.set_major_locator(plt.MaxNLocator(len(data_arr)))
#     labels = ['a', 'b', 'c', 'd', 
#         'e', 'f', 'g']
#     # pos, _ = plt.xticks()
#     # plt.xticks(ticks=pos, labels=labels)
#     # print(pos)
#     # print(labels)

#     fig.savefig(f"paper-materials/{identifier}.png")
#     fig.savefig(f"paper-materials/{identifier}.eps")

#     # #TODO: Continue on from here, construct violin plots you have all the data



def main():
    pass
    # param_arr = ["alpha", "beta", "gamma", "delta", "latentDims", "lstmLayers", "dropout", "hiddenSize"]
    # param_dict = {"alpha": [], "beta": [], "gamma": [], "delta": [], 
    #      "latentDims": [], "lstmLayers": [], "dropout": [], "hiddenSize": []}
    # param_dict = gather_hyperparameter_spreads('1-12-weighted-non-kfold-results', param_arr=param_arr, parameter_dictionary=param_dict)
    # print(param_dict) DONE

    # gather_training_data_info('data-for-sampling/processed-data-files/clean-data-base.npz', '1-20-22-models-info/a19lds19b0.007g1.0d1.0h13.pt')
    # gather_sampled_metrics('data-for-sampling/past-samples-with-info/samples-model-2/detailed-sequences-model-2')
    # create_wavelength_lii_dist_plots('data-for-sampling/processed-data-files/clean-data-base.npz',
    # 'all-results/1-20-22-models-info/a19lds19b0.007g1.0d1.0h13.pt','data-for-sampling/past-samples-with-info/samples-model-2/detailed-sequences-model-2') #DONE

    generate_acc_vs_param_plots(directory_to_search='all-results/results-2-3-22')

    generate_acc_vs_param_plot_lin_dims(['all-results/results-2-9-22/runs/a0.01lds19b0.007g1.0d1.0h13lin12.npz',
    'all-results/results-2-3-22/runs/a0.01lds19b0.007g1.0d1.0h13.npz',
    'all-results/results-2-9-22/runs/a0.01lds19b0.007g1.0d1.0h13lin20.npz'])

    #write_excel_data_base_files_to_csv('data-and-cleaning/210623 all lengths all peaks.xlsx', keyword="norm")

    # write_padded_8_base_files('data-and-cleaning/excel-data-files/8-base_norm-base.csv', 
    # 'two-pad-front-8-base-norm.csv', 'one-and-one-8-base-norm.csv', 'two-pad-back-8-base-norm.csv')

    # write_sliding_window_files('data-and-cleaning/excel-data-files/12-base_norm-base.csv',
    # 'data-and-cleaning/excel-data-files/12-base_norm-sliding-window-result.csv')

    # write_sliding_window_files('data-and-cleaning/excel-data-files/16-base_norm-base.csv',
    # 'data-and-cleaning/excel-data-files/16-base_norm-sliding-window-result.csv')

    # process_editted_data_file('data-and-cleaning/excel-data-files/two-pad-front-8-base-norm.csv')
    # process_editted_data_file('data-and-cleaning/excel-data-files/one-and-one-8-base-norm.csv')
    # process_editted_data_file('data-and-cleaning/excel-data-files/two-pad-back-8-base-norm.csv')
    # process_editted_data_file('data-and-cleaning/excel-data-files/12-base_norm-sliding-window-result.csv')
    # process_editted_data_file('data-and-cleaning/excel-data-files/16-base_norm-sliding-window-result.csv')

    # generate_violin_plots(npz_a='data-for-sampling/processed-data-files/clean-data-base.npz',
    # npz_b='data-for-sampling/processed-data-files/two-pad-front-8-base-norm.npz', 
    # path_to_model='all-results/1-20-22-models-info/a19lds19b0.007g1.0d1.0h13.pt', identifier='8-base-2-pad-front', isWavelengthPlot=True)

    # generate_violin_plots(npz_a='data-for-sampling/processed-data-files/clean-data-base.npz',
    #     npz_b='data-for-sampling/processed-data-files/two-pad-back-8-base-norm.npz', 
    #     path_to_model='all-results/1-20-22-models-info/a19lds19b0.007g1.0d1.0h13.pt', identifier='8-base-2-pad-back', isWavelengthPlot=True)

    # generate_violin_plots(npz_a='data-for-sampling/processed-data-files/clean-data-base.npz',
    #     npz_b='data-for-sampling/processed-data-files/one-and-one-8-base-norm.npz', 
    #     path_to_model='all-results/1-20-22-models-info/a19lds19b0.007g1.0d1.0h13.pt', identifier='1-and-1-8-base', isWavelengthPlot=True)

    # generate_violin_plots(npz_a='data-for-sampling/processed-data-files/clean-data-base.npz',
    #     npz_b='data-for-sampling/processed-data-files/12-base_norm-sliding-window-result.npz', 
    #     path_to_model='all-results/1-20-22-models-info/a19lds19b0.007g1.0d1.0h13.pt', identifier='12-base-norm', isWavelengthPlot=True)

    # generate_violin_plots(npz_a='data-for-sampling/processed-data-files/clean-data-base.npz',
    #     npz_b='data-for-sampling/processed-data-files/16-base_norm-sliding-window-result.npz', 
    #     path_to_model='all-results/1-20-22-models-info/a19lds19b0.007g1.0d1.0h13.pt', identifier='16-base-norm', isWavelengthPlot=True)


main()
    