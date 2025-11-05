import pandas as pd
def write_unique(path_to_sequences: str):
    sequences_dict = {}
    with open(path_to_sequences, 'r+') as f:
        f.readline()  # Omits the header row for the names of columns
        read_data = f.readlines()
        for i, line in enumerate(read_data):
            split = line.split(',')
            seq = split[0]
            sequences_dict[seq] = seq
        f.truncate(0)

    with open(path_to_sequences, 'r+') as f:
        f.write("Sequence,Label,class\n")
        keys = sequences_dict.keys()
        for key in keys:
            # print("key    ", key)
            f.write(key)


def fill_training_data_dict(path_to_data: str):
    data_set_dict = {}
    df = pd.read_excel(path_to_data).to_numpy()

    
    for seq in df[:,0]:
        data_set_dict[seq] = seq

    return data_set_dict


def remove_duplicate(data_set_dict: dict, path_to_sequences: str):
    file_contents = None
    with open(path_to_sequences, 'r+') as f:
        file_contents = f.readlines()
        f.truncate(0)
    
    with open(path_to_sequences, 'r+') as f:
        f.write(file_contents[0])
        file_contents = file_contents[1:]

        for line in file_contents:
            split = line.split(',')
            seq = split[0]
            try:
                if data_set_dict.get(line) is None:
                    f.write(seq)
            except:
                print("ERROR ENCOUNTERED")