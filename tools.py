import torch
import os

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    return device


def write_data(file_path,
               data_list: list or tuple):
    if os.path.exists(file_path):
        raise FileExistsError("File: {} already exists.".format(file_path))
    else:
        with open(file_path, 'w') as f:
            for data in data_list:
                f.write(data + '\n')