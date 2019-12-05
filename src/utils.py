import random
import numpy as np
from pdb import set_trace as pause

def parse_dataset(file):
    data = np.genfromtxt(file, delimiter=',', skip_header=2)
    problem_args = list(np.genfromtxt(file, skip_footer=len(data), dtype=str))
    dims = int(problem_args[0].split(':')[1])
    classes = int(problem_args[1].split(':')[1])
    dataset = {
        'args': {
            'dims': dims,
            'classes': classes
        },
        'data': data
    }

    return dataset

def get_data_label(dataset):
    data = []
    labels = []

    for i in range(len(dataset['data'])):
        _data, _labels = get_row(dataset, i)
        data.append(_data)
        labels.append(_labels)

    return (data, labels)

def get_row(dataset, index=-1):
    resp = dataset['data'][random.randrange(len(dataset['data']))] if index == -1 else dataset['data'][index]
    label = resp[dataset['args']['dims']:]

    return resp[:dataset['args']['dims']], np.argmax(label) if len(label) > 1 else int(label[0])
