import numpy as np
import json

def read_landmark(path):
    with open(path) as f:
        file_content = json.load(f)
        coords = np.array(file_content)
        coords_reshaped = np.transpose(coords)
        return np.transpose(coords_reshaped[0:2]).flatten()

def print_confusion_matrix(matrix, labels):
    result = ""
    for i, row in enumerate(matrix):
        support = sum(row)
        print("support", i, support)
        if (support > 0):
            for el in row:
                result += "%.2f" % round(float(el) / support * 100,2) + '\t'
            result += '\n'
    return result

# read_landmark("/home/npc/marta/mediapipe/npc_27_08_przyciete_227/filmy_ds_1_l.mov/90/landmark.txt")