import numpy as np
import os

dict_values = {}
files = os.listdir("predictions/proxd_valid")
for file in files:
    vertices = np.load(open("predictions/proxd_valid/{}".format(file), "rb"))
    for x in vertices:
        for y in x:
            for z in y:
                if z not in dict_values:
                    dict_values[z] = None
    # print(vertices.data.shape)
print(dict_values)