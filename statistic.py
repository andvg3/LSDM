import os

files = os.listdir('data/proxd_valid/vertices_can')
files = list(map(lambda x: x[:-14], files))
print(len(files))
for file in files:
    print(file)
