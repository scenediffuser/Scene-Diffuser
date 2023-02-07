
import glob
import os
import pickle
import numpy as np

pkls = glob.glob(os.path.join('/home/wangzan/Data/SceneDiffuser/scannet_path_planning/path/', '*.pkl'))
print(len(pkls))

paths = 0
path_length = []

for p in pkls:
    with open(p, 'rb') as fp:
        data = pickle.load(fp)
        paths += len(data)

        for coarse_path, refined_path in data:
            path_length.append(len(refined_path))

print(paths)
print(np.mean(path_length))
print(np.max(path_length))
print(np.min(path_length))

hist, bin_edges = np.histogram(path_length, bins=list(range(1, 123)))

print(hist)
print(hist.sum())

print('-' * 30)

region = []

files = glob.glob('/home/wangzan/Data/SceneDiffuser/scannet_path_planning/height/*.pkl')
for p in files:
    with open(p, 'rb') as fp:
        data = pickle.load(fp)
    
    minx, miny = data['minx'], data['miny']
    maxx, maxy = data['maxx'], data['maxy']

    region.append(max(maxx - minx, maxy - miny))

region = np.array(region)
print(region.mean() / 2)
print(np.median(region) / 2)
print(region.max() / 2)
print(region.min() / 2)

