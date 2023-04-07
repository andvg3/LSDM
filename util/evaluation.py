from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def emd(x, y):
    if len(x.shape) == 3:
        x = x.squeeze(0)
        y = y.squeeze(0)
    d = cdist(x, y)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / min(len(x), len(y))
