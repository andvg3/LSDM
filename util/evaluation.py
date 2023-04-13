from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def emd(x, y):
    if len(x.shape) == 3:
        x = x.squeeze(0)
        y = y.squeeze(0)
    d = cdist(x, y)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / min(len(x), len(y))

def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
      correct_k = correct[:k].reshape(-1).float().sum(0)
      res.append(correct_k.mul_(100.0 / batch_size))
  return res
