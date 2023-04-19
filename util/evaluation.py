from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import open3d

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

def calculate_fscore(gt_tensor, pr_tensor, th: float=0.1):
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    gt = open3d.geometry.PointCloud()
    gt.points = open3d.utility.Vector3dVector(gt_tensor.cpu().numpy())

    pr = open3d.geometry.PointCloud()
    pr.points = open3d.utility.Vector3dVector(pr_tensor.cpu().numpy())
    
    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)
    
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall+precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0

    return fscore, precision, recall


if __name__ == '__main__':
    import torch
    x = torch.rand(1024, 3).cuda()
    y = torch.rand(1024, 3).cuda()
    print(calculate_fscore(x, y))