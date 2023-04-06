import torch
import open3d as o3d
import numpy as np


def translate_obj_to_bbox(obj):
    """
    Input: obj: torch.Tensor([num_points, 3])
    Output: translations: torch.Tensor([3])
            sizes: torch.Tensor([3])
    """
    xyz = obj.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
    return bbox.center, bbox.extent

def translate_objs_to_bbox(objs, mask):
    """
    Input: obj: torch.Tensor([bs, num_objs, num_points, 3])
    Output: translations: torch.Tensor([bs, num_objs, 3])
            sizes: torch.Tensor([bs, num_objs, 3])
    """
    bs, num_objs, _, _ = objs.shape
    translations = torch.zeros(bs, num_objs, 3)
    sizes = torch.zeros(bs, num_objs, 3)
    for i, objs_batch in enumerate(objs):
        mask_batch = mask[i]
        for j, obj in enumerate(objs_batch):
            if mask_batch[j] == 0:
                break
            translation, size = translate_obj_to_bbox(obj)
            translations[i][j] = torch.tensor(translation)
            size[i][j] = torch.tensor(size)

    return translations, sizes
    
def translate_target_obj_to_bbox(obj):
    """
    Input: obj: torch.Tensor([bs, num_points, 3])
    Output: translations: torch.Tensor([bs, 3])
            sizes: torch.Tensor([bs, 3])
    """
    translations = []
    sizes = []
    for obj_batch in obj:
        translation, size = translate_obj_to_bbox(obj_batch)
        translations.append(translation)
        sizes.append(size)
    
    translations = torch.tensor(translations)
    sizes = torch.tensor(sizes)
    return translations, sizes

def translate_bbox_obj(translation, size, point_size=1024):
    """
    Input: translation: numpy.array([3])
            size: numpy.array([3])
    Output: obj: torch.tensor([1024, 3])
    """
    bs  = size.shape[0]
    device = translation.device
    obj = []
    size = size.squeeze(1)
    translation = translation.squeeze(1)
    for pnt in range(point_size):
        xyz = (torch.rand(bs, 3).to(device) - 0.5) * size + translation
        xyz = xyz.unsqueeze(1)
        obj.append(xyz)
    obj = torch.cat(obj, dim=1)
    return obj

