from posa.dataset import ProxDataset_txt, HUMANISE
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from util.evaluation import *
import numpy as np
import torch
import open3d as o3d

def transform_pcd(source, target):
    # Initialize hyperparams
    threshold = 0.02
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                            [-0.139, 0.967, -0.215, 0.7],
                            [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])

    # Initialize source and target point clouds
    dv = source.device
    source = source.squeeze(0).cpu().numpy()
    target = target.squeeze(0).cpu().numpy()
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(source)
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_source, pcd_target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000),)
    pcd_source = pcd_source.transform(reg_p2p.transformation)

    # Retrieve transformed points
    target_obj = pcd_source.points
    target_obj = torch.from_numpy(np.asarray(target_obj)).unsqueeze(0).to(dv)
    return target_obj


def get_gt_obj(keyword: str, origin_obj):
    target_cat = torch.zeros(1, 13)
    obj_files = {
        'rectangle table': ('BasementSittingBooth/table_0', 2),
        'round table': ('MPH8/table_1', 2),
        'square table': ('N0SittingBooth/table_0', 2),
        'two seater sofa': ('MPH8/sofa_0', 4),
        'single bed': ('MPH8/bed_0', 5),
        'meeting table': ('MPH1Library/table_0', 2),
        'eames chair': ('MPH1Library/chair_3', 1),
        'office chair': ('MPH11/chair_0', 1),
        'side cabinet': ('MPH11/cabinet_0', 3),
        'file cabinet': ('MPH11/shelving_0', 3),
        'chest of drawers': ('MPH112/chest_of_drawers_1', 6),
        'double bed': ('MPH112/bed_0', 5),
        'sofa stool': ('N0Sofa/sofa_0', 4),
        'cafe table': ('N0Sofa/table_0', 2),
        'one seater sofa': ('N0Sofa/sofa_2', 4),
        'wall table': ('N3Library/furniture_0', 2),
        'desk': ('N3Office/table_0', 2),
        'monitor': ('N3Office/tv_monitor_0', 8),
        'accent chair': ('N3OpenArea/chair_2', 1),
        'accent table': ('N3OpenArea/table_0', 2),
        'recliner': ('MPH1Library/chair_3', 1),
        'dining chair': ('N0SittingBooth/seating_0', 1),
    }
    obj_folder = 'data/protext/objs'
    if keyword not in obj_files:
        return False
    obj_handle, obj_cat = obj_files[keyword]
    with open(os.path.join(obj_folder, obj_handle + '.npy'), 'rb') as f:
        target_obj = np.load(f)
        target_obj = torch.from_numpy(target_obj)
        target_obj = target_obj.unsqueeze(dim=0)
        target_cat[0][obj_cat] = 1
    target_obj = target_obj.to(origin_obj.device)
    target_cat = target_cat.to(origin_obj.device)
    # Perform transformation method
    target_obj = transform_pcd(target_obj, origin_obj)
    return target_obj, target_cat

data_dir = 'data/protext/proxd_test'
output_dir = 'training/sdm_bert_dgcnn_p2r/output/predictions'
# data_dir = 'data/humanise/valid/'
valid_dataset = ProxDataset_txt(data_dir, max_frame=256, fix_orientation=True,
                                    step_multiplier=1, jump_step=8)
valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

# valid_dataset = HUMANISE(data_dir, max_frame=256, fix_orientation=True,
#                                     step_multiplier=1, jump_step=8)
# valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)


f1_list = []
# Setup names for output files
context_dir = os.path.join(data_dir, 'context')
files = os.listdir(context_dir)
lookup_tab = dict()
for file in files:
    reduced_file = file.split('.')[0]

    with open(os.path.join(context_dir, file), 'r') as f:
        prompt = f.readlines()[0].strip()
        lookup_tab[prompt] = reduced_file

for mask, given_objs, given_cats, target_obj, target_cat, y in tqdm(valid_data_loader):
    seq_name = lookup_tab[y[0]]
    given_objs = given_objs.view(-1, 3)
    with open(os.path.join(output_dir, seq_name + '.npy'), 'rb') as f:
        pred = np.load(f)
    
    # tokens = y[0].split(' ')
    # # We will try from position 2 to 4
    # tokens = tokens[2:5]
    # current_keyword = ''
    # tokens = [tokens[0], tokens[0] + ' ' + tokens[1], tokens[0] + ' ' + tokens[1] + ' ' + tokens[2]]
    # for i in range(3):
    #     current_keyword = tokens[i]
    #     if get_gt_obj(current_keyword, target_obj):
    #         target_obj, target_cat = get_gt_obj(current_keyword, target_obj)
    #         break

    f1_score = calculate_fscore(torch.from_numpy(pred).squeeze(0), given_objs.detach().cpu())
    f1_list.append(f1_score[2])

print(sum(f1_list)/len(f1_list))