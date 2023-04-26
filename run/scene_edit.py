import os
import math
import numpy as np
import argparse
import torch
import trimesh
import open3d as o3d
from pytorch3d.loss import chamfer_distance
from random import randrange
from tqdm import tqdm
import sklearn.cluster
from torch.utils.data import DataLoader

import posa.vis_utils as vis_utils
import posa.general_utils as general_utils
import posa.data_utils as du
from posa.dataset import ProxDataset_txt, HUMANISE

from util.model_util import create_model_and_diffusion
from util.evaluation import *
"""
Running sample:
python test_contactformer.py ../data/proxd_valid/ --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --model_name contactformer --fix_ori --test_on_valid_set --output_dir ../test_output
python test_contactformer.py ../data/proxd_valid/ --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --model_name contactformer --fix_ori --single_seq_name MPH112_00151_01 --save_video --output_dir ../test_output
"""
def list_mean(list):
    acc = 0.
    for item in list:
        acc += item
    return acc / len(list)

"""
We implement masking techniques for three editing applications: object modification, object displacement and shape alternation
"""
def get_mask_obj_mod(gt):
    return torch.zeros(*gt.shape).to(gt.device)

def get_mask_obj_dis(gt):
    return torch.zeros(*gt.shape).to(gt.device)

def get_mask_shape_alt(gt):
    size = gt.shape[1] // 4
    z = gt[:,:,-1]
    x_mask = torch.zeros(*gt.shape).to(gt.device)
    indices = torch.topk(z, size, largest=False).indices
    x_mask[:, indices, :] = 1
    return x_mask


def get_mask(gt, edit_type):
    if edit_type == "obj_mod":
        return get_mask_obj_mod(gt)
    elif edit_type == "obj_dis":
        return get_mask_obj_dis(gt)
    else:
        return get_mask_shape_alt(gt)


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
    target_obj, reg_p2p = transform_pcd(target_obj, origin_obj)
    return target_obj, target_cat, reg_p2p

def transform_pcd(source, target):
    # Initialize hyperparams
    threshold = 0.2

    # Initialize source and target point clouds
    dv = source.device
    source = source.squeeze(0).cpu().numpy()
    target = target.squeeze(0).cpu().numpy()
    vec = target.mean(0) - source.mean(0)
    source += vec
    best_set = -1
    best_trans = None
    best_reg = None
    for n_try in range(1000):
        trans_init = np.random.rand(4, 4)
        trans_init[:-1,-1] = vec
        trans_init[-1] = np.array([0.0, 0.0, 0.0, 1.0])
        pcd_source = o3d.geometry.PointCloud()
        pcd_source.points = o3d.utility.Vector3dVector(source)
        pcd_target = o3d.geometry.PointCloud()
        pcd_target.points = o3d.utility.Vector3dVector(target)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_source, pcd_target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000),)
        # pcd_source = pcd_source.transform(reg_p2p.transformation)
        if len(reg_p2p.correspondence_set) > best_set:
            best_trans = reg_p2p.transformation
            best_set = len(reg_p2p.correspondence_set)
            best_reg = reg_p2p
    
    pcd_source = pcd_source.transform(best_trans)
    # Retrieve transformed points
    target_obj = pcd_source.points
    target_obj = torch.from_numpy(np.asarray(target_obj)).unsqueeze(0).to(dv)
    return target_obj, best_reg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data_dir", type=str,
                        help="path to POSA_temp dataset dir")
    parser.add_argument("--load_model", type=str, default="training/contactformer/model_ckpt/best_model_recon_acc.pt",
                        help="checkpoint path to load")
    parser.add_argument("--posa_path", type=str, default="training/posa/model_ckpt/epoch_0349.pt")
    parser.add_argument("--visualize", dest="visualize", action='store_const', const=True, default=False)
    parser.add_argument("--scene_dir", type=str, default="data/scenes",
                        help="the path to the scene mesh")
    parser.add_argument("--tpose_mesh_dir", type=str, default="data/mesh_ds",
                        help="the path to the tpose body mesh (primarily for loading faces)")
    parser.add_argument("--save_video", dest='save_video', action='store_const', const=True, default=False)
    parser.add_argument("--output_dir", type=str, default="../test_output",
                        help="the path to save test results")
    parser.add_argument("--cam_setting_path", type=str, default="posa/support_files/ScreenCamera_0.json",
                        help="the path to camera settings in open3d")
    parser.add_argument("--single_seq_name", type=str, default="BasementSittingBooth_00142_01")
    parser.add_argument("--test_on_train_set", dest='do_train', action='store_const', const=True, default=False)
    parser.add_argument("--test_on_valid_set", dest='do_valid', action='store_const', const=True, default=False)
    parser.add_argument("--model_name", type=str, default="default_model",
                        help="The name of the model we are testing. This is also the suffix for result text file name.")
    parser.add_argument("--fix_ori", dest='fix_ori', action='store_const', const=True, default=False)
    parser.add_argument("--encoder_mode", type=int, default=1,
                        help="Encoder mode (different number represents different versions of encoder)")
    parser.add_argument("--decoder_mode", type=int, default=1,
                        help="Decoder mode (different number represents different versions of decoder)")
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--jump_step", type=int, default=8)
    parser.add_argument("--dim_ff", type=int, default=512)
    parser.add_argument("--f_vert", type=int, default=64)
    parser.add_argument("--max_frame", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--datatype", type=str, default="proxd")
    parser.add_argument("--edit_type", type=str, default="obj_mod")

    # Parse arguments and assign directories
    args = parser.parse_args()
    args_dict = vars(args)

    data_dir = args_dict['data_dir']
    scene_dir = args_dict['scene_dir']
    tpose_mesh_dir = args_dict['tpose_mesh_dir']
    ckpt_path = args_dict['load_model']
    save_video = args_dict['save_video']
    output_dir = args_dict['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    cam_path = args_dict['cam_setting_path']
    single_seq_name = args_dict['single_seq_name']
    model_name = args_dict['model_name']
    fix_ori = args_dict['fix_ori']
    encoder_mode = args_dict['encoder_mode']
    decoder_mode = args_dict['decoder_mode']
    do_train = args_dict['do_train']
    do_valid = args_dict['do_valid']
    visualize = args_dict['visualize']
    n_layer = args_dict['n_layer']
    n_head = args_dict['n_head']
    jump_step = args_dict['jump_step']
    max_frame = args_dict['max_frame']
    dim_ff = args_dict['dim_ff']
    f_vert = args_dict['f_vert']
    posa_path = args_dict['posa_path']
    seed = args_dict['seed']
    datatype = args_dict['datatype']
    edit_type = args_dict['edit_type']

    # Argument logic check
    if visualize and save_video:
        save_video = False
    if do_train or do_valid:
        save_video = False
        visualize = False

    device = torch.device("cuda")
    num_obj_classes = 8
    pnt_size = 1024
    # For fix_ori
    ds_weights = torch.tensor(np.load("posa/support_files/downsampled_weights.npy"))
    associated_joints = torch.argmax(ds_weights, dim=1)
    torch.manual_seed(seed)

    if datatype == 'proxd':
        valid_dataset = ProxDataset_txt(data_dir, max_frame=max_frame, fix_orientation=fix_ori,
                                    step_multiplier=1, jump_step=jump_step)
        valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        valid_dataset = HUMANISE(data_dir, max_frame=max_frame, fix_orientation=fix_ori,
                                    step_multiplier=1, jump_step=jump_step)
        valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)
    model, diffusion = create_model_and_diffusion(datatype)
    model.eval()
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Setup names for output files
    context_dir = os.path.join(data_dir, 'context')
    files = os.listdir(context_dir)
    lookup_tab = dict()
    for file in files:
        reduced_file = file.split('.')[0]

        with open(os.path.join(context_dir, file), 'r') as f:
            prompt = f.readlines()[0].strip()
            lookup_tab[prompt] = reduced_file
    
    seq_name_list = []
    chamfer_list = []
    emd_list = []
    f1_list = []
    total_acc = []
    total_topk_acc = []
    # Evaluation for reg transformation
    reg_fitness = []
    reg_mse = []
    reg_set = []
    seq_class_acc = [[] for _ in range(num_obj_classes)]
    
    f = open(os.path.join(output_dir, "results.txt"), "w+")
    for mask, given_objs, given_cats, target_obj, target_cat, y in tqdm(valid_data_loader):
        # Loop over video frames to get predictions
        # Metrics for semantic labels
        # Switch to cuda
        mask = mask.to(device)
        given_objs = given_objs.to(device)
        given_cats = given_cats.to(device)
        target_obj = target_obj.to(device)
        target_cat = target_cat.to(device)
        x_mask = get_mask(target_obj, edit_type)
        # Get the alternative ground truth as we are editing
        tokens = y[0].split(' ')
        # We will try from position 2 to 4
        tokens = tokens[2:5]
        current_keyword = ''
        tokens = [tokens[0], tokens[0] + ' ' + tokens[1], tokens[0] + ' ' + tokens[1] + ' ' + tokens[2]]
        for i in range(3):
            current_keyword = tokens[i]
            if get_gt_obj(current_keyword, target_obj):
                target_obj, target_cat, reg_p2p = get_gt_obj(current_keyword, target_obj)
                break

        reg_fitness.append(reg_p2p.fitness)
        reg_mse.append(reg_p2p.inlier_rmse)
        reg_set.append(len(reg_p2p.correspondence_set))                

        chamfer_s = 0
        emd_s = 0
        class_acc_list = [[] for _ in range(num_obj_classes)]
        class_acc = dict()
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )
        target_obj_shape = list(target_obj.shape)

        with torch.no_grad():
            sample = sample_fn(
                model,
                target_obj_shape,
                mask,
                given_objs,
                given_cats,
                y=y,
                clip_denoised=clip_denoised,
                model_kwargs=None,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
                # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
            )
        pred = sample.float().to(device)
        target_obj = target_obj.float().to(device)
        if edit_type == "obj_dis":
            new_centroid = pred[0].mean(dim=0)
            old_centroid = target_obj[0].mean(dim=0)
            pred[0] = target_obj[0] - old_centroid + new_centroid
        else:
            pred = x_mask * target_obj + (1-x_mask) * pred

        loss, loss_normals = chamfer_distance(pred, target_obj)
        chamfer_s += loss
        chamfer_list.append(chamfer_s)

        # Calculate EMD loss
        emd_loss = emd(pred.detach().cpu(), target_obj.detach().cpu())
        emd_s += emd_loss
        emd_list.append(emd_s)

        # Calculate F1 score
        f1_score = calculate_fscore(pred.squeeze(0).detach().cpu(), target_obj.squeeze(0).detach().cpu())
        f1_list.append(f1_score[0])

        # Calculate for categorical
        pred_cat = model.saved_cat
        # Beside, we retrieve guiding points as the procedure is similar
        guiding_points = model.saved_guiding_points
        pred_cat = pred_cat.squeeze(1)
        target_cat = torch.argmax(target_cat, dim=1)
        total_topk_acc.append(accuracy(pred_cat, target_cat, topk=(3,))[0].item())
        pred_cat = torch.argmax(pred_cat, dim=1)
        acc = (pred_cat==target_cat).sum().item()
        total_acc.append(acc)
        # visualizer.destroy_window()
        out_file = lookup_tab[y[0]]
        f.write("Chamfer distance for seq {}: {:.4f}".format(out_file, chamfer_s) + '\n')

        # Write predicted points to files
        if not os.path.exists(os.path.join(output_dir, 'predictions')):
            os.makedirs(os.path.join(output_dir, 'predictions'))
        with open(os.path.join(output_dir, 'predictions', out_file + '.npy'), 'wb') as fp:
            pred = pred[0].cpu().numpy()
            np.save(fp, pred)
        
        # Write guiding points to files
        if not os.path.exists(os.path.join(output_dir, 'guiding_points')):
            os.makedirs(os.path.join(output_dir, 'guiding_points'))
        with open(os.path.join(output_dir, 'guiding_points', out_file + '.npy'), 'wb') as fp:
            guiding_points = guiding_points[0].cpu().numpy()
            np.save(fp, guiding_points)

    f.write("Final Chamfer distance: {:.4f}".format(list_mean(chamfer_list)) + '\n')
    f.write("Final EMD: {:.4f}".format(list_mean(emd_list)) + '\n')
    f.write("Final F1 score: {:.4f}".format(list_mean(f1_list)) + '\n')
    f.write("Category accuracy: {:.4f}".format(list_mean(total_acc)) + '\n')
    f.write("Top 3 accuracy: {:.4f}".format(list_mean(total_topk_acc)) + '\n')
    f.write("Fitness: {:.4f}".format(list_mean(reg_fitness)) + '\n')
    f.write("MSE: {:.4f}".format(list_mean(reg_mse)) + '\n')
    f.write("Corr set: {:.4f}".format(list_mean(reg_set)) + '\n')
    f.close()
