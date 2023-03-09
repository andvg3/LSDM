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

import posa.vis_utils as vis_utils
import posa.general_utils as general_utils
import posa.data_utils as du

from util.model_util import create_model_and_diffusion

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

def _setup_static_objs(objs_dir, cases_dir):
        _cat = {
            "chair": 0,
            "table": 1,
            "cabinet": 2,
            "sofa": 3,
            "bed": 4,
            "chest_of_drawers": 5,
            "chest": 5,
            "stool": 6,
            "tv_monitor": 7,
            "tv": 7,
            "lighting": 8,
            "shelving": 9,
            "seating": 10,
            "furniture": 11,
        }
        scenes = os.listdir(objs_dir)
        max_objs = 8
        objs = dict()
        cats = dict()
        handle = 2
        pnt_size = 1024
        for scene in scenes:
            objs[scene] = dict()
            cats[scene] = [-1 for _ in range(max_objs)]
            case_path = os.path.join(cases_dir, scene)
            case_fn = os.path.join(case_path, 'case_{}.txt'.format(handle))

            with open(case_fn, 'r') as f:
                given_objs, target_obj = f.readlines()
                given_objs = given_objs.strip('\n').split(' ')
            
            # Read given objects
            given_objs_tensor = torch.zeros(max_objs, pnt_size, 3)
            for idx, given_obj in enumerate(given_objs):
                given_cat = given_obj.split('_')[0]
                cats[scene][idx] = _cat[given_cat]
                given_obj_fn = os.path.join(objs_dir, scene, given_obj + '.npy')
                with open(given_obj_fn, 'rb') as f:
                    given_obj_verts = torch.from_numpy(np.load(f))
                    given_objs_tensor[idx] = given_obj_verts

            # Read target object
            target_cat = target_obj.split('_')[0]
            target_obj_fn = os.path.join(objs_dir, scene, target_obj + '.npy')
            with open(target_obj_fn, 'rb') as f:
                target_obj_verts = torch.from_numpy(np.load(f))
        
            objs[scene] = (given_objs_tensor, target_obj_verts)
            cats[scene] = torch.tensor(cats[scene]), torch.tensor([_cat[target_cat]])
        return objs, cats


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

    # Argument logic check
    if visualize and save_video:
        save_video = False
    if do_train or do_valid:
        save_video = False
        visualize = False

    device = torch.device("cuda")
    num_obj_classes = 8
    # For fix_ori
    ds_weights = torch.tensor(np.load("posa/support_files/downsampled_weights.npy"))
    associated_joints = torch.argmax(ds_weights, dim=1)
    torch.manual_seed(seed)


    seq_name_list = []
    chamfer_list = []
    seq_class_acc = [[] for _ in range(num_obj_classes)]

    vertices_dir = os.path.join(data_dir, "vertices")
    contacts_s_dir = os.path.join(data_dir, "semantics")
    vertices_can_dir = os.path.join(data_dir, "vertices_can")
    pre_data_dir = data_dir.split('/')[0]
    objs_dir = os.path.join(pre_data_dir, "objs")
    cases_dir = os.path.join(pre_data_dir, "cases")
    objs, cats = _setup_static_objs(objs_dir, cases_dir)

    if do_valid or do_train:
        vertices_file_list = os.listdir(vertices_dir)
        seq_name_list = [file_name.split('_verts')[0] for file_name in vertices_file_list]
    else:
        seq_name_list = [single_seq_name]

    # Load in model checkpoints and set up data stream
    model, diffusion = create_model_and_diffusion()
    model.eval()
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    for seq_name in seq_name_list:
        if save_video or visualize:
            save_seq_dir = os.path.join(output_dir, seq_name)
            os.makedirs(save_seq_dir, exist_ok=True)
            save_seq_model_dir = os.path.join(save_seq_dir, model_name)
            os.makedirs(save_seq_model_dir, exist_ok=True)
            output_image_dir = os.path.join(save_seq_model_dir, cam_path.split("/")[-1])
            os.makedirs(output_image_dir, exist_ok=True)
        print("Test scene: {}".format(seq_name))

        scene = seq_name.split('_')[0]
        given_objs, target_obj = objs[scene]
        given_cats, target_cat = cats[scene]
        given_objs =  given_objs.unsqueeze(0).to(device)
        target_obj = target_obj.unsqueeze(0).to(device)
        given_cats = given_cats.unsqueeze(0).to(device)
        target_cat = target_cat.unsqueeze(0).to(device)
        verts = torch.tensor(np.load(os.path.join(vertices_dir, seq_name + "_verts.npy"))).to(device)
        verts_can = torch.tensor(np.load(os.path.join(vertices_can_dir, seq_name + "_verts_can.npy"))).to(device)
        contacts_s = torch.tensor(np.load(os.path.join(contacts_s_dir, seq_name + "_cfs.npy"))).to(device)

        if save_video:
            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()

        if visualize or save_video:
            scene_name = seq_name.split("_")[0]
            test_scene_mesh_path = "{}/{}.ply".format(scene_dir, scene_name)
            scene = o3d.io.read_triangle_mesh(test_scene_mesh_path)
            down_sample_level = 2
            # tpose for getting face_arr
            tpose_mesh_path = os.path.join(tpose_mesh_dir, "mesh_{}.obj".format(down_sample_level))
            faces_arr = trimesh.load(tpose_mesh_path, process=False).faces

        # Loop over video frames to get predictions
        # Metrics for semantic labels
        chamfer_s = 0
        class_acc_list = [[] for _ in range(num_obj_classes)]
        class_acc = dict()
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        verts_can_batch = verts_can[::jump_step]
        if fix_ori:
            verts_can_batch = du.normalize_orientation(verts_can_batch, associated_joints, device)
        if verts_can_batch.shape[0] > max_frame:
            verts_can_batch = verts_can_batch[:max_frame]

        mask = torch.zeros(1, max_frame, device=device)
        mask[0, :verts_can_batch.shape[0]] = 1
        verts_can_padding = torch.zeros(max_frame - verts_can_batch.shape[0], *verts_can_batch.shape[1:], device=device)
        verts_can_batch = torch.cat((verts_can_batch, verts_can_padding), dim=0)
        verts_can_batch = verts_can_batch.unsqueeze(0)
        target_obj_shape = list(target_obj.shape)

        with torch.no_grad():
            sample = sample_fn(
                model,
                target_obj_shape,
                verts_can_batch,
                torch.ones(verts_can_batch.shape[0], verts_can_batch.shape[1]),
                given_objs,
                given_cats,
                y=["" for _ in range(verts_can_batch.shape[0])],
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
        loss, loss_normals = chamfer_distance(pred, target_obj.float().to(device))
        chamfer_s += loss
        chamfer_list.append(chamfer_s)
        # visualizer.destroy_window()
        with open(os.path.join(output_dir, "results.txt"), "a+") as f:
            f.write("Chamfer distance for seq {}: {:.4f}".format(seq_name, chamfer_s) + '\n')

    with open(os.path.join(output_dir, "results.txt"), "a+") as f:
        f.write("Final Chamfer distance: {:.4f}".format(list_mean(chamfer_list)) + '\n')
