import os
import numpy as np
import argparse
import torch
from tqdm import tqdm

import posa.data_utils as du

from util.model_util import create_model_and_diffusion

# Example usage
# python predict_contact.py ../data/amass --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --output_dir ../results/amass
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
    parser.add_argument("--load_model", type=str, default="../training/model_ckpt/epoch_0045.pt",
                        help="checkpoint path to load")
    parser.add_argument("--encoder_mode", type=int, default=1,
                        help="different number represents different variants of encoder")
    parser.add_argument("--decoder_mode", type=int, default=1,
                        help="different number represents different variants of decoder")
    parser.add_argument("--n_layer", type=int, default=3, help="Number of layers in transformer")
    parser.add_argument("--n_head", type=int, default=4, help="Number of heads in transformer")
    parser.add_argument("--jump_step", type=int, default=8, help="Frame skip size for each input motion sequence")
    parser.add_argument("--dim_ff", type=int, default=512,
                        help="Dimension of hidden layers in positionwise MLP in the transformer")
    parser.add_argument("--f_vert", type=int, default=64, help="Dimension of the embeddings for body vertices")
    parser.add_argument("--max_frame", type=int, default=256,
                        help="The maximum length of motion sequence (after frame skipping) which model accepts.")
    parser.add_argument("--posa_path", type=str, default="../training/posa/model_ckpt/epoch_0349.pt",
                        help="The POSA model checkpoint that ContactFormer can pre-load")
    parser.add_argument("--output_dir", type=str, default="../results/output")
    parser.add_argument("--save_probability", dest='save_probability', action='store_const', const=True, default=False,
                        help="Save the probability of each contact labels, instead of the most possible contact label")

    # Parse arguments and assign directories
    args = parser.parse_args()
    args_dict = vars(args)

    data_dir = args_dict['data_dir']
    ckpt_path = args_dict['load_model']
    encoder_mode = args_dict['encoder_mode']
    decoder_mode = args_dict['decoder_mode']
    n_layer = args_dict['n_layer']
    n_head = args_dict['n_head']
    jump_step = args_dict['jump_step']
    max_frame = args_dict['max_frame']
    dim_ff = args_dict['dim_ff']
    f_vert = args_dict['f_vert']
    posa_path = args_dict['posa_path']
    output_dir = args_dict['output_dir']
    save_probability = args_dict['save_probability']

    device = torch.device("cuda")
    num_obj_classes = 8
    # For fix_ori
    fix_ori = True
    ds_weights = torch.tensor(np.load("posa/support_files/downsampled_weights.npy"))
    associated_joints = torch.argmax(ds_weights, dim=1)
    os.makedirs(output_dir, exist_ok=True)

    seq_name_list = []
    vertices_file_list = os.listdir(data_dir)
    seq_name_list = [file_name.split('_verts_can')[0] for file_name in vertices_file_list]
    list_set = set(seq_name_list)
    seq_name_list = list(list_set)
    use_ddim = False  # FIXME - hardcoded
    clip_denoised = False  # FIXME - hardcoded

    # Establish objects dir
    pre_data_dir = data_dir.split('/')[0]
    objs_dir = os.path.join(pre_data_dir, "objs")
    cases_dir = os.path.join(pre_data_dir, "cases")
    objs, cats = _setup_static_objs(objs_dir, cases_dir)

    # Load in model checkpoints and set up data stream
    model, diffusion = create_model_and_diffusion()
    model.eval()
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    sample_fn = (
        diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
    )
    for seq_name in seq_name_list:
        print("Test scene: {}".format(seq_name))

        verts_can = torch.tensor(np.load(os.path.join(data_dir, seq_name + "_verts_can.npy"))).to(device).to(torch.float32)

        # Loop over video frames to get predictions
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
        scene = seq_name.split('_')[0]
        given_objs, target_obj = objs[scene]
        given_cats, target_cat = cats[scene]
        given_objs =  given_objs.unsqueeze(0).to(device)
        target_obj = target_obj.unsqueeze(0).to(device)
        given_cats = given_cats.unsqueeze(0).to(device)
        target_cat = target_cat.unsqueeze(0).to(device)
        cf_shape = [verts_can_batch.shape[0], verts_can_batch.shape[1], verts_can_batch.shape[2], num_obj_classes]
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
            pr_cf = sample

        pred = pr_cf
        # pred = pred[:(int)(mask.sum())]

        cur_output_path = os.path.join(output_dir, seq_name + ".npy")
        np.save(cur_output_path, pred.cpu().numpy())
        # if save_probability:
        #     softmax = torch.nn.Softmax(dim=2)
        #     pred_npy = softmax(pred).detach().cpu().numpy()
        # else:
        #     pred_npy = torch.argmax(pred, dim=-1).unsqueeze(-1).detach().cpu().numpy()
        # np.save(cur_output_path, pred_npy)