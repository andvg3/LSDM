import os
import numpy as np
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import posa.data_utils as du
from posa.dataset import ProxDataset_txt

from util.model_util import create_model_and_diffusion

# Example usage
# python predict_contact.py ../data/amass --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --output_dir ../results/amass

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
    pnt_size = 1024
    # For fix_ori
    fix_ori = True
    ds_weights = torch.tensor(np.load("posa/support_files/downsampled_weights.npy"))
    associated_joints = torch.argmax(ds_weights, dim=1)
    os.makedirs(output_dir, exist_ok=True)

    seq_name_list = sorted(os.listdir(os.path.join(data_dir, 'context')))
    use_ddim = False  # FIXME - hardcoded
    clip_denoised = False  # FIXME - hardcoded
    
    # Setup names for output files
    context_dir = os.path.join(data_dir, 'context')
    files = os.listdir(context_dir)
    up_tab = dict()
    for file in files:look
        reduced_file = file.split('.')[0]

        with open(os.path.join(context_dir, file), 'r') as f:
            prompt = f.readlines()[0].strip()
            lookup_tab[prompt] = reduced_file
    
    # Load in model checkpoints and set up data stream
    valid_dataset = ProxDataset_txt(data_dir, max_frame=max_frame, fix_orientation=fix_ori,
                                   step_multiplier=1, jump_step=jump_step)
    valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    model, diffusion = create_model_and_diffusion()
    model.eval()
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    sample_fn = (
        diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
    )

    idx = 0
    for mask, given_objs, given_cats, target_obj, target_cat, y in tqdm(valid_data_loader):
        target_obj_shape = list(target_obj.shape)
        mask = mask.to(device)
        given_objs = given_objs.to(device)
        given_cats = given_cats.to(device)
        target_obj = target_obj.to(device)
        target_cat = target_cat.to(device)
        out_file = lookup_tab[y[0]]
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
            pr_cf = sample

        pred = pr_cf
        # pred = pred[:(int)(mask.sum())]

        cur_output_path = os.path.join(output_dir, "{}.npy".format(out_file))
        np.save(cur_output_path, pred.cpu().numpy())
        idx += 1
        # if save_probability:
        #     softmax = torch.nn.Softmax(dim=2)
        #     pred_npy = softmax(pred).detach().cpu().numpy()
        # else:
        #     pred_npy = torch.argmax(pred, dim=-1).unsqueeze(-1).detach().cpu().numpy()
        # np.save(cur_output_path, pred_npy)