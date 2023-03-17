from model.sdm import SceneDiffusionModel

from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

from util.fixseed import fixseed
from run.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion():
    # model = SceneDiffusionModel(**get_model_args(args, data))
    model = SceneDiffusionModel(**get_default_model())
    diffusion = create_gaussian_diffusion(get_default_diffusion())
    return model, diffusion


def get_default_model():
    return {
        'seq_len': 256, 
        'modality': 'text',
        'clip_version': 'ViT-B/32', 
        'clip_dim': 512, 
        'dropout': 0.1, 
        'n_layer': 6, 
        'n_head': 8, 
        'f_vert': 64, 
        'dim_ff': 512,
        'd_hid': 256, 
        'mesh_ds_dir': "data/mesh_ds", 
        'posa_path': None, 
        'latent_dim': 64,
        'pcd_dim': 3,
        'cond_mask_prob': 1.0, 
        'device': 0, 
        'vert_dims': 655, 
        'obj_cat': 8, 
        'data_rep': 'rot6d', 
        'njoints': 251,
    }

def get_default_diffusion():
    args = {
        "lambda_fc": 0.0,
        "lambda_rcxyz": 0.0,
        "lambda_vel": 0.0,
        "noise_schedule": "cosine",
        "sigma_small": True,
    }
    return args


def get_model_args():
    return {
        "arch": "trans_enc",
        "batch_size": 64,
        "cond_mask_prob": 0.1,
        "cuda": True,
        "data_dir": "",
        "dataset": "humanml",
        "device": 0,
        "diffusion_steps": 1000,
        "emb_trans_dec": False,
        "eval_batch_size": 32,
        "eval_during_training": False,
        "eval_num_samples": 1000,
        "eval_rep_times": 3,
        "eval_split": "test",
        "lambda_fc": 0.0,
        "lambda_rcxyz": 0.0,
        "lambda_vel": 0.0,
        "latent_dim": 512,
        "layers": 8,
        "log_interval": 1000,
        "lr": 0.0001,
        "lr_anneal_steps": 0,
        "noise_schedule": "cosine",
        "num_frames": 60,
        "num_steps": 600000,
        "overwrite": False,
        "resume_checkpoint": "",
        "save_dir": "save/my_humanml_trans_enc_512",
        "save_interval": 50000,
        "seed": 10,
        "sigma_small": True,
        "train_platform_type": "NoPlatform",
        "unconstrained": False,
        "weight_decay": 0.0
    }

def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args['noise_schedule'], steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args['sigma_small']
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args['lambda_vel'],
        lambda_rcxyz=args['lambda_rcxyz'],
        lambda_fc=args['lambda_fc'],
    )

def get_training_platform():
    args = {
        'seed': 10,
        'train_platform_type': "NoPlatform",
        'save_dir': "debug/"
    }

    fixseed(args['seed'])
    train_platform_type = eval(args['train_platform_type'])
    train_platform = train_platform_type(args['save_dir'])
