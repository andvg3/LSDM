# Language-driven Scene Synthesis using Multi-conditional Diffusion Model
This is the official implementation of the NeurIPS 2023 paper: Language-driven Scene Synthesis using Multi-conditional Diffusion Model.

https://github.com/andvg3/scene-synthesis/assets/140178004/4eee3406-087f-41a1-b729-ce2d461ab4b0

## Table of contents
   1. [Installation](#installation)
   1. [Training and Testing](#training-and-testing)
   1. [Scene Synthesis](#scene-synthesis)

## Installation
### Environment
We highly recommand you to create a Conda environment to better manage all the Python packages needed.
```
conda create -n lsdm python=3.8
conda activate lsdm
```
After you create the environment, please install pytorch with CUDA. You can do it by running
```
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
```
The other dependencies needed for this work is listed in the requirements.txt. 
We recommend the following commands to install dependencies: 
```
pip install git+https://github.com/openai/CLIP.git
pip install transformers
conda install pytorch3d -c pytorch3d
pip install -r requirements.txt
```

### Datasets and Model Checkpoints
PRO-teXt is an extension of [PROXD](https://prox.is.tue.mpg.de/). Please visit their website to obtain the PROXD dataset first. We provide the extension of PRO-teXt as in the [link](https://forms.gle/gYFUrD6wrDJRbLPD7). You also need to obtain HUMANISE via their [project page](https://silvester.wang/HUMANISE/). The dataset hierarchy should follow this template:
```
|- data/
    |- protext
         |- mesh_ds
         |- objs
         |- proxd_test
         |- proxd_test_edit
         |- proxd_train
         |- proxd_valid
         |- scenes
    |- supp
         |- proxd_train
         |- proxd_valid
```

All model checkpoints that are used to benchmark in the paper are available at this [link](https://drive.google.com/file/d/1T1CqAG2UxdtugqxrPj1trRe_UajX1OJl/view?usp=sharing).

## Training and Testing
To train a baseline, use the following command:
```
python -m run.train_<baseline> --train_data_dir data/protext/proxd_train --valid_data_dir data/protext/proxd_valid --fix_ori --epochs 1000 --out_dir training --experiment <baseline>
```
For example, if you want to train LSDM, use the following command:

```
python -m run.train_sdm --train_data_dir data/protext/proxd_train --valid_data_dir data/protext/proxd_valid --fix_ori --epochs 1000 --out_dir training --experiment sdm
```
To test a baseline, use the following command:
```
python -m run.test_<baseline> data/protext/proxd_test/ --load_model training/<baseline>/model_ckpt/best_model_cfd.pt --model_name <baseline> --fix_ori --test_on_valid_set --output_dir training/<baseline>/output
```
For example, you can use:
```
python -m run.test_sdm data/protext/proxd_test/ --load_model training/sdm/model_ckpt/best_model_cfd.pt --model_name sdm --fix_ori --test_on_valid_set --output_dir training/sdm/output
```
to test an LSDM checkpoint. Note that, you can also train on HUMANISE dataset. Just replace the path of `data/protext/proxd_train` by `data/humanise/train`.

## Scene Synthesis
To generate a video sequence as in our paper, you can proceed by using the following steps:

1. Step 1: Generate objects from contact points
```
python fit_custom_obj.py --sequence_name <sequence_name> --vertices_path data/supp/proxd_valid/vertices/<sequence_name>_verts.npy --contact_labels_path data/supp/proxd_valid/semantics/<sequence_name>_cfs.npy --output_dir fitting_results/<baseline> --label 3 --file_name training/sdm/output/predictions/<interaction_name>.npy
```
where `sequence_name` is the name of the *human motion* and `interaction_name` is the name of the *human_pose*. Note that, we name human pose very closely to its corresponding human motion. For example, you can use the following command:

2. Step 2: Visualization
```
python vis_fitting_results.py --fitting_results_path fitting_results/<baseline>/<sequence_name>/ --vertices_path data/supp/proxd_valid/vertices/<sequence_name>_verts.npy
```
For example,
```
python vis_fitting_results.py --fitting_results_path fitting_results/sdm/N0Sofa_00034_02/ --vertices_path data/supp/proxd_valid/vertices/N0Sofa_00034_02_verts.npy
```
The script ran above will save rendered frames in `fitting_results/N0Sofa_00034_02/rendering`. 
**Note that you need a screen to run this command.** In case you are testing the project on a server which doesn't have a display service, you can still load the saved objects and human meshes and use other approaches to visualize them. To get the human meshes, you can still run the above command and wait until the program automatically exits. The script will save the human meshes of your specified motion sequence in `fitting_results/<sequence name>/human/mesh`.

Best fitting objects are stored in `fitting_results/<sequence name>/fit_best_obj/<object category>/<object index>/<best_obj_id>/opt_best.obj`. As mentioned before, you can get `<best_obj_id>` in `fitting_results/<sequence name>/fit_best_obj/<object category>/<object index>/best_obj_id.json`.

## Citation
Part of our codebase is based on [Ye et al.](https://github.com/onestarYX/summon). If you find this work helpful, please consider citing:
```
TBD
```
