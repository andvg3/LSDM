# Language-driven Scene Synthesis using Multi-conditional Diffusion Model
This is the official implementation of the NeurIPS 2023 paper: Language-driven Scene Synthesis using Multi-conditional Diffusion Model.

[Overview.](https://owncloud.tuwien.ac.at/index.php/s/Ul9vM69MWdLKNyJ/download)

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
We recommend you to use pip to install them: 
```
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install transformers
conda install pytorch3d -c pytorch3d
```

### Datasets and Model Checkpoints
Our extension PRO-teXt and model checkpoints will be updated soon.

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
To generate contact label predictions for all motion sequences stored in the 
`.npy` format in a folder (e.g. `amass/` in our provided data folder),
you can run
```
cd contactFormer
python predict_contact.py ../data/amass --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --output_dir PATH_OF_OUTPUT
```
Please replace the `PATH_OF_OUTPUT` to any path you want. If you want to
generate predictions for the PRO-teXt dataset, you can try
```
python predict_contact.py ../data/proxd_valid/vertices_can --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --output_dir PATH_OF_OUTPUT
```
The above example command generate predictions for the validation split of PRO-teXt.
If you want save the probability for each contact object category in order
to generate more diverse scenes, you can add a `--save_probability` flag
in addition to the above command.

### Train and test ContactFormer
You can train and test your own model. To train the model, still under `contactformer/`, you can run
```
python train_contactformer.py --train_data_dir ../data/proxd_train --valid_data_dir ../data/proxd_valid --fix_ori --epochs 1000 --out_dir ../training --experiment default_exp
```
Replace the train and validation dataset paths after `--train_data_dir` and `--valid_data_dir`
with the path of your downloaded data. `--fix_ori` normalizes the orientation of
all motion sequences in the dataset: for each motion sequence, rotate all poses in that sequence
so that the first pose faces towards some canonical orientation (e.g. pointing out of the screen)
and the motion sequence continues with the rotated first pose. `--experiment` specifies the name
of the current experiment. Running the above command, all model checkpoints and 
training logs will be saved at `<path_to_project_root>/training/default_exp`.

To test a model checkpoint, you can run
```
python test_contactformer.py ../data/proxd_valid/ --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --model_name contactformer --fix_ori --test_on_valid_set --output_dir PATH_OF_OUTPUT
```
The above command tests ContactFormer on the validation split of PRO-teXt dataset.
The first argument is location of the validation set folder. 
`--model_name` is an arbitrary name you can set for disguishing the model you are testing. 
It can also help you pinpoint the location of the test result
since the result will be saved in a text file at the location `PATH_OF_OUTPUT/validation_results_<model_name>.txt`.

You can also run add a `--save_video` flag to save the visualization of contact label prediction
for some specific motion sequence. For example, you can run
```
python test_contactformer.py ../data/proxd_valid/ --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --model_name contactformer --fix_ori --single_seq_name MPH112_00151_01 --save_video --output_dir PATH_OF_OUTPUT
```
to save the visualization for predicted contact labels along with rendered body and scene meshes
for each frame in the MPH112_00151_01 motion sequence. The rendered frames will be saved at
`PATH_OF_OUTPUT/MPH112_00151_01/`. **Note that you need a screen to run this command.**

There are other parameters you can set to change the training scheme or the model architecture. Check
`train_contactformer.py` and `test_contactformer.py` for more details.


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

### Visualization
If you want to visualize the fitting result (i.e. recovered objects along with the human motion),
using the same example as mentioned above, you can run
```
python vis_fitting_results.py --fitting_results_path fitting_results/MPH11_00150_01 --vertices_path data/proxd_valid/vertices/MPH11_00150_01_verts.npy
```
The script will save rendered frames in `fitting_results/MPH11_00150_01/rendering`. 
**Note that you need a screen to run this command.** In case you are testing the project on a server which doesn't have a display service, you can still load the saved objects and human meshes and use other approaches to visualize them. To get the human meshes, you can still run the above command and wait until the program automatically exits. The script will save the human meshes of your specified motion sequence in `fitting_results/<sequence name>/human/mesh`.

Best fitting objects are stored in `fitting_results/<sequence name>/fit_best_obj/<object category>/<object index>/<best_obj_id>/opt_best.obj`.
As mentioned before, you can get `<best_obj_id>` in `fitting_results/<sequence name>/fit_best_obj/<object category>/<object index>/best_obj_id.json`.

## Citation
Part of our codebase is based on [Ye et al.](https://github.com/onestarYX/summon). If you find this work helpful, please consider citing:
```
TBD
```
