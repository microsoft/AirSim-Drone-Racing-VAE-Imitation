# Learning Controls Using Cross-Modal Representations: Bridging Simulation and Reality for Drone Racing

![Teaser](figs/main_2_low.png)

This repository provides a code base to evaluate and train models from the paper "*Learning Controls Using Cross-Modal Representations: Bridging Simulation and Reality for Drone Racing*". The original paper and video can be found in the following links:

Paper: [https://arxiv.org/abs/1909.06993](https://arxiv.org/abs/1909.06993)

Video: [https://youtu.be/aSugOL0qI44](https://youtu.be/aSugOL0qI44)

## License and Citation
This project is licensed under the terms of the MIT license. By using the software, you are agreeing to the terms of the [license agreement](LICENSE).

If you use this code in your research, please cite us as follows:

```
@article{bonatti2019learning,
  title={Learning Controls Using Cross-Modal Representations: Bridging Simulation and Reality for Drone Racing},
  author={Bonatti, Rogerio and Madaan, Ratnesh and Vineet, Vibhav and Scherer, Sebastian and Kapoor, Ashish},
  journal={arXiv preprint arXiv:1909.06993},
  year={2019}
}
```

## Recommended system
Recommended system (tested):
- Ubuntu 18.04
- Python 2.7.15

Python packages used by the example provided and their recommended version:
- tensorflow==2.0.0-beta1
- msgpack-rpc-python==0.4.1
- numpy==1.16.4
- matplotlib==2.1.1
- scikit-learn==0.20.4
- scipy==1.2.2


## Downloading the drone racing files
In order for you to train the models and run Airsim you first need to download all image datasets, behavior cloning datasets, network weights and Airsim binaries:  
- Download all files and datasets [Drone Racing files v. 1.0](https://drive.google.com/drive/folders/19tFUG8bCg3_d_PeQMDHJQvj-ZBv8Ogs_?usp=sharing)
- Extract all individual files in the folders
- Place the 'settings.json' file inside '~/Documents/AirSim' in your computer

## Training and testing the cross-modal VAE representation
In order to train the cross-modal representations you can either use the downloaded image dataset from the previous step, or generate the data yourself using Airsim.

### Training with downloaded dataset

- Go to folder 'cmvae', and inside file 'train_cmvae.py' edit variable 'data_dir' to the correct path of the extracted dataset within your computer. The default value is the directory with 1K images. But for final training you will need more images, such as the 50K or 300K datasets
- Also, edit variable 'output_dir' to the correct place where you want the models to be saved 
- Run

```
train_cmvae.py
```

- Network weights will be saved every 5 epochs by default, and you can check loss values with tensorboard or by looking at the terminal
- Once the network is trained you can evaluate it using another script, which will automatically plot histograms of errors, image reconstructions and latent space interpolations:
```
eval_cmvae.py
```

### Generating your own dataset with Airsim
You may want to generate a custom dataset for training you cross-modal VAE. Here are the steps to do it:

- Start the Airsim environment from the binary file:
```
$ cd /yourpath/all_files/airsim_binaries/vae_env
$ ./AirSimExe.sh -windowed
```
- If it asks if you want the car model, click 'No'
- Inside the file 'datagen/img_generator/main.py' first change the desired number of samples and saved dataset path
- Run the script for generating data:
```
main.py     # inside datagen/img_generator
```
- Once the dataset is generated, follow the previous scripts for training the CM-VAE


## Generating imitation learning data for racing
In order to train the behavior cloning networks you can either use the downloaded image-action pairs dataset or generate the data yourself using Airsim.

### Training with downloaded dataset

- Go to folder 'imitation_learning', and inside file 'train_bc.py' edit variables 'base_path', 'data_dir_list', and 'output_dir'. By default you will be using downloaded datasets with  0m to 3m of random gate displacement amplitude over a course with 8m of nominal radius
- Edit the variables relative to the training mode (full end-to-end, latent representation or regression as latent representation) and weights path for the latent representations (not applicable for full end-to-end learning)
- Run the script for training the behavior cloning policies:
```
train_bc.py
```

### Generating your own imitation learning dataset with Airsim
You may want to generate a custom dataset for training you behavior cloning policies. Here are the steps to do it:

- Start the Airsim environment from the binary file:
```
$ cd /yourpath/all_files/airsim_binaries/recording_env
$ ./AirSimExe.sh -windowed
```
- If it asks if you want the car model, click 'No'
- Inside the file 'datagen/action_generator/src/soccer_datagen.py' first change the desired number of samples and saved dataset path
- Run the script for generating data:
```
main.py     # inside datagen/img_generator
```
- Once the dataset is generated, follow the previous scripts for training the CM-VAE
