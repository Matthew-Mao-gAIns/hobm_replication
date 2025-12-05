# Hand-object-based Mask Obfuscation
This is the code repository for the paper [_Impacts of Image Obfuscation on Fine-grained Activity Recognition in Egocentric Video_](https://ieeexplore.ieee.org/abstract/document/9767447)
## Installation
```sh
git clone git@github.com:HAbitsLab/HOBM.git
cd HOBM
git submodule init
git submodule update
```
This repository uses SlowFast library by Meta but don't get confused. The library contains I3D implementation which we used in this paper. Go ahead and follow the [guides](https://github.com/facebookresearch/SlowFast) to install SlowFast on your machine.

## Dataset Preparation
#### Download and generate obfuscated frames
Download the dataset (make sure you set the right path for $DESTDIR in the bash file):
```sh
cd dataset_scripts/dataset_prep
. build.sh
```
This downloads the EGTEA Gaze+ dataset and extracts the frames from video clips. We provided the hand+object masks for this dataset in numpy files. It generates the obfuscated jpg images using the numpy files. Depending on your internet connection speed, the download process might take some time. I'm planning to speed up the other steps by doing things in parallel. 
#### Interpolation
For those frames where the hand is not visible, the obfuscated image is basically a black frame. To account for this, we replaced these frames by interpolation. Refer to `interpolate.py` for more information.

#### Splits
We removed some of the classes that didn't deal with objects in hand (e.g., fridge) and merged similar ones. Use the splits that we provided located at 'dataset_prep/splits'

## Training and Testing
#### Download pre-trained weights:
We used pre-trained weights of I3D trained on kinetics400 to start training. The weights are provided by Meta and can be downloaded:
```sh
cd experiments/I3D/weights
. download.sh
```
We provided config files that can be used to train and test the model both for raw and obfuscated images. Change the config file paths accordingly and run:
#### Train
```sh
python tools/run_net.py --cfg ../experiments/I3D/configs/train/R50_raw_32x4.yaml NUM_GPUS 1
python tools/run_net.py --cfg ../experiments/I3D/configs/train/R50_hands_obj_32x4.yaml NUM_GPUS 1
```
#### Test
```sh
python tools/run_net.py --cfg ../experiments/I3D/configs/test/R50_raw_32x4.yaml NUM_GPUS 1
python tools/run_net.py --cfg ../experiments/I3D/configs/test/R50_hands_obj_32x4.yaml NUM_GPUS 1
```

You can ask questions by openning up an issue. Thank you!


## Citation
```
@INPROCEEDINGS{9767447,
  author={Shahi, Soroush and Alharbi, Rawan and Gao, Yang and Sen, Sougata and Katsaggelos, Aggelos K and Hester, Josiah and Alshurafa, Nabil},
  booktitle={2022 IEEE International Conference on Pervasive Computing and Communications Workshops and other Affiliated Events (PerCom Workshops)}, 
  title={Impacts of Image Obfuscation on Fine-grained Activity Recognition in Egocentric Video}, 
  year={2022},
  volume={},
  number={},
  pages={341-346},
  doi={10.1109/PerComWorkshops53856.2022.9767447}}
```


 
# hobm_replication
