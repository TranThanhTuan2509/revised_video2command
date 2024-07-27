## [Translating Videos to Commands for Robotic Manipulation with Deep Recurrent Neural Networks](https://arxiv.org/pdf/1710.00290.pdf)
- Authors: A Nguyen et al. - ICRA 2018 - Revised Time 2024 By Thanh Tuan.


### Requirements
- The training code uses TensorFlow version 1.x. Before training, you need to create a new Anaconda environment if your Python version is >= 3.8:

      conda create -n c2v python=3.7
- Activate the new environment using:

      conda activate c2v
- Install all required libraries with:

      pip install -r requirements.txt

### Training
- Clone the repository to your `$VC_Folder`
- We train the network using [IIT-V2C dataset](https://sites.google.com/site/iitv2c/)
- In case, you don't see `train.txt`, `test.txt` in `train_test_split` folder. Extract the file you downloaded to `$VC_Folder`
- You can extract the features for each frames in the input videos using any network (e.g., VGG, ResNet, etc.)
- For a quick start, the pre-extracted features with ResNet50 is available [here](https://drive.google.com/file/d/1Y_YKHB4Bw6MPXj05S36d1G_3rMx73Uv5/view?usp=sharing). (RECOMMEND OPTION)
- Run this file if you don't see `subtitle` folder:
- Changing the path to `./dataset/breakfast` and running `fid_extraction` file
- Extract the file you downloaded to `$VC_Folder`
- Start training: `python train.py` in `$VC_Folder/main`
- Here is training structure

      The_visual_features_of_ResNet50
      fid files (image captions - labels)
        │
        ├── Train
        │ ├── train_resnet50.pkl
        │ │   ├── image1.npy
        │ │   ├── image2.npy
        │ │   └── ...
        │ ├── train_labels
        │ │   ├── image1.fid
        │ │   ├── image2.fid
        │ │   └── ...
        │  
        ├── Validation
          ├── val_resnet50.pkl
          │   ├── image1.npy
          │   ├── image2.npy
          │   └── ...
          ├── val_labels
          │   ├── image1.fid
          │   ├── image2.fid
          │   └── ...


### Predict & evaluate
- Predict: `python predict.py` in `$VC_Folder/main` folder
- Prepare the results for evaluation: `python prepare_evaluation_format.py` in `$VC_Folder/evaluation` folder
- Evaluate: `python cocoeval.py` in `$VC_Folder/evaluation` folder

		
		@inproceedings{nguyen2018translating,
		  title={Translating videos to commands for robotic manipulation with deep recurrent neural networks},
		  author={Nguyen, Anh and Kanoulas, Dimitrios and Muratore, Luca and Caldwell, Darwin G and Tsagarakis, Nikos G},
		  booktitle={2018 IEEE International Conference on Robotics and Automation (ICRA)},
		  year={2018},
		  organization={IEEE}
		}


### Contact
If you have any questions or comments, please send an to `22023506@vnu.edu.vn`


Original author contact is `anh.nguyen@iit.it`
