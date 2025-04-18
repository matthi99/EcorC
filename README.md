# Error correcting 2D-3D cascaded network for myocardial infarct segmentation



This repository is created for the publication:

```
Schwab, M., Pamminger, M., Kremser, C., Obmann, D., Haltmeier, M., Mayr, A. (2023). Error correcting 2D-3D cascaded network for myocardial infarct scar segmentation on late gadolinium enhancement cardiac magnetic resonance images. arXiv preprint arXiv:2306.14725, 2023.
```


# Instalation

1. Clone the git repository. 
   ```bash
   git clone https://github.com/matthi99/EcorC.git
   ``` 

2. Install and activate the virtual environment:
   - Windows
     ```bash
     cd EcorC
     conda env create -f env_lge.yml
     conda activate env_lge
     ```
   - Linux
     ```bash
     cd EcorC
     conda env create -f env_lge_linux.yml
     conda activate env_lge
     ```
   
3. Install [PyTorch](https://pytorch.org/get-started/locally/)
   ```bash
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
   ```

# Usage

## Preprocessing
1. Download the [EMIDEC Dataset](https://emidec.com/dataset#download) and/or the [MyoPS Dataset](https://mega.nz/folder/BRdnDISQ#FnCg9ykPlTWYe5hrRZxi-w). If you downloaded both datasets the folder structure in your `DATA_FOLDER` should look like this 
   ```bash
   DATA_FOLDER/
   ├── emidec-dataset-1.0.1 
   ├── emidec-segmentation-testset-1.0.0
   ├── MyoPS 2020 Dataset
   ```
   Note that for training the emidec-segmentation-testset folder is optional and does not have to be present.

2. Prepare the downloaded dataset for training. For this run the following command in your console
   ```bash
   python preprocessing.py DATASET_NAME PATH_TO_DATA_FOLDER
   ``` 
   - `DATASET_NAME` specifies which datset should be preprocessed. Possible arguments are `EMIDEC` or `MyoPS`. 

## Training

### 2D U-Net

To train the two dimensional U-Nets run the command
```bash
python 2d-net.py DATASET_NAME FOLD 
``` 
- `FOLD` specifies on which on which od the five folds (0,1,2,3,4) the network should be trained.  
- Trained networks and training progress will get saved in a folder called `RESULTS_FOLDER` located in the `DATA_FOLDER` directory. 

### 2D-3D cascade

To train the Error correcting 2D-3D cascaded framework run the command
```bash
python 3d-cascade.py DATASET_NAME FOLD 
``` 
Note that to be able to train the cascade the 2D U-Net had to be trained beforehand. 

## Testing

To test the final framework on the testsets of the correspond of the EMICED callenge run 
```bash
python inference.py DATASET_NAME
```
- Note that inference will be done with all 5 folds from the cross-validation as an ensemble. Thus, 2D and 3D cascade must have been trained on all 5 folds prior to running inference.   
- Predictions and plots of the results will be saved in `RESULTS_FOLDER/DATASET_NAME/inference`.


## Authors and acknowledgment
Matthias Schwab<sup>1</sup>, Mathias Pamminger<sup>1</sup>, Christian Kremser <sup>1</sup>, Daniel Obmann <sup>2</sup>, Markus Haltmeier <sup>2</sup>, Agnes Mayr <sup>1</sup>

<sup>1</sup> University Hospital for Radiology, Medical University Innsbruck, Anichstraße 35, 6020 Innsbruck, Austria 

<sup>2</sup> Department of Mathematics, University of Innsbruck, Technikerstrasse 13, 6020 Innsbruck, Austria



