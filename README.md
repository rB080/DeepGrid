# DeepGrid
This is the official implementation of our paper titled "DeepGrid: A Surprisingly Deep Grid-shaped Network for Medical Image Segmentation".

![Alt text](DeepGrid.drawio.png?raw=true "Model Architecture")

## Dependencies:
To install required dependencies, use the following command, preferably in a new virtual environment:
```
pip install -r requirements.txt
```

## Initial setup:
- Add the ISIC 2017 dataset root directory to the main directory and rename it to "ISIC_2017".
- Add other dataset root directories to the main directory.
- Add trained models (if any) and logfiles to "logmods"

## Running a basic training:
Use this to check syntax for running "train.py":
```
python train.py
```
Error message will give a list of arguments to be passed. Complete syntax will be added to the markdown shortly.

If you wish to use default training settings to train a basic model, simply run the shell file as:
```
bash run.sh
```
Change paths in run.sh to suit the requirements.

Repository will be further upgraded shortly for better accessibility. Model evaluation codes will be coming soon.
