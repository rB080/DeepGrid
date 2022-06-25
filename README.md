# DeepGrid
This is the official implementation of our paper titled "DeepGrid: A Surprisingly Deep Grid-shaped Network for Medical Image Segmentation".

![Alt text](DeepGrid.drawio.pdf?raw=true "Model Architecture")

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
and wait for the error. Complete syntax will be added to the markdown shortly.

If you wish to use default training settings to train a basic model, simply run the shell file as:
```
bash run.sh
```

## Model Zoo
Various deepgrid model weights are available for download

**ISIC2017**
|Model|Parameter Count|mIoU|weights|
|----------|----------|----------|----------|
|DeepGrid 3x2|19.59M|0.8413|link|
|DeepGrid 3x3|27.79M|0.8523|link|
|DeepGrid 3x4|35.99M|0.8498|link|
|DeepGrid 4x2|52.41M|0.8567|link|
|DeepGrid 4x3|73.40M|0.8563|link|
|DeepGrid 4x4|94.38M|0.8497|link|
|DeepGrid 5x2|141.71M|0.8595|link|
|DeepGrid 5x3|197.06M|0.8536|link|
|DeepGrid 5x4|252.39M|0.8534|link|

Repository will be further upgraded shortly for better accessibility.
