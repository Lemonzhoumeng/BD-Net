# BD-Net Semi/Weakly-Supervised Medical Image Segmentation
Pytorch implementation of our Boundary-Enhanced and Density-Guided Contrastive Learning for Semi/Weakly-Supervised Medical Image Segmentation (BIBM 2024). <br/>


# Dataset
* The ACDC dataset with mask annotations can be downloaded from: [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).
* The Scribble annotations of ACDC can be downloaded from: [Scribble](https://gvalvano.github.io/wss-multiscale-adversarial-attention-gates/data).
* The data processing code [Link](https://github.com/Luoxd1996/WSL4MIS/blob/main/code/dataloaders/acdc_data_processing.py)  the pre-processed ACDC data [Link](https://github.com/HiLab-git/WSL4MIS/tree/main/data/ACDC).

# Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* TensorBoardX
* Python >= 3.6 
* Efficientnet-Pytorch `pip install efficientnet_pytorch`
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

# Usage

1. Clone this project.
```
git clone https://github.com/Lemonzhoumeng/BD-Net
cd BD-Net
```
2. Data pre-processing or directly [download](https://github.com/HiLab-git/WSL4MIS/tree/main/data/ACDC) the pre-processed data.
```
cd code
python dataloaders/acdc_data_processing.py
```

3.  Train the model
```
python train_weakly_semi_weakly_supervised.py 
```

4. Test the model
```
python test_2D_boun.py
```

## Acknowledgement
The code is modified from [WSL4MIS](https://github.com/HiLab-git/WSL4MIS). 



## Note
* If you have any questions, feel free to contact Meng at (1155156866@link.cuhk.edu.hk)
