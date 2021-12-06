# Atomic Action Recognition for Surgical Scene Understanding by using Transformers

In this repository, you will find an approximation to the atomic surgery actions recognition task. We trained [Multiscale Vision Transformers](https://arxiv.org/abs/2104.11227.pdf) with PSI-AVA, a dataset created by the Biomedical Computer Vision (BCV) research group from Universidad de los Andes.

![MViTResults](MviTResults.png)

## Baseline

To approach the problem mentioned above, we proposed a baseline according to [Symmetric Dilated Convolution for Surgical Gesture Recognition](https://arxiv.org/pdf/2007.06373.pdf) (SdConv). This algorithm takes as inputs Temporal-Spatial Features resulting from a combination of CNN-TCN models. SdConv was proposed for the JIGSAWS dataset, which provides annotations of 15 surgical gestures that correspond to activity segments. Thus, we modified SdConv for PSI-AVA dataset. This dataset differs from JIGSAWS because PSI-AVA has assigned from one to three actions per instrument, and one frame can have more than one instrument. We also proposed other Temporal-Spatial Features derived from a CNN-LSTM combination, since the literature reports that LSTM has a good performance for this task too.

The metrics that we obtained can be seen as follows (you can note that they differ slightly from the article as the models were not correctly saved):

| Metrics | PSI-AVA (TCN) | PSI-AVA (LSTM) | JIGSAWS (TCN) | JIGSAWS (LSTM) |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | 
| **Acc** | 0.1841 | 0.1130 | 0.6593 | 0.6591 |
| **F-Score** | 0.4081 | 0.3740 | 0.6593 | 0.6591 |

In this repository, there is the option of running test with the model for each of the dataset and the CNN-TCN or CNN-LSTM implementation.

Start by creating the following softlinks to access the data and trained models: 
```
    ln -s /media/user_home0/mverlyck/AMLProject/JIGSAWS/ JIGSAWS
    
    ln -s /media/user_home0/mverlyck/AMLProject/PSI-AVA_code/PSI-AVA/ PSI-AVA
    
    ln -s /media/user_home0/mverlyck/AMLProject/Models/ Models
```
The followings lines allow to run test with the baseline method for both dataset, first with TCN features and then with LSTM features:

```
    python main.py --method baseline_PSIAVA --test --n_classes 16 --data_root PSI-AVA/STFeatures --test_label PSI-AVA/splits/Split_7/test.txt --checkpoint Models/PSI-AVA/TCN_best_PSIAVA_split7.pth
    
    python main.py --method baseline_JIGSAWS --test --n_classes 15 --data_root JIGSAWS/STFeatures --test_label JIGSAWS/splits/Split_2/test.txt --checkpoint Models/JIGSAWS/TCN_best_JIGSAWS_split2.pth
    
    python main.py --method baseline_PSIAVA --test --n_classes 16 --data_root PSI-AVA/LSTMFeatures --test_label PSI-AVA/splits/Split_4/test.txt --checkpoint Models/PSI-AVA/LSTM_best_PSIAVA_split4.pth
    
    python main.py --method baseline_JIGSAWS --test --n_classes 15 --data_root JIGSAWS/LSTMFeatures --test_label JIGSAWS/splits/Split_3/test.txt --checkpoint Models/JIGSAWS/LSTM_best_JIGSAWS_split3.pth
   
```

It is important to highlight that these lines allow to run test on the split which obtains best metrics.

## Final Method

### Preparation

To implement our final method, you must verify some libraries versions and run the following lines first:
- Pytorch = 1.10
- CUDA = 10.2
```
conda create -n mvit anaconda python=3.8
    
conda activate mvit
    
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
```
- simplejson: `pip install simplejson`
- PyAV: `conda install av -c conda-forge`
- skelearn: `pip install sklearn`
- OpenCV: `pip install opencv-python`
- FairScale: `pip install 'git+https://github.com/facebookresearch/fairscale'`


The following lines allow to run test and demo with the final method MViT:

```
    python main.py --method MVIT --test --n_classes 16 --cfg method/configs/PSI-AVA/MVIT.yaml 
    
    python main.py --method MVIT --demo --img PSI-AVA/data/CASE001/00000.png --n_classes 16 --cfg method/configs/PSI-AVA/MVIT.yaml 
   
```
NB: the demo is done with images from the best split. Therefore you have to choose an annotated image from CASE001 from 00000 to 10255. Frames are annotated each 35s, therefore you can run demo on 00000, 00035, 00070, etc..

The visualization of the demo is saved at method/prediction.png.


## Credits

Our algorithms were based on:

https://github.com/lulucelia/SdConv

https://github.com/facebookresearch/SlowFast/blob/main/projects/mvit


