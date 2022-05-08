# PMSnetwork
Depth estimation from stereo images is essential to computer vision applications. Given a pair of rectified stereo images, the goal of depth estimation is to compute the disparity d. Disparity refers to the horizontal displacement between a pair of corresponding pixels on the left and right images for each pixel in the reference image. The traditional pipeline for stereo matching involves the finding of corresponding points based on matching cost and post-processing. Although CNN yields significant gains compared to conventional approaches in terms of both accuracy and speed, it is still difficult to find accurate corresponding points in inherently ill-posed regions such as occlusion areas, repeated patterns, textureless regions, and reflective surfaces. Solely applying the intensity-consistency constraint between different viewpoints is generally insufficient for accurate correspondence estimation in such ill-posed regions, and is useless in textureless regions. Therefore, regional support from global context information must be incorporated into stereo matching.

In this project, a novel pyramid stereo matching
network (PSMNet) is used to exploit global context information in stereo matching. Spatial pyramid pooling and dilated convolution are used to enlarge the receptive fields. In this way, PSMNet extends pixel-level features to region-level features with different scales of receptive fields; the resultant combined global and local feature clues are used to form the cost volume for reliable disparity estimation. Moreover, we design a stacked hourglass 3D CNN in conjunction with intermediate supervision to regularize the cost volume. The stacked hourglass 3D CNN repeatedly processes the cost volume in a top-down/bottomup manner to further improve the utilization of global context information.

This repository contains the code (in PyTorch) for ["Pyramid Stereo Matching Network" paper](https://arxiv.org/pdf/1803.08669.pdf ) (CVPR 2018) by Jia-Ren Chang and Yong-Sheng Chen. 

Python files are organized as following:

1- preprocessing.py ---> responsible for processing images (normalization and augmentation)

2- data_loader.py  ---> -responsible for loading dataset and splitting it into tain-validation-test sets, also it feeds the images into preprocessing pipeline if needed.
                        - Map-style dataset class is created to get fed into TORCH.UTILS later on.
                      
3-building_blocks.py---> contains building blocks to be used on to create the full model (e.g. Hourglass and basic block)

4-feature_estimation.py ---> contains the CNN that extracts feature before the PMS model

5-model.py ---> calls building_blocks.py and feature_estimation.py to build the whole model
