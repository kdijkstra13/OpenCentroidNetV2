# Getting started with OpenCentroidNetV2
CentroidNet is a hybrid convolutional neural network

1) run **create_dataset.py** to generate a synthetic dataset of your liking.
2) run **train.py** to train a model using this generated dataset.
3) adjust **config.py** to change hyper parameters.

# Generated data files and folders
## The input images and annotation data
**./data/dataset/** contains the image generated by **create_dataset.py** and the annotations in **train.csv** and **validation.csv**.\
Replace this data with your own set.

## The model
The weights of the CentroidNet model are stored in **./data/CentroidNet.pth**

## The input tensors to the model
**./data/validation_result/_inputs.npy** contains the normalized input tensor.\
**./data/validation_result/_targets.npy** contains the normalized target tensor.

## The output tensors of the model
**./data/validation_result/_centroid_vectors.npy** contains the normalized 2-d voting vectors (use this to see the quality of the training).\
**./data/validation_result/_border_vectors.npy** contains the normalized 2-d voting vectors (use this to see the quality of the training).\
**./data/validation_result/_centroid_votes.npy** contains the voting space (use this to tune Config.centroid_threshold and Config.nm-size).\
**./data/validation_result/_border_votes.npy** contains the voting space (use this to tune Config.centroid_threshold and Config.nm-size).\
**./data/validation_result/_overlay.png** contains circles drawn around each object.\
**./data/validation_result/_class_ids.npy** contains the class ids for every pixel.\
**./data/validation_result/_class_probs.npy** contains the class probability for every pixel.\
**./data/validation_result/validation.txt** contains the final circle shapes and class info.

# Citing OpenCentroidNetV2

If this code benefits your research please cite:

@article{dijkstra2020centroidnetv2,\
&nbsp;&nbsp;title={CentroidNetV2: A Hybrid Deep Neural Network for Small-Object Segmentation and Counting.},\
&nbsp;&nbsp;author={Dijkstra, Klaas and van de Loosdrecht, Jaap and Waatze A., Atsma and Schomaker, L.R.B. and Wiering, Marco A.},\
&nbsp;&nbsp;booktitle={Neurocomputing},\
&nbsp;&nbsp;year={2020},\
&nbsp;&nbsp;organization={Elsevier}\
&nbsp;&nbsp;DOI={https://doi.org/10.1016/j.neucom.2020.10.075}\
}

# p.s.
For a numpy viewer go to: https://github.com/ArendJanKramer/Numpyviewer.
