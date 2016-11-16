# Multi-Structure Region-of-interest


## What this is ?
    * Find all semantic regions in an image in a single pass
    * Train without the localization data
    * Maximize the number of objects detected (maybe all?)
    * Need not be precise


## What this is NOT ?

* Not an object detector. 
    For that checkout
    * Fast RCNN, Faster-RCNN 
* Not a weakly labelled class detector or Class activation Map
    For that checkout
    * Weakly detector
    * CAM
* Not saliency map or guided backprop
    For that checkout
    * Lasagne saliency
    * Grad-CAM
* Not Semantic segmentation
    For that checkout
    * Oxford CRF CNN
    * Fully convolutional neural network

## What is the intended use ?
    
    __Image Compression__

    Say what ?

## Design Choices
    

    * Tensorflow 3D convolutions

    * Multi-label nn.softmax instead of nn.sparse
        (non-exclusive classes)

    * Argsort and not argmax
    

## FAQ

1. But how can you improve JPEG?
