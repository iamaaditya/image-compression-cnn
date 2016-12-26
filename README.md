This code is part of the paper <arxiv paper id>. It consists of two parts:
1. Code to generate Multi-structure region of interest (MSROI)
2. Code to use MSROI map to semantically compress image as JPEG

Table of Contents
=================

   * [How to use this code ?](#how-to-use-this-code-)
      * [Generating Map](#generating-map)
      * [Compressing image using the Map](#compressing-image-using-the-map)
      * [Training your own model](#training-your-own-model)
      * [Evaluating metrics](#evaluating-metrics)
   * [Multi-Structure Region-of-interest](#multi-structure-region-of-interest)
      * [What this is ?](#what-this-is-)
      * [What this is NOT ?](#what-this-is-not-)
      * [Design Choices](#design-choices)
   * [FAQ about image compression](#faq-about-image-compression)

# How to use this code ?

## Generating Map

    ```
    python generate_map.py <image_file>
    ```

## Compressing image using the Map
    

## Training your own model

## Evaluating metrics


# Multi-Structure Region-of-interest

## What this is ?
    * Find all semantic regions in an image in a single pass
    * Train without the localization data
    * Maximize the number of objects detected (maybe all?)
    * Need not be precise
    * It is used for image compression because we need less precision 
    but more generic information about the content of the image


## What this is NOT ?

* Not an object detector. For that checkout-
    *[Fast RCNN](https://github.com/rbgirshick/fast-rcnn)
    *[Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn) 
* Not a weakly labelled class detector or Class activation Map. For that checkout -
    *[Weakly detector](https://github.com/jazzsaxmafia/Weakly_detector) 
    *[CAM](https://github.com/metalbubble/CAM) 
* Not saliency map or guided backprop. For that checkout -
    *[Lasagne saliency](https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb) 
    *[Grad-CAM](https://github.com/ramprs/grad-cam) 
* Not Semantic segmentation. For that checkout -
    *[Oxford CRF CNN](https://github.com/torrvision/crfasrnn) 
    *[Fully convolutional neural network](https://github.com/shelhamer/fcn.berkeleyvision.org) 


## Design Choices
    
    * Tensorflow 3D convolutions

    * Multi-label nn.softmax instead of nn.sparse
        (non-exclusive classes)

    * Argsort and not argmax
    

# FAQ about image compression

1. Is the final image really a standard JPEG?
   Yes, the final image is a standard JPEG as it is encoded using standard JPEG.

2. But how can you improve JPEG using JPEG ?
   Standard JPEG uses a image level Quantization scaling Q. However, not all parts 
   of the image be compressed at same level. Our method allows to use variable Q.

3. Don't we have to store the variable Q in the image file?
   No. Because the final image is encoded using a single Q. Please see Section 4 of our paper. 


