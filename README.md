# Style Transfer Techniques

This folder includes Tensorflow Implementations of five style transfer techniques:
- Basic Style Transfer (as described [here][paper1])
- Patch based Style Transfer (as described [here][paper2])
- Doodle to Fine Art (as described [here][paper2])
- Region based Style Transfer 
- Gradient Mask based Style Transfer

The code has been written with the aims of providing an intuitive understanding of the technique, as well as facilitating experimentation.

*Note:* Pre-trained parameters of the VGG19 model are required in the local directory. Link to vgg19.npy can be found [here][npy]

## Simple Explanation of Style Transfer
[A Neural Algorithm of Artistic Style][paper1] introduced the foundational research where we can calculate a numeric representation of an image's style using outputs of layers in an image classification CNN.
Now, given two images, 1 and 2, we can not only calculate their respective styles but also calculate their difference. We refer to this difference as style loss. 

By backpropagating style loss to Image 1, we cause it to take on the style of Image 2. However, if ran for too many iterations, Image 1 just becomes a copy of Image 2 because that will minimize the style loss. 
To prevent this from happening, we also calculate the difference between the "stylised" Image 1 and what it originally looked like. We refer to this difference as content loss.

The combination of style and content loss means that the "stylised" image takes on the style of Image 2 while retaining the original content of Image 1. 


## Basic Style Transfer
`style_transfer.py` implements the foundational technique where the style of the whole style image is transferred to the whole content image. 

![basic style transfer][Fig1]

*Note:* when transferring style between images of vastly different dimensions, the style of each image should be normalized by the number of pixels in that image. This normalization factor can be adjusted to control the "intensity" of the transferred style.


## Patch based Style Transfer
`patch_style_transfer.py` implements a patch-based technique where the first content image is split into patches, and a second style image is split into semantically similar patches. These patches need not be the same size in both images. Then, for each patch we transfer the style from the second image to the content of the counterpart patch in the first image.
For example, if a patch in content image contains a tree and the sky, we want to transfer the style of a patch from the style image which which also contains a tree and the sky. It can be computationally difficult to findi semantically similar patches using the images, so we create simpler semantic maps instead. 

![patch based style transfer][Fig2]

*Note:* When calculating the style of a patch, we can mask the image by making all pixels outside of patch 0. However, this means that we are doing a lot of unecessary operations. So to avoid this, we extract the patch as a sub-image and perform the calculation.


## Doodle to Fine Art
`doodle_2_art.py` implements a variation of the patch based technique where there is no content image, only a content semantic map. This way when style transfer is performed, each patch essentially becomes a copy of the corresponding patch in style image.

![doodle to fine art][Fig3]

## Region based Style Transfer
`region_style_transfer.py` implements a region/mask based technique inspired by the patch based technique. Instead of transfering style between patches, we transfer syle between entire regions instead. 
Regions can also act as a hard mask where style is only transferred to certain regions of the content image.

![region based style transfer][Fig4]

*Note:* when transferring style between regions of vastly different sizes, we suggest the style of each region is normalized by the number of pixels in that region. 


## Gradient Mask based Style Transfer
`mask_style_transfer.py` implements a gradient mask based technique inspired by the patch based technique. We observed that the previous patch/region based techniques are essentially multiplying all pixels outside the patch/region by 0, while multiplying the pixels inside the patch/region by 1. 
The gradient mask technique extends this by allowing for values between 0-1.

![gradient mask based style transfer][Fig5]

*Note:* because we are using the entire image in the style calculation instead of extracting a sub-image, the operation is expensive. We found that having even just a few gradient masks will result in performance and/or memory issues.
*Note2:* we suggest that the style of an image without a mask is normalized by the number of pixels, and the style of an image with a mask is normalized by the sum of the mask (assuming its values lie in range 0-1)

[paper1]: https://arxiv.org/abs/1508.06576
[paper2]: http://arxiv.org/abs/1603.01768
[npy]: https://github.com/machrisaa/tensorflow-vgg
[Fig1]: examples/Fig1.jpg
[Fig2]: examples/Fig2.jpg
[Fig3]: examples/Fig3.jpg
[Fig4]: examples/Fig4.jpg
[Fig5]: examples/Fig5.jpg
