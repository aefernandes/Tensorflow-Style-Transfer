"""
Tensorflow Implementation of Region based Style Transfer

Academic Paper: 
Author: Ying Chan
GitHub: https://github.com/yc5915/Tensorflow-Implementations/

"""

import skimage.io
import skimage.transform
import tensorflow as tf
import numpy as np

""" 
Model Creation Functions
"""

def VGG19(img, vgg_path='vgg19.npy'):
    """
    Description:
        Creates a pre-trained VGG19 model without the Fully Connected Layers.
        This allows the input image to have a variable size.
        
        link to download vgg19.npy @ https://github.com/machrisaa/tensorflow-vgg
        
    Args:
        img: TF Tensor with shape [N,H,W,C]. Values should be in range 0.0-1.0
        vgg_path: path to vgg19.npy
        
    Returns:
        vgg: dictionary of layers in VGG19 model
    """
        
    # Scale img to 0.0-255.0 and split into R, G, B channels
    red, green, blue = tf.split(img * 255.0, num_or_size_splits=3, axis=3)
        
    # Normalize the input image by subtracting mean
    # Gather the channels in B, G, R order
    vgg_mean = [103.939, 116.779, 123.68]    
    bgr = tf.concat(axis=3, values=[
        blue - vgg_mean[0],
        green - vgg_mean[1],
        red - vgg_mean[2],
    ])
    
    # Load pre-trained VGG19 parameters
    vgg_params = np.load(vgg_path, encoding='latin1').item()
    
    # Note that we do not include FC layers
    vgg_layers = ["conv1_1", "conv1_2", "pool1",
              "conv2_1", "conv2_2", "pool2",
              "conv3_1", "conv3_2", "conv3_3", "conv3_4", "pool3",
              "conv4_1", "conv4_2", "conv4_3", "conv4_4", "pool4",
              "conv5_1", "conv5_2", "conv5_3", "conv5_4", "pool5"]
              
    # Dictionary to store the layers
    vgg = {}
    
    # Create VGG19 model and load pre-trained parameters
    curr = bgr
    for layer in vgg_layers:
        if layer[:4] == "conv": 
            curr = tf.nn.conv2d(curr, filter=vgg_params[layer][0], 
                                     strides=[1, 1, 1, 1], padding='SAME')
            curr += vgg_params[layer][1]
            curr = tf.nn.relu(curr)
        elif layer[:4] == "pool":
            curr = tf.nn.max_pool(curr, ksize=[1, 2, 2, 1], 
                                  strides=[1, 2, 2, 1], padding='SAME')
        vgg[layer] = curr

    return vgg

    
def FeatureLayer(layer):
    """
    Description:
        Flattens 2D images into 1D vectors
        
    Args:
        layer: TF Tensor with shape [N,H,W,C] (expecting N=1 image)
        
    Returns:
        Flattened layer with shape [D,C] where:
            D = NxHxW   No. of Features in a Column Vector
            C           Number of Column Vectors
    """
    shape = layer.get_shape().as_list() 
    return tf.reshape(layer, shape=[-1, shape[3]])
    

def StyleLayer(layer):
    """
    Description:
        Calculates a Gramm Matrix from Array of Column Vectors 
        (uses all Features in a Column Vector)
        
    Args:
        layer: TF Tensor with shape [D,C] (expecting FeatureLayer output)
        
    Returns:
        Gramm Matrix with shape [C,C]
    """      
    return tf.matmul(a=layer, b=layer, transpose_a=True)
    
    
def StyleLayerRegion(layer, region):
    """
    Description:
        Extracts a subset of Features from Array of Column Vectors before 
        calculating a Gramm Matrix
        
    Args:
        layer: TF Tensor with shape [D,C] (expecting FeatureLayer output)
        region: 1D array with shape [F] where F is number of features to be extracted
                Values should be in range 0..D-1, which are idxs of first dimension in layer
        
    Returns:
        Gramm Matrixs with shape [C,C]
    """      
    # idxs has shape [F]
    idxs = tf.constant(region, dtype=tf.int64)
    
    # layer_region has shape [F,C]
    layer_region = tf.gather(layer, indices=idxs)
        
    # Calculate Gramm Matrix with subset of Features
    return StyleLayer(layer_region)
    
"""
Helper Functions
"""
def NHWC(img):
    """
    Description:
        Transforms image with shape [H,W,C] to [N,H,W,C] where N = 1
    """
    return np.expand_dims(a=img, axis=0).astype(np.float32)    
    
def RegionIdxsByColors(img, colors, maxdist = 0.14):
    """
    Description:
        For each color in colors, find idx of pixels in the flattened image that are similar
        Uses euclidian distance for similarity measure
        
    Args:
        img: an image with shape [H,W,C]. values should be in range 0.0-1.0
        colors: an array of colors to be extracted. has shape [N,C] where N is number of colors
                values should be in range 0.0-1.0
        maxdist: maxmimum euclidian distance for a color to be considered similar
        
    Returns:
        N x 1D arrays. Arrays can have different lengths. 
        Values are in range 0...HxW-1 (idxs in flattened image)
    """
    # Flattens image, except for channels
    img_flat = np.reshape(img, [-1, img.shape[2]])
    
    region_idxs = []
    for i in range(len(colors)): 
        # calculate similarity between image and a color
        similarity = np.sqrt(np.sum(np.square(img_flat - colors[i,:]), axis=1))
        
        # extract the idxs in the image where the color is similar
        region_idxs.append(np.where(similarity <= maxdist)[0])
    return region_idxs

    
"""
Parameters
"""    
# Define layers of VGG19 model to use for Content and Style
content_layers = ["conv4_2"]
style_layers = ["conv2_1","conv3_1"]

# Load images & scale values from 0.0-255.0 to 0.0-1.0
content_img = skimage.io.imread("images\\Seth.jpg") / 255.0
style_img = skimage.io.imread("images\\Gogh.jpg") / 255.0


# Load semantic map images
content_map = skimage.io.imread("images\\Seth_sem.png") / 255.0
style_map = skimage.io.imread("images\\Gogh_sem.png") / 255.0

# Color (RGB) of the regions to do style transfer
region_colors = np.array([[0,0,255]]) #blue
region_colors = region_colors / 255.0


"""
Region Extraction
"""   
print("Extracting Regions from Semantic Maps")

# Dictionary to store the region idxs
content_map_idxs = {}
style_map_idxs = {}

# unique no. of times layers in style layers have been pooled
# e.g. both conv2_1 and conv2_2 have been pooled only once
pool_layers = set([int(layer[4]) for layer in style_layers])


content_map_shape = np.array(content_map.shape)
style_map_shape = np.array(style_map.shape)
for pool in range(1,6):
    if pool in pool_layers:
        # Find idxs in the maps which are similar to each color in region_colors
        content_map_idxs[pool] = RegionIdxsByColors(
                                  skimage.transform.resize(content_map, content_map_shape),
                                  colors=region_colors)
        
        style_map_idxs[pool] = RegionIdxsByColors(
                                  skimage.transform.resize(style_map, style_map_shape),
                                  colors=region_colors)
    
    # resize the images the same way as VGG19's max pool layers
    content_map_shape[0:2] = (content_map_shape[0:2] + 1) // 2
    style_map_shape[0:2] = (style_map_shape[0:2] + 1) // 2
               


    
"""
Evaluate Content and Style Image
"""   
print("Building Model to Evaluate Content & Style Image")

tf.reset_default_graph()

# Build a Model with variable sized image
img = tf.placeholder(tf.float32, shape=[1,None,None,3])
vgg = VGG19(tf.Variable(img, dtype=tf.float32, validate_shape=False))

# for every layer in style and content layers, create a feature layer
for layer in set([*content_layers, *style_layers]):
    vgg["feat_" + layer] = FeatureLayer(vgg[layer])
    
# for every layer in style layers and for every region, create a style layer
for layer in style_layers:
    pool = int(layer[4])
    for i in range(len(region_colors)):
        vgg["style" + str(i) + "_" + layer] = StyleLayerRegion(vgg["feat_" + layer], style_map_idxs[pool][i])
    
    
# Dictionaries to store Content and Style Layer outputs
content = {}
style = {}

with tf.Session() as sess:
    print("Evaluating Content Image")
    # initialize img as Content Image
    sess.run(tf.global_variables_initializer(), feed_dict={img: NHWC(content_img)})
    for layer in content_layers:
        name = "feat_" + layer
        content[name] = vgg[name].eval()

    print("Evaluating Style Image")
    # initialize img as Style Image
    sess.run(tf.global_variables_initializer(), feed_dict={img: NHWC(style_img)})
    for layer in style_layers:
        for i in range(len(region_colors)):
            name = "style" + str(i) + "_" + layer
            style[name] = vgg[name].eval()


  
"""
Style Transfer Model
"""          
print("Building Model to Transfer Style")       
  
tf.reset_default_graph()

# Build a Model with Content Image (alternatively can use random noise)
# We use Variable instead of placeholder as we wish to make use of Tensorflow's optimizers 
img = tf.Variable(NHWC(content_img))
vgg = VGG19(img)

# for every layer in style and content layers, create a feature layer
for layer in set([*content_layers, *style_layers]):
    vgg["feat_" + layer] = FeatureLayer(vgg[layer])
    
# for every layer in style layers, create a masked style layer
for layer in style_layers:
    pool = int(layer[4])
    for i in range(len(region_colors)):
        vgg["style" + str(i) + "_" + layer] = StyleLayerRegion(vgg["feat_" + layer], content_map_idxs[pool][i])

    
  
"""
Loss Function
"""    
  
print("Creating Loss Function") 
    
content_loss = 0
for layer in content_layers:
    name = "feat_" + layer
    content_loss += tf.reduce_mean(tf.square(content[name] - vgg[name]))
    
content_loss /= len(content_layers) # each content layer has equal weight


style_loss = 0
for layer in style_layers: 
    for i in range(len(region_colors)):
        name = "style" + str(i) + "_" + layer
        style_loss += tf.reduce_mean(tf.square(style[name] - vgg[name]))
                    
style_loss /= len(style_layers) # each style layer has equal weight

               
# Total variation loss (encourages adjacent pixels to be similar color)
tv_loss = tf.reduce_mean(((img[:,1:,:-1,:] - img[:,:-1,:-1,:])**2 + (img[:,:-1,1:,:] - img[:,:-1,:-1,:])**2)**1.25)


loss = 5*content_loss + 25*style_loss + 1*tv_loss 
        


"""
Backprop
"""   
print("Stylising")

# Backpropagate error to img using scipy's L-BFGS optimizer
train_step =tf.contrib.opt.ScipyOptimizerInterface(loss, 
                                                   var_list=[img], 
                                                   options={"maxiter":200, "disp" : True})
        
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_step.minimize(sess, 
                        fetches=[loss], 
                        loss_callback=lambda x : print("loss: %f"%x))

    # img is TF Tensor with shape [N,H,W,C]. Discard the first dimension
    stylised_img = img.eval()[0]
    
    # Clip the values to 0.0-1.0
    stylised_img = np.clip(stylised_img, 0, 1)
    
    # Show img
    skimage.io.imshow(stylised_img)
  
    
     
 