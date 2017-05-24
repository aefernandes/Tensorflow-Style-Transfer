"""
Tensorflow Implementation of "Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks"

Academic Paper: https://arxiv.org/pdf/1603.01768.pdf
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
        img: TF Tensor with shape [N,H,W,C]. Values are scaled to 0.0-1.0
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
    
    # Do not include FC layers to allow for variable sized images
    vgg_layers = ["conv1_1", "conv1_2", "pool1",
              "conv2_1", "conv2_2", "pool2",
              "conv3_1", "conv3_2", "conv3_3", "conv3_4", "pool3",
              "conv4_1", "conv4_2", "conv4_3", "conv4_4", "pool4",
              "conv5_1", "conv5_2", "conv5_3", "conv5_4", "pool5"]
              
    # Initialize dictionary to store the layers
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
    
    
def StyleLayerPatches(layer, patches):
    """
    Description:
        For each patch, extracts a subset of Features from Array of Column Vectors before 
        calculating a Gramm Matrix
        
    Args:
        layer: Tensorflow Tensor with shape [D,C] (expecting content layer)
        patches: shape [M,P] where M is the number of patches, and P is the no. of pixels in patch.
                   Values should be in range 0..D-1, which are idxs of first dimension in layer
        
    Returns:
        Gramm Matrixs with shape [M,C,C]
    """      
    # idxs has shape [M,P]
    idxs = tf.constant(patches, dtype=tf.int64)
    
    # layer_patches has shape [M,P,C]
    layer_patches = tf.gather(layer, indices=idxs)
    # M may be quite large, so to avoid OOM issues, we do M separate Gramm Matrix 
    # calculations instead of one big row-wise calculation
    layer_patches = tf.split(layer_patches, num_or_size_splits=1)
        
    # merge M x [C,C] Gramm Matrixs to shape [M,C,C]
    return tf.concat([StyleLayer(p) for p in layer_patches], axis=0)
    
"""
Helper Functions
"""
def NHWC(img):
    """
    Args:
        img: an image loaded using skimage with shape [H,W,C]
        
    Returns:
        img casted to float32 and reshaped to [N,H,W,C]
    """
    return np.expand_dims(a=img, axis=0).astype(np.float32)    


def GeneratePatchIdxs(img_shape, stride=3, patch_size=3):
    """
    Description: 
        Generates idxs which can be used to extract patch_size x patch_size patches 
        from a flattened image. Note that image is not padded
        
        e.g. image  [[0 1 2 3 4]     flattened image [0 1 2 3 4 5 6 7 8 9]
                    [5 6 7 8 9]]
            
                    2x2 patches with stride of 2 results in idxs [[0,1,5,6],[2,3,7,8]]
    Args:
        img_shape: array of 3 values [H,W,C]
        stride: no. of pixels between each patch. (stride=patch_size for non overlapping)
        patch_size: length of side of square patch
        
    Returns:
        idxs corresponding to patches in flattened image
        has shape [N,P] where N = number of non-overlapping patches, P = patch_size x patch_size
    """
    # determine how many patches we can fit along the image height and width
    # number of patches = rows*cols
    rows = int(np.ceil((img_shape[0] - patch_size + 1) / stride))
    cols = int(np.ceil((img_shape[1] - patch_size + 1) / stride))
    
    # create a 2D array same size as the image with cells containing a value 
    # corresponding to their idx when flattened
    idxs = np.arange(0, img_shape[0] * img_shape[1])
    idxs = np.reshape(idxs, newshape=[img_shape[0], img_shape[1]])
    
    # loop through every patch: extract the idxs, and flatten to 1D vector
    # patch_idxs has shape [N,P] where N = rows x cols, and P = patch_size x patch_size
    patch_idxs = [idxs[i*stride:i*stride+patch_size, j*stride:j*stride+patch_size].reshape([-1])
                                   for i in range(rows)
                                   for j in range(cols)] 
    # convert to numpy array                               
    patch_idxs = np.array(patch_idxs, dtype=np.int64)
                                   
    return patch_idxs, rows, cols
      
    
def PatchMatching(content_map, style_map, patch_size=3): 
    """
    Description: 
        For every patch in content_map, find Nearest Neighbour patch in style_map.
        Similarity measure is cosine similarity
        
        e.g. content_map  [[0 1 2 3 4]     style_map    [[0 1 2 3]
                           [5 6 7 8 9]]                 [4 5 6 7]]
            
             assume patch 1 in content_map is most similar to patch 2 in style_map
             and patch 2 in content_map is most similar to patch 1 in style_map
             
             returns [[0,1,5,6],[2,3,7,8]]      [[2,3,6,7],[0,1,4,5]]
             
    Args:
        content_map: an image with shape [H1,W1,C]
        style_map: an image with shape [H2,W2,C]
        patch_size: length of a side of the square patch
        
    Returns:
        a tuple of 2 arrays, both with shapes [N1,P] where:
            N1 = number of patches in content_map 
            P = patch size x patch size
            array1 = idxs of every patch in content_map
            array2 = idxs of Nearest Neighbour patch in style_map corresponding to the 
                        patch in array1 of the same index
                Note that patches in style_map may appear multiple times or not at all                 
    """  
    # flatten the maps from [H,W,C] to [HxW,C]
    content_map_flat = np.reshape(content_map, [-1, content_map.shape[2]])
    style_map_flat = np.reshape(style_map, [-1, style_map.shape[2]])  
       
    # generate the idxs to extract the patches from the maps  
    # c_patch_idxs has shape [N1,P] where N1 = c_rows x c_cols, and P = patch_size x patch_size
    # s_patch_idxs has shape [N2,P] where N2 = s_rows x s_cols, and P = patch_size x patch_size                     
    c_patch_idxs, c_rows, c_cols = GeneratePatchIdxs(content_map.shape, stride=patch_size, patch_size=patch_size)
    s_patch_idxs, s_rows, s_cols = GeneratePatchIdxs(style_map.shape, stride=1, patch_size=patch_size) 
    
    # extract the patches from the maps, and further flatten the channels for each patch
    # c_patches has shape [N1,PxC]
    # s_patches has shape [N2,PxC]
    c_patches = content_map_flat[c_patch_idxs,:]   
    c_patches = np.reshape(c_patches, [c_rows*c_cols, -1])              
    s_patches = style_map_flat[s_patch_idxs,:]   
    s_patches = np.reshape(s_patches, [s_rows*s_cols, -1])  
    
    # calculate the euclidian norm for each patch
    # c_patches_norm has shape [N1,1]
    # s_patches_norm has shape [N2,1]
    c_patches_norm = np.linalg.norm(c_patches, axis=1, keepdims=True)
    s_patches_norm = np.linalg.norm(s_patches, axis=1, keepdims=True)
    
    # normalize each patch, making it a unit vector
    c_patches = np.divide(c_patches, c_patches_norm)
    s_patches = np.divide(s_patches, s_patches_norm)
    
    # calculate the cosine similarity between every patch in c_patches with 
    # every patch in s_patches
    # similarity has shape [N1,N2]
    similarity = np.dot(c_patches, np.transpose(s_patches))
    
    # every time a patch in s_patches is chosen we discount it to reduce 
    # the chance of it being picked again
    discount = np.zeros(s_rows*s_cols, dtype=np.float32)
    
    # an array to store the Nearest Neighbour patches 
    s_nn_patch_idxs = np.zeros_like(c_patch_idxs, dtype=np.int64)
    
    # loop through every c_patch
    for i in range(c_rows*c_cols): 
        
        max_similarity = np.max(similarity[i,:])
                
        # find the Nearest Neighbour patch
        nn = np.random.choice(np.where(similarity[i,:] > max_similarity - 0.0002)[0])
        
        # reduce the chance of that patch being picked again
        discount[nn] += 0.002
                
        s_nn_patch_idxs[i,:] = s_patch_idxs[nn,:]
        
    return (c_patch_idxs, s_nn_patch_idxs)
    
   
    
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

# Length of side of Square Patch
patch_size = 8


"""
Nearest Neighbour Patch Matching
"""   
print("Matching Patches in Semantic Maps")


# Dictionary to store the patch idxs
content_map_idxs = {}
style_map_idxs = {}

# unique no. of times layers in style layers have been pooled
# e.g. both conv2_1 and conv2_2 have been pooled only once
pool_layers = set([int(layer[4]) for layer in style_layers])


content_map_shape = np.array(content_map.shape)
style_map_shape = np.array(style_map.shape)
for pool in range(1,6):
    if pool in pool_layers:
        # Divides the content map into non overlapping patches and for each 
        # patch find the most similar patch in style map
        # content_map_idxs[pool] has shape [N,P] where N is number of patches, P is patch area
        # style_map_idxs[pool] has shape [N,P] where N is number of patches, P is patch area
        # style_map_idxs[pool][i] is the most similar patch to content_map_idxs[pool][i]        
        content_map_idxs[pool], style_map_idxs[pool] = PatchMatching(
                                      skimage.transform.resize(content_map, content_map_shape),
                                      skimage.transform.resize(style_map, style_map_shape),
                                      patch_size) 
    
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
    
# for every layer in style layers, create a style layer
for layer in style_layers:
    pool = int(layer[4])
    vgg["style_" + layer] = StyleLayerPatches(vgg["feat_" + layer], style_map_idxs[pool])

    
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
        name = "style_" + layer
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
    
# for every layer in style layers, create a style layer
for layer in style_layers:
    pool = int(layer[4])
    vgg["style_" + layer] = StyleLayerPatches(vgg["feat_" + layer], content_map_idxs[pool])

    
  
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
    name = "style_" + layer
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