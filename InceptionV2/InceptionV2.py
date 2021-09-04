import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import  l2

class InceptionV2:
    
    def __init__(self):
        
        """
        The inception framework to implement the inception network v2
        from the original paper - 
        Rethinking the Inception Architecture for Computer Vision
        @ https://arxiv.org/abs/1512.00567
        
        """
        
        pass
    
    
    def __conv2d_bn(self, inp, filters, kernel_size, padding='same', strides=(1, 1), name=None):
        
        """
        Utility function to apply convolution operation 
        followed by batch normalization.

        Arguments:
            inp: input tensor.
            filters: number of filters for the convolution operation.
            kernel_size: size of the convolving kernel. Tuple of (height, width)
            padding: padding mode in `Conv2D`. Default is 'same'.
            strides: strides in `Conv2D`. Default is (1, 1).
                
        Return:
            out_tensor: Output tensor after the Convolution and BatchNormalization.
        """
        
        if name is not None:
            conv2d_name = name + "_conv2d"
            bn_name = name + "_bn"
        else:
            conv2d_name = "Conv2d"
            bn_name = "bn"
        
        layer = Conv2D(filters=filters, 
                       kernel_size=kernel_size,
                       strides=strides,
                       padding=padding,
                       name=conv2d_name)(inp)
        
        layer = BatchNormalization(axis=3, 
                                   scale=False,
                                   name=bn_name)(layer) # assume channels_last
        
        out_tensor = Activation(tf.nn.relu)(layer)
        
        return out_tensor
        
    def InceptionFigure5(self, in_layer, f1x1=64, f3x3_red=96, f3x3=128, f5x5_red=16, f5_3x3=32, fpool=32, name=None):
        
        """
        Constructs the Indeption Block as shows in
        Figure 5 in the original paper @
        https://arxiv.org/pdf/1512.00567.pdf
        
        The 5x5 convolution in InceptionV2 was replaced by 2 stacked 3x3 convolutions.
        This architectural modification by factorization resulted in computation cost 
        reduction with a relative gain of ~28%. 
        
        From the paper: "For the Inception part of the network, we have 3 traditional
        inception modules at the 35×35 with 288 filters each." 
        
        Parameters:
            in_layer: the input layer.
            f1x1: number of filters for the 1x1 convolutions
            f3x3_red: number of filters for the 1x1 convolutions that 
                      reduce the parameters befor applying the 3x3 convolutions.
            f3x3: number of filters for the 3x3 convolutions
            f5x5_red: number of filters for the 1x1 convolutions that 
                      reduce the parameters befor applying the 3x3 double convolutions.
            f5_3x3: number of filters for the 2 stacked 3x3 convolutions
            fpool: number of filters for the pooling layer.
        
        Return:
            out_layer: the inception block.
        
        """
        
        # #1×1 conv
        conv1x1 = self.__conv2d_bn(inp=in_layer, filters=f1x1, kernel_size=(1,1), padding='same', name=name+"_1x1")

        # 3x3 reduce (#3×3reduce) and #3x3 conv
        conv3x3_red = self.__conv2d_bn(inp=in_layer, filters=f3x3_red, kernel_size=(1,1), padding='same', name=name+"_3x3_reduce")
        conv3x3 = self.__conv2d_bn(inp=conv3x3_red, filters=f3x3, kernel_size=(3,3), padding='same', name=name+"_3x3")

        # 5x5 is now replaced by 2 3x3 conv operations
        conv_dbl_3x3_red = self.__conv2d_bn(inp=in_layer, filters=f5x5_red, kernel_size=(1,1), padding='same', name=name+"_dbl_3x3_reduce")
        conv_dbl_3x3 = self.__conv2d_bn(inp=conv_dbl_3x3_red, filters=f5_3x3, kernel_size=(3,3), padding='same', name=name+"_dbl_3x3_1")
        conv_dbl_3x3 = self.__conv2d_bn(inp=conv_dbl_3x3, filters=f5_3x3, kernel_size=(3,3), padding='same', name=name+"_dbl_3x3_2")

        # 3x3 max pooling and 1x1 projection layer - poolproj
        pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name=name + "_MaxPool2d")(in_layer)
        pool = self.__conv2d_bn(inp=pool, filters=fpool, kernel_size=(1,1), padding='same', name=name+"_1x1_projection")

        # concatenate the convolutional layers , poling layer to be passed
        # onto the next layers.
        out_layer = concatenate([conv1x1, conv3x3, conv_dbl_3x3, pool], axis=-1, name=name+"_concat")
        
        return out_layer
    
    def InceptionFigure6(self, in_layer, f1x1, f7x7_red, f7x7_1, f7x7_2, f7x7b2_dbl_red, f7x7b2_dbl1, f7x7b2_dbl2, fpool, name=None):
        
        """
        Constructs the Indeption Block as shows in
        Figure 6 in the original paper @
        https://arxiv.org/pdf/1512.00567.pdf
        
        "5×Inception As in figure 6 17×17×768 - 
        Inception modules after the factorization of the n × n
        convolutions. In our proposed architecture, we chose n = 7 for
        the 17 × 17 grid."
        
        Parameters:
            in_layer: the input layer.
            f1x1: number of filters in the 1x1 convolution block.
            f7x7_red: number of filters in the dimension reduction block before the single 7x7 convolution branch.
            f7x7_1: number of filters in the single 1x7 convolution branch. 
            f7x7_2: number of filters in the single 7x1 convolution branch. 
            f7x7b2_dbl_red: number of filters in the dimension reduction block before the double 7x7 convolution branch.
            f7x7b2_dbl1: number of filters in the 1x7 block in the double branch.  
            f7x7b2_dbl2: number of filters in the 7x1 block in the double branch (last 7x1 block). 
            fpool: number of filters for the pooling layer.
        
        Return:
            out_layer: the inception block (corresponding the figure 6 defined in the original paper).
        
        """
        
        # branch 1: 7x7 double convolutions using factorization of  
        # 2 stacked [(1x7), (7x1)] convolutions.
        branch7x7dbl_red = self.__conv2d_bn(inp=in_layer, filters=f7x7b2_dbl_red, kernel_size=(1,1), padding='same', 
                                       name=name+"_branch7x7dbl_reduce")
        branch7x7dbl = self.__conv2d_bn(inp=branch7x7dbl_red, filters=f7x7b2_dbl1, kernel_size=(1,7), padding='same',
                                        name=name+"_branch7x7dbl")
        branch7x7dbl = self.__conv2d_bn(inp=branch7x7dbl, filters=f7x7b2_dbl1, kernel_size=(7,1), padding='same', 
                                        name=name+"_branch1_7x1_3")
        branch7x7dbl = self.__conv2d_bn(inp=branch7x7dbl, filters=f7x7b2_dbl1, kernel_size=(1,7), padding='same', 
                                        name=name+"_branch1_1x7_4")
        branch7x7dbl = self.__conv2d_bn(inp=branch7x7dbl, filters=f7x7b2_dbl2, kernel_size=(7,1), padding='same', 
                                        name=name+"_branch1_7x1_5")
        
        # branch 2: 7x7 single convolution factorized to a set of 
        # [(1x7), (7x1)] convolutions.
        branch7x7_red = self.__conv2d_bn(inp=in_layer, filters=f7x7_red, kernel_size=(1,1), padding='same',
                                       name=name+"_branch2_1x1_1")
        branch7x7 = self.__conv2d_bn(inp=branch7x7_red, filters=f7x7_1, kernel_size=(1,7), padding='same', name=name+"_branch2_1x7_2")
        branch7x7 = self.__conv2d_bn(inp=branch7x7, filters=f7x7_2, kernel_size=(7,1), padding='same', name=name+"_branch2_7x1_3")

        # branch 3: 3x3 max pooling and 1x1 projection layer - poolproj
        pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name=name + "_MaxPool2d")(in_layer)
        branch_pool = self.__conv2d_bn(inp=pool, filters=fpool, kernel_size=(1,1), padding='same', name=name+"_branch3_1x1_projection_")
        
        # branch 4: #1×1 conv
        branch1x1 = self.__conv2d_bn(inp=in_layer, filters=f1x1, kernel_size=(1,1), padding='same', name=name+"_branch4_1x1_")

        # concatenate the convolutional layers , poling layer to be passed
        # onto the next layers.
        out_layer = concatenate([branch7x7dbl, branch7x7, branch_pool, branch1x1], axis=-1, name=name+"_concat")
        
        return out_layer
    
    def InceptionFigure7(self, in_layer, f7x7_red, f7x7, f3x3_red, f3x3, f1x1, fpool, name=None):
        
        """
        Constructs the Indeption Block as shows in
        Figure 7 in the original paper @
        https://arxiv.org/pdf/1512.00567.pdf
        
        "2×Inception As in figure 7 8×8×1280 - 
        Inception modules with expanded the filter bank outputs.
        This architecture is used on the coarsest (8 × 8) grids to promote
        high dimensional representations. This solution is used only on the coarsest grid,
        since that is the place where producing high dimensional sparse
        representation is the most critical as the ratio of local processing
        (by 1 × 1 convolutions) is increased compared to the spatial aggregation."
        
        Parameters:
            in_layer: the input layer.
            f7x7_red: number of filters in the dimension reduction block before the larger 7x7 convolution.
            f7x7: number of filters in the 7x7 convolution branch.
            f3x3_red: number of filters in the dimension reduction block before the larger 3x3 convolution.
            f3x3: number of filters in the 3x3 convolution branch 
            f1x1: number of filters in the 1x1 convolution branch  
            fpool: number of filters in the pooling branch.
            name: the generic name for the current inception block.
            
        Return:
            out_layer: the inception block (corresponding the figure 7 defined in the original paper).
        
        """
        
        # branch 1: 7x7 convolution branch split into two 3x3, [(1x3), (3x1)] 
        # convolution blocks respectively.
        branch7x7_red = self.__conv2d_bn(inp=in_layer, filters=f7x7_red, kernel_size=(1,1), padding='same', 
                                       name=name+"_branch7x7_reduce")
        branch7x7 = self.__conv2d_bn(inp=branch7x7_red, filters=f7x7, kernel_size=(3,3), padding='same',
                                        name=name+"_branch7x7_1")
        branch7x7_1 = self.__conv2d_bn(inp=branch7x7, filters=f7x7, kernel_size=(1,3), padding='same', 
                                        name=name+"_branch7x7_2")
        branch7x7_2 = self.__conv2d_bn(inp=branch7x7, filters=f7x7, kernel_size=(3,1), padding='same', 
                                        name=name+"_branch7x7_3")
        branch7x7 = concatenate([branch7x7_1, branch7x7_2], axis=-1, name=name+"_7x7_concat")
        
        # branch 2: 3x3 comvolution split into [(1x3), (3x1)] 
        # convolution blocks respectively.
        branch3x3_red = self.__conv2d_bn(inp=in_layer, filters=f3x3_red, kernel_size=(1,1), padding='same', 
                                       name=name+"_branch3x3_reduce")
        branch3x3_1 = self.__conv2d_bn(inp=branch3x3_red, filters=f3x3, kernel_size=(1,3), padding='same',
                                        name=name+"_branch3x3_1")
        branch3x3_2 = self.__conv2d_bn(inp=branch3x3_red, filters=f3x3, kernel_size=(3,1), padding='same', 
                                        name=name+"_branch3x3_2")
        branch3x3 = concatenate([branch3x3_1, branch3x3_2], axis=-1, name=name+"_3x3_concat")
        
        # branch 3: 3x3 max pooling and 1x1 projection layer - poolproj
        pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name=name + "_MaxPool2d")(in_layer)
        pool = self.__conv2d_bn(inp=pool, filters=fpool, kernel_size=(1,1), padding='same', name=name+"_1x1_projection")
        
        # branch 4: #1×1 conv
        branch1x1 = self.__conv2d_bn(inp=in_layer, filters=f1x1, kernel_size=(1,1), padding='same', name=name+"_branch4_1x1_")
        
        # concatenate the convolutional layers , poling layer to be passed
        # onto the next layers.
        out_layer = concatenate([branch7x7, branch3x3, pool, branch1x1], axis=-1, name=name+"_concat")
        
        return out_layer
        
        
        
        
        
        
        
        