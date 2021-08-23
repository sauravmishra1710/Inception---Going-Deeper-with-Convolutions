import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation, Input, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GaussianNoise, GaussianDropout
from tensorflow.keras.layers import Activation, Dropout, ZeroPadding3D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

class InceptionFramework:
    
    def __init__(self):
        
        """
        The inception framework to implement the inception network v1
        from the original paper - Going deeper with convolutions
        @ https://arxiv.org/abs/1409.4842
        
        """
        
        pass
    
    def conv2d_bn(self, inp, filters, kernel_size, padding='same', strides=(1, 1)):
        
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
        
        layer = Conv2D(filters=filters, 
                       kernel_size=kernel_size,
                       strides=strides,
                       padding=padding)(inp)
        
        layer = BatchNormalization(axis=3, 
                                   scale=False)(layer) # assume channels_last
        
        out_tensor = Activation(tf.nn.relu)(layer)
        
        return out_tensor
    
    
    def Naive_Inception_Module(self, f1x1, f3x3, f5x5, INPUT_SHAPE=(299, 299, 3)):
        
        """
        Builds the naive incception module where convolution on an input is performed
        with 3 different sizes of filters (1x1, 3x3, 5x5). 
        Additionally, max pooling is also performed. The outputs are concatenated 
        and sent to the next inception module.
        
        Parameters:
            f1x1: number of filters for the 1x1 convolutions
            f3x3: number of filters for the 3x3 convolutions
            f5x5: number of filters for the 5x5 convolutions
            INPUT_SHAPE: the input layer shape. Default Valus is (299, 299, 3).
        
        Return:
            model: the keras model instance.
        
        """
        
        # conv2d_bn(inp, filters, kernel_size, padding='same', strides=(1, 1))
        
        # model input
        in_layer = Input(shape = INPUT_SHAPE)
        
        # 1x1 conv
        conv1x1 = self.conv2d_bn(inp=in_layer, filters=f1x1, kernel_size=(1,1), padding='same')
        
        # 3x3 conv
        conv3x3 = self.conv2d_bn(inp=in_layer, filters=f3x3, kernel_size=(3,3), padding='same')
        
        # 5x5 conv
        conv5x5 = self.conv2d_bn(inp=in_layer, filters=f5x5, kernel_size=(5,5), padding='same')
        
        # 3x3 max pooling
        pool3x3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(in_layer)
        
        # concatenate the convolution and the pooling layers,
        out_tensor = concatenate([conv1x1, conv3x3, conv5x5, pool3x3], axis = -1)
        
        # create model
        model = Model(inputs = in_layer, outputs = out_tensor, name = "Naive Inception Block")
        
        return model
    
    def Build_Inception_With_Dimension_Reduction(self, in_layer, f1x1, f3x3_red, f3x3, f5x5_red, f5x5, fpool):
        
        """
        Builds the incception module where convolution on an input is performed
        with 3 different sizes of filters (1x1, 3x3, 5x5) with dimension reduction. 
        Additionally, max pooling is also performed. The outputs are concatenated 
        and sent to the next inception module.
        
        Parameters:
            f1x1: number of filters for the 1x1 convolutions
            f3x3_red: number of filters for the 1x1 convolutions that 
                      reduce the parameters befor applying the 3x3 convolutions.
            f3x3: number of filters for the 3x3 convolutions
            f5x5_red: number of filters for the 1x1 convolutions that 
                      reduce the parameters befor applying the 5x5 convolutions.
            f5x5: number of filters for the 5x5 convolutions
            fpool: number of filters for the pooling layer.
            INPUT_SHAPE: the input layer shape. Default Values is (299, 299, 3).
        
        Return:
            model: the inception block.
        
        """
        
        # 1x1 conv
        conv1x1 = Conv2D(f1x1, (1,1), padding='same', activation=tf.nn.relu)(in_layer)

        # 3x3 reduce and conv
        conv3x3_red = Conv2D(f3x3_red, (1,1), padding='same', activation=tf.nn.relu)(in_layer)
        conv3x3 = Conv2D(f3x3, (3,3), padding='same', activation=tf.nn.relu)(conv3x3_red)

        # 5x5 reduce and conv
        conv5x5_red = Conv2D(f5x5_red, (1,1), padding='same', activation=tf.nn.relu)(in_layer)
        conv5x5 = Conv2D(f5x5, (5,5), padding='same', activation=tf.nn.relu)(conv5x5_red)

        # 3x3 max pooling
        pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(in_layer)
        pool = Conv2D(fpool, (1,1), padding='same', activation=tf.nn.relu)(pool)

        # concatenate the convolutional layers , poling layer to be passed
        # onto the next layers.
        out_layer = concatenate([conv1x1, conv3x3, conv5x5, pool], axis=-1)
        
        return out_layer
    
    def Build_Sample_Inception_Network(self, INPUT_SHAPE=(299, 299, 3)):
        
        """
        Builds a sample incception network with the parameters taken 
        from the inception blocks 3(a) and 3(b) defined in the 
        Table 1: GoogLeNet incarnation of the Inception architecture in the paper
        @ https://arxiv.org/pdf/1409.4842.pdf
        
        Parameters:
            
            INPUT_SHAPE (Optional): the input layer shape. Default Valus is (299, 299, 3).
        
        Return:
            model: the keras model instance.
        
        """
        
        # define model input
        inp = Input(shape = INPUT_SHAPE)
        
        # add inception block 1
        incep1 = self.Build_Inception_With_Dimension_Reduction(in_layer=inp, f1x1=64, f3x3_red=96, f3x3=128, 
                                                               f5x5_red=16, f5x5=32, fpool=32)
        
        # add inception block 2
        incep2 = self.Build_Inception_With_Dimension_Reduction(in_layer=incep1, f1x1=128, f3x3_red=128, f3x3=192, 
                                                               f5x5_red=32, f5x5=96, fpool=32)

        gap = GlobalAveragePooling2D(data_format='channels_last')(incep2)
        dense = Dense(1024, activation=tf.nn.relu)(gap)
        dropout = Dropout(0.4)(dense)
        out  = Dense(1, activation='sigmoid')(dropout)

        # create model
        model = Model(inputs=visible, outputs=out, name='inception_network')
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        # summarize model
        print(model.summary())
        return model

    
    def InceptionV1(self, INPUT_SHAPE=(299, 299, 3)):
        
        """
        Builds the full incception network with the parameters defined 
        in the Table 1: GoogLeNet incarnation of the Inception architecture 
        in the original paper @ https://arxiv.org/pdf/1409.4842.pdf
        
        Parameters:
            
            INPUT_SHAPE (Optional): the input layer shape. Default Valus is (299, 299, 3).
        
        Return:
            model: the keras model instance.
        
        """