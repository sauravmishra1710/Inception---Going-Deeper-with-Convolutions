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
    
    def __conv2d_bn(self, inp, filters, kernel_size, padding='same', strides=(1, 1)):
        
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
        conv1x1 = self.__conv2d_bn(inp=in_layer, filters=f1x1, kernel_size=(1,1), padding='same')
        
        # 3x3 conv
        conv3x3 = self.__conv2d_bn(inp=in_layer, filters=f3x3, kernel_size=(3,3), padding='same')
        
        # 5x5 conv
        conv5x5 = self.__conv2d_bn(inp=in_layer, filters=f5x5, kernel_size=(5,5), padding='same')
        
        # 3x3 max pooling
        pool3x3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(in_layer)
        
        # concatenate the convolution and the pooling layers,
        out_tensor = concatenate([conv1x1, conv3x3, conv5x5, pool3x3], axis = -1)
        
        # create model
        model = Model(inputs = in_layer, outputs = out_tensor, name = "Naive Inception Block")
        
        return model
    
    def __Inception_With_Dimension_Reduction(self, in_layer, f1x1, f3x3_red, f3x3, f5x5_red, f5x5, fpool):
        
        """
        Builds the incception module where convolution on an input is performed
        with 3 different sizes of filters (1x1, 3x3, 5x5) with dimension reduction. 
        Additionally, max pooling is also performed. The outputs are concatenated 
        and sent to the next inception module.
        
        1×1 convolutions are used to compute reductions before the expensive 
        3×3 and 5×5 convolutions. Besides being used as reductions, they also 
        include the use of rectified linear activation which makes them dual-purpose.
        
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
        
        
        # #1×1 conv
        conv1x1 = self.__conv2d_bn(inp=in_layer, filters=f1x1, kernel_size=(1,1), padding='same')

        # 3x3 reduce (#3×3reduce) and #3x3 conv
        conv3x3_red = self.__conv2d_bn(inp=in_layer, filters=f3x3_red, kernel_size=(1,1), padding='same')
        conv3x3 = self.__conv2d_bn(inp=conv3x3_red, filters=f3x3, kernel_size=(3,3), padding='same')

        # 5x5 reduce (#5x5reduce) and #5x5 conv
        conv5x5_red = self.__conv2d_bn(inp=in_layer, filters=f5x5_red, kernel_size=(1,1), padding='same')
        conv5x5 = self.__conv2d_bn(inp=conv5x5_red, filters=f5x5, kernel_size=(5,5), padding='same')

        # 3x3 max pooling and 1x1 projection layer - poolproj
        pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(in_layer)
        pool = self.__conv2d_bn(inp=pool, filters=fpool, kernel_size=(1,1), padding='same')

        # concatenate the convolutional layers , poling layer to be passed
        # onto the next layers.
        out_layer = concatenate([conv1x1, conv3x3, conv5x5, pool], axis=-1)
        
        return out_layer
    
    def __IncepAuxiliaryClassifierModule(self, inp_tensor):
        
        """
        Builds the inception auxiliary classifier. 
        GoogLeNet introduces two auxiliary losses before the 
        actual loss and makes the gradients flow backward more sensible. 
        These gradients travel a shorter path and help initial layers converge faster.
        
        These classifiers take the form of smaller convolutional networks put 
        on top of the output of the Inception (4a) and (4d) modules. 
        During training, their loss gets added to the total loss of the
        network with a discount weight (the losses of the auxiliary classifiers were weighted by 0.3). At
        inference time, these auxiliary networks are discarded.
        
        The auxiliary classifier architecture is as follows - 
            1. An average pooling layer with 5×5 filter size and stride 3.
            2. A 1×1 convolution with 128 filters for dimension reduction and rectified linear activation.
            3. A fully connected layer with 1024 units and rectified linear activation.
            4. A dropout layer with 70% ratio of dropped outputs.
            5. A linear layer with softmax loss as the classifier (predicting the same 1000 classes as the
               main classifier, but removed at inference time).
            
        Ref: Section 5 in the paper @ https://arxiv.org/pdf/1409.4842.pdf 
        
        """
        
        aux = AveragePooling2D(pool_size=(5, 5), strides=3)(inp_tensor)
        aux = Conv2D(128, 1, padding='same', activation=tf.nn.relu)(aux)
        aux = Flatten()(aux)
        aux = Dense(1024, activation=tf.nn.relu)(aux)
        aux = Dropout(0.7)(aux)
        aux = Dense(10, activation=tf.nn.softmax)(aux)
        
        return aux
    
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
        incep1 = self.__Inception_With_Dimension_Reduction(in_layer=inp, f1x1=64, f3x3_red=96, f3x3=128, 
                                                               f5x5_red=16, f5x5=32, fpool=32)
        
        # add inception block 2
        incep2 = self.__Inception_With_Dimension_Reduction(in_layer=incep1, f1x1=128, f3x3_red=128, f3x3=192, 
                                                               f5x5_red=32, f5x5=96, fpool=32)

        gap = GlobalAveragePooling2D(data_format='channels_last')(incep2)
        dense = Dense(1024, activation=tf.nn.relu)(gap)
        dropout = Dropout(0.4)(dense)
        out  = Dense(1, activation='sigmoid')(dropout)

        # create model
        model = Model(inputs=inp, outputs=out, name='inception_network')
        
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
            
            INPUT_SHAPE (Optional): the input layer shape. Default Value is (299, 299, 3).
        
        Return:
            model: the keras model instance.
        
        """