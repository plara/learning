import sys
sys.path.append("..")
sys.path.append("../..")

import tensorflow as tf
import tensorflow.contrib.slim as slim # only for dropout - need different behaviour while training and
#testing
from object_detection.meta_architectures import faster_rcnn_meta_arch
import collections

#define names similar to keras from tf.keras so that you can copy paste the model code
Input = tf.keras.layers.Input
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Dense = tf.keras.layers.Dense
Model = tf.keras.models.Model
K = tf.keras.backend

class FasterRCNNVGG16FeatureExtractor(faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
    """
    VGG-16 Faster RCNN Feature Extractor
    """
    def __init__(self,
                 is_training,
                 first_stage_features_stride,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0):
        super(FasterRCNNVGG16FeatureExtractor, self).__init__(
            is_training,first_stage_features_stride,batch_norm_trainable,reuse_weights,weight_decay
        )

    def preprocess(self, resized_inputs):
        """
        Faster R-CNN VGG-16 preprocessing
         mean subtraction

        :param resized_inputs: a [batch,height_in,width_in,channels] float32 Tensor representing
        a batch of images with values between 0 and 255.0
        :return:  preprocessed inputs: A [batch,height_out,width_out,channels] float32 Tensor
            representing a batch of images
        """
        channel_means = [123.68,116.779,103.939]
        return resized_inputs - [[channel_means]]

    def _extract_proposal_features(self,preprocessed_inputs,scope):
        """
        extracts first stage RPN features.
        :param preprocessed_inputs: preprocessed [batch,height,width,channels] float32 tensor
        :param scope: A scope name (unused)
        :return: rpn_feature_map : a tensor with shape [batch,height,width,depth]

        NOTE: Make sure the naming are similar to keras, else creates problem while loading weights
        """

        #Block 1
        tensor_shape = preprocessed_inputs.get_shape().as_list()
        input = Input(shape=(tensor_shape[1],tensor_shape[2],tensor_shape[3]),tensor=preprocessed_inputs,name="input")
        x = Conv2D(64,(3,3),activation='relu',padding='same',name='block1_conv1')(input)
        # x = Conv2D(64,(3,3),activation='relu',padding='same',name='block1_conv2')(x)
        x = MaxPooling2D((2,2),strides=(2,2),name='block1_pool')(x)

        #Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        #Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        #Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        #Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        model = Model(input,x)

        activations = collections.OrderedDict()
        for layer in model.layers:
            activations[layer.name] = layer.output

        return activations["block5_pool"],activations

    def _extract_box_classifier_features(self, proposal_feature_maps, scope):
        """
        Extract second stage box classifier features
        :param proposal_feature_maps: A 4-D float tensor with shape:
        [batch_size*self.max_num_proposals,crop_height, crop_width,depth]
        representing the feature map cropped to each proposal.
        :param scope: A scope name (unused)
        :return: proposal_classifier_features: A 4-D tensor with shape
        [batch_size*self.max_num_proposals,height,width,depth]
        representing box classifier features for each proposal.


        use tf.slim for dropout because you need different behaviour while training and testing
        """
        x = Dense(512,activation='relu',name='fc1')(proposal_feature_maps)
        x = slim.dropout(x,0.5,scope="Dropout_1",is_training=self._is_training)
        x = Dense(256,activation='relu',name='fc2')(x)
        proposal_classifier_features = slim.dropout(x,0.5,scope="Dropout_2",is_training=self._is_training)

        return proposal_classifier_features

























