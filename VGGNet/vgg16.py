import tensorflow as tf
from baseNet.network import network



class vgg16(network):
    VGG_MEAN = [103.939, 116.779, 123.68]
    def __init__(self, vgg16_ckpt_path=None, trainable=True, keep_prob=0.5):
        self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.keep_prob = keep_prob

    def inference(self, rgb_x):
         """
        load variable from npy to build the VGG
        :param x: x image [batch, height, width, 3] values scaled [0, 1]
        """
        rgb_scale = rgb_x * 255.0
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert reg.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr_x = tf.concat(axis=3, value=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2]   
        ])
        assert bgr_x.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self._conv_layer(rgb_x, self.data_dict['conv1_1'], name='conv1_1')
        self.conv1_2 = self._conv_layer(self.conv1_1, self.data_dict['conv1_2'], name='conv1_2')
        self.pool1 = self._max_pool(self.conv1_2, name='pool1')

        self.conv2_1 = self._conv_layer(self.pool1, self.data_dict['conv2_1'], name='conv2_1')
        self.conv2_2 = self._conv_layer(self.conv2_1, self.data_dict['conv2_2'], name='conv2_2')
        self.pool2 = self._max_pool(self.conv2_2, name='pool2')

        self.conv3_1 = self._conv_layer(self.pool2, self.data_dict['conv3_1'], name='conv3_1')
        self.conv3_2 = self._conv_layer(self.conv3_1, self.data_dict['conv3_2'], name='conv3_2')
        self.conv3_3 = self._conv_layer(self.conv3_2, self.data_dict['conv3_3'], name='conv3_3')
        self.pool3 = self._max_pool(self.conv3_3, name='pool3')

        self.conv4_1 = self._conv_layer(self.pool3, self.data_dict['conv4_1'], name='conv4_1')
        self.conv4_2 = self._conv_layer(self.conv4_1, self.data_dict['conv4_2'], name='conv4_2')
        self.conv4_3 = self._conv_layer(self.conv4_2, self.data_dict['conv4_3'], name='conv4_3')
        self.pool4 = self._max_pool(self.conv4_3, name='pool4')

        self.conv5_1 = self._conv_layer(self.pool4, self.data_dict['conv5_1'], name='conv5_1')
        self.conv5_2 = self._conv_layer(self.conv5_1, self.data_dict['conv5_2'], name='conv5_2')
        self.conv5_3 = self._conv_layer(self.conv5_2, self.data_dict['conv5_3'], name='conv5_3')
        self.pool5 = self._max_pool(self.conv5_3, name='pool5')

        self.fc6 = self._fc_layer(self.pool5, self.data_dict['fc6'], name='fc6')
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)
        if self.trainable:
            self.relu6 = self._dropout_layer(self.relu6, name='fc6_drop', keep_prob=self.keep_prob)

        self.fc7 = self._fc_layer(self.relu6, self.data_dict['fc7'], name='fc7')
        self.relu7 = tf.nn.relu(self.fc7)
        if self.trainable:
            self.relu7 = self._dropout_layer(self.relu7, name='fc7_drop', keep_prob=self.keep_prob)

        self.fc8 = self._fc_layer(self.relu7, name='fc8')

        self.prob = tf.nn.softmax(self.fc8, name='prob')
        
        pass
