import tensorflow as tf

def orig_conv(features,keep_prob,num_final_neurons):
    """The original model by the Google guy, works for a 2d speech image feature (MFCC or Log Mel)"""
    fingerprint_4d = tf.reshape(features,[-1,features.shape[1],features.shape[2],1])

    conv_1_channels = 64
    conv_2_channels = 64

    weights_1 = tf.Variable(tf.truncated_normal([20,8,1,conv_1_channels],stddev=0.01)) # [height,width,depth,channels]
    bias_1 = tf.Variable(tf.zeros([64]))
    conv_1 = tf.nn.conv2d(fingerprint_4d,weights_1,[1,1,1,1],"SAME") + bias_1
    relu_1 = tf.nn.relu(conv_1)
    dropout_1 = tf.nn.dropout(relu_1,keep_prob)
    maxpool_1 = tf.nn.max_pool(dropout_1,[1,2,2,1],[1,2,2,1],"SAME")

    weights_2 = tf.Variable(tf.truncated_normal([10,4,conv_1_channels,conv_2_channels],stddev=0.01)) # [height,width,first_conv_channels,second_conv_channels]
    bias_2 = tf.Variable(tf.zeros([64]))
    conv_2 = tf.nn.conv2d(maxpool_1,weights_2,[1,1,1,1],"SAME") + bias_2
    relu_2 = tf.nn.relu(conv_2)
    dropout_2 = tf.nn.dropout(relu_2,keep_prob)

    _ , now_height, now_width, _ = dropout_2.get_shape()
    now_height = int(now_height)
    now_width = int(now_width)
    now_flat_elements = now_height * now_width * conv_2_channels

    flat_layer = tf.reshape(dropout_2,[-1,now_flat_elements])

    weights_3 = tf.Variable(tf.truncated_normal([now_flat_elements,num_final_neurons],stddev=0.01))
    bias_3 = tf.Variable(tf.zeros(num_final_neurons))
    final_layer = tf.matmul(flat_layer,weights_3) + bias_3
    return final_layer

def orig_with_extra_fc(features,keep_prob,num_final_neurons):
    fingerprint_4d = tf.reshape(features,[-1,features.shape[1],features.shape[2],1])

    conv_1_channels = 64
    conv_2_channels = 64

    weights_1 = tf.Variable(tf.truncated_normal([20,8,1,conv_1_channels],stddev=0.01)) # [height,width,depth,channels]
    bias_1 = tf.Variable(tf.zeros([64]))
    conv_1 = tf.nn.conv2d(fingerprint_4d,weights_1,[1,1,1,1],"SAME") + bias_1
    relu_1 = tf.nn.relu(conv_1)
    dropout_1 = tf.nn.dropout(relu_1,keep_prob)
    maxpool_1 = tf.nn.max_pool(dropout_1,[1,2,2,1],[1,2,2,1],"SAME")

    weights_2 = tf.Variable(tf.truncated_normal([10,4,conv_1_channels,conv_2_channels],stddev=0.01)) # [height,width,first_conv_channels,second_conv_channels]
    bias_2 = tf.Variable(tf.zeros([64]))
    conv_2 = tf.nn.conv2d(maxpool_1,weights_2,[1,1,1,1],"SAME") + bias_2
    relu_2 = tf.nn.relu(conv_2)
    dropout_2 = tf.nn.dropout(relu_2,keep_prob)

    _ , now_height, now_width, _ = dropout_2.get_shape()
    now_height = int(now_height)
    now_width = int(now_width)
    now_flat_elements = now_height * now_width * conv_2_channels

    flat_layer = tf.reshape(dropout_2,[-1,now_flat_elements])

    weights_3 = tf.Variable(tf.truncated_normal([now_flat_elements,now_flat_elements],stddev=0.01))
    bias_3 = tf.Variable(tf.zeroes(now_flat_elements))
    flat_layer_2 = tf.nn.relu(tf.matmul(flat_layer,weights_3) + bias_3)
    dropout_3 = tf.nn.relu(flat_layer_2,keep_prob)

    weights_4 = tf.Variable(tf.truncated_normal([now_flat_elements,num_final_neurons],stddev=0.01))
    bias_4 = tf.Variable(tf.zeros(num_final_neurons))
    final_layer = tf.matmul(dropout_3,weights_4) + bias_4
    return final_layer


    
