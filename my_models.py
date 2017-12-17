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
    print("Flat Layer 1 shape {}".format(now_flat_elements))

    weights_3 = tf.Variable(tf.truncated_normal([now_flat_elements,2048],stddev=0.01))
    bias_3 = tf.Variable(tf.zeros(2048))
    flat_layer_2 = tf.nn.relu(tf.matmul(flat_layer,weights_3) + bias_3)
    dropout_3 = tf.nn.dropout(flat_layer_2,keep_prob)

    weights_4 = tf.Variable(tf.truncated_normal([2048,num_final_neurons],stddev=0.01))
    bias_4 = tf.Variable(tf.zeros(num_final_neurons))
    final_layer = tf.matmul(dropout_3,weights_4) + bias_4
    return final_layer

def vggnet(features,keep_prob,num_final_neurons):
    fingerprint_4d = tf.reshape(features,[-1,features.shape[1],features.shape[2],1])

    fc_1_neurons = 512
    fc_2_neurons = 256

    def make_vgg_conv_layer(in_layer,in_channels,out_channels,name="conv_layer",maxpool=False):
        with tf.name_scope(name,"vgg_conv_layer") as scope:
            relu = tf.contrib.layers.conv2d(in_layer,out_channels,[3,3],[1,1])
            if maxpool:
                return tf.nn.max_pool(relu,[1,2,2,1],[1,2,2,1],"VALID")
            return relu

    def make_vgg_fc_layer(in_layer,in_neurons,out_neurons,keep_prob,name="fc_layer"):
        with tf.name_scope(name,"vgg_fc_layer") as scope:
            relu = tf.contrib.layers.fully_connected(in_layer,out_neurons)
            dropout = tf.nn.dropout(relu,keep_prob)
            return dropout

    # c1 = make_vgg_conv_layer(features,1,64,name="layer_1")
    # c2 = make_vgg_conv_layer(c1,64,64,name="layer_2",maxpool=True)
    # c3 = make_vgg_conv_layer(c2,64,128,name="layer_3")
    # c4 = make_vgg_conv_layer(c3,128,128,name="layer_4",maxpool=True)
    # c5 = make_vgg_conv_layer(c4,128,256,name="layer_5")
    # c6 = make_vgg_conv_layer(c5,256,256,name="layer_6")
    # c7 = make_vgg_conv_layer(c6,256,256,name="layer_7",maxpool=True)
    # c8 = make_vgg_conv_layer(c7,256,512,name="layer_8")
    # c9 = make_vgg_conv_layer(c8,512,512,name="layer_9")
    # c10 = make_vgg_conv_layer(c9,512,512,name="layer_10",maxpool=True)
    # c11 = make_vgg_conv_layer(c10,512,512,name="layer_11")
    # c12 = make_vgg_conv_layer(c11,512,512,name="layer_12")
    # c_last = make_vgg_conv_layer(c12,512,512,name="layer_13",maxpool=True)

    c1 = make_vgg_conv_layer(fingerprint_4d,1,64,name="layer_1",maxpool=True)
    c2 = make_vgg_conv_layer(c1,64,128,name="layer_2",maxpool=True)
    c3 = make_vgg_conv_layer(c2,128,256,name="layer_3")
    c4 = make_vgg_conv_layer(c3,256,256,name="layer_4",maxpool=True)
    # c5 = make_vgg_conv_layer(c4,256,512,name="layer_5")
    # c6 = make_vgg_conv_layer(c5,512,512,name="layer_6",maxpool=True)
    # c7 = make_vgg_conv_layer(c6,512,512,name="layer_7")
    c_last = make_vgg_conv_layer(c4,512,512,name="layer_8",maxpool=True)


    _, h, w, c = c_last.get_shape()

    flattened_conv = tf.reshape(c_last,[-1,h*w*c])

    print("Flattened Conv Shape {}".format(h*w*c))

    fc_1_out = make_vgg_fc_layer(flattened_conv,h*w*c,fc_1_neurons,keep_prob,name="fc_1")
    fc_2_out = make_vgg_fc_layer(fc_1_out,fc_1_neurons,fc_2_neurons,keep_prob,name="fc_2")
    final_layer = make_vgg_fc_layer(fc_2_out,fc_2_neurons,num_final_neurons,keep_prob,name="final_layer")

    return final_layer


