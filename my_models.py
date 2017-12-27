import tensorflow as tf
from keras.layers import GlobalMaxPool2D

def vggnet(features,keep_prob,num_final_neurons):
    fingerprint_4d = tf.reshape(features,[-1,features.shape[1],features.shape[2],1])

    fc_1_neurons = 3000
    fc_2_neurons = 1500

    print("Using {} neurons in FC Layers".format(fc_2_neurons))

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

    # This worked for my best-performing model yet
    c1 = make_vgg_conv_layer(fingerprint_4d,1,64,name="layer_1",maxpool=True)
    c2 = make_vgg_conv_layer(c1,64,128,name="layer_2",maxpool=True)
    c3 = make_vgg_conv_layer(c2,128,256,name="layer_3")
    c4 = make_vgg_conv_layer(c3,256,256,name="layer_4",maxpool=True)
    # c5 = make_vgg_conv_layer(c4,256,512,name="layer_5")
    # c6 = make_vgg_conv_layer(c5,512,512,name="layer_6",maxpool=True)
    # c7 = make_vgg_conv_layer(c6,512,512,name="layer_7")
    c_last = make_vgg_conv_layer(c4,512,512,name="layer_8",maxpool=True)

    print(c_last.shape)

    # c1 = make_vgg_conv_layer(fingerprint_4d,1,64,name="layer_1",maxpool=True)
    # c2 = make_vgg_conv_layer(c1,64,128,name="layer_2",maxpool=True)
    # c3 = make_vgg_conv_layer(c2,128,256,name="layer_3")
    # c4 = make_vgg_conv_layer(c3,256,256,name="layer_4",maxpool=True)
    # c5 = make_vgg_conv_layer(c4,256,512,name="layer_5")
    # c6 = make_vgg_conv_layer(c5,512,512,name="layer_6",maxpool=True)
    # c7 = make_vgg_conv_layer(c6,512,512,name="layer_7")
    # c_last = make_vgg_conv_layer(c7,512,512,name="layer_8",maxpool=True)


    _, h, w, c = c_last.get_shape()

    flattened_conv = tf.reshape(c_last,[-1,h*w*c])

    print("Flattened Conv Height {}".format(h))
    print("Flattened Conv Width {}".format(w))

    print("Flattened Conv Shape {}".format(h*w*c))

    fc_1_out = make_vgg_fc_layer(flattened_conv,h*w*c,fc_1_neurons,keep_prob,name="fc_1")
    fc_2_out = make_vgg_fc_layer(fc_1_out,fc_1_neurons,fc_2_neurons,keep_prob,name="fc_2")
    final_layer = make_vgg_fc_layer(fc_2_out,fc_2_neurons,num_final_neurons,keep_prob,name="final_layer")

    return final_layer

# I'm now using this as the Ferrari version of what I just tried
def drive_conv_log_mel(features,keep_prob,num_final_neurons,is_training):
    """Let's assume we have a 128 bin spectrogram here"""
    fingerprint_4d = tf.reshape(features,[-1,features.shape[1],features.shape[2],1])

    c = conv2d(fingerprint_4d,128,[7,7],is_training,mp=[1,3])
    c = conv2d(c,192,[1,7],is_training,mp=[1,2])
    c = conv2d(c,256,[1,7],is_training,mp=[1,2])
    print(c.shape)

    c = conv2d(c,512,[1,c.shape[2]],is_training,padding="VALID")
    c = conv2d(c,1024,[1,1],is_training,mp=[c.shape[1],1])
    c = tf.contrib.layers.flatten(c)
    c = tf.nn.dropout(c,keep_prob)
    print(c.shape)

    fc1 = tf.contrib.layers.fully_connected(c,2048)
    fc1 = tf.nn.dropout(fc1,keep_prob)

    fc2 = tf.contrib.layers.fully_connected(fc1,1024)
    fc2 = tf.nn.dropout(fc2,keep_prob)

    final_layer = tf.contrib.layers.fully_connected(fc2,num_final_neurons)

    print(final_layer.shape)

    return final_layer





def conv2d(x,channels,kernel_size,is_training,strides=[1,1],padding="SAME",mp=None,bn=False):
    """Make sure to update training ops when using this, can run something like:

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    sess.run(...,update_ops)

    Also make sure to use the is_training placeholder
    """
    if bn:
        c = tf.contrib.layers.conv2d(x,channels,kernel_size,strides,normalizer_fn=tf.contrib.layers.batch_norm,normalizer_params={"is_training":is_training},padding=padding)
    else:
        c = tf.contrib.layers.conv2d(x,channels,kernel_size,strides,padding=padding)
    if mp is not None:
        return tf.nn.max_pool(c,[1,mp[0],mp[1],1],[1,mp[0],mp[1],1],"VALID")
    else:
        return c

def spec2phone(features,is_training,type=None):
    """These are the various strategies for going from 128 Log Mel Spectrogram => Phone"""
    fingerprint_4d = tf.reshape(features,[-1,features.shape[1],features.shape[2],1])
    if type == "drive_conv": # not "fixed"
        c = conv2d(fingerprint_4d,64,[3,3],is_training,mp=[2,2])
        c = conv2d(c,128,[3,3],is_training,mp=[2,2])
        c = conv2d(c,128,[3,3],is_training,mp=[1,2])

        for channels in [192,256,384,512]:
            c = conv2d(c,channels,[3,1],is_training,mp=[1,2])
        c = conv2d(c,256,[1,1],is_training)

        print(c.shape)
    elif type == "overdrive":
        c = conv2d(fingerprint_4d,64,[7,7],is_training,mp=[1,5])
        c = conv2d(c,128,[1,3],is_training,mp=[1,2])
        print(c.shape)

        c = conv2d(c,256,[1,c.shape[2]],is_training,padding="VALID")
        c = conv2d(c,512,[3,1],is_training,mp=[3,1])
        print(c.shape)


def parts_conv(features,keep_prob,num_final_neurons,is_training):
    c = spec2phone(features,is_training,"overdrive")

def overdrive(features,keep_prob,num_final_neurons,is_training):
    fingerprint_4d = tf.reshape(features,[-1,features.shape[1],features.shape[2],1])

    c = conv2d(fingerprint_4d,64,[7,3],is_training,mp=[1,3])
    c = conv2d(c,128,[1,7],is_training,mp=[1,4])

    c = conv2d(c,256,[1,10],is_training,padding="VALID")
    c = conv2d(c,512,[7,1],is_training,mp=[c.shape[1],1])

    c = tf.contrib.layers.flatten(c)

    d = tf.nn.dropout(c,keep_prob)
    fc = tf.contrib.layers.fully_connected(d,128)
    d2 = tf.nn.dropout(fc,keep_prob)

    final_layer = tf.contrib.layers.fully_connected(d2,num_final_neurons)
    print(c.shape)
    print(d.shape)
    return final_layer

def rnn_overdrive(features,keep_prob,num_final_neurons,is_training):
    fingerprint_4d = tf.reshape(features,[-1,features.shape[1],features.shape[2],1])

    c = conv2d(fingerprint_4d,64,[7,3],is_training,mp=[1,3])
    c = conv2d(c,128,[1,7],is_training,mp=[1,4])

    c = conv2d(c,256,[1,10],is_training,padding="VALID")
    c = conv2d(c,512,[7,1],is_training,mp=[3,1])
    print(c.shape)
    c = tf.reshape(c,[-1,c.shape[1],c.shape[3]])
    print(c.shape)

    lstmcell = tf.contrib.rnn.LSTMCell(500, use_peepholes=True,num_proj=188)
    _, last = tf.nn.dynamic_rnn(cell=lstmcell, inputs=c,
                dtype=tf.float32)
    flow = last[-1]

    print(flow.shape)
    dflow = tf.nn.dropout(flow,keep_prob)

    fc = tf.contrib.layers.fully_connected(dflow,128)
    dfc = tf.nn.dropout(fc,keep_prob)

    final_layer = tf.contrib.layers.fully_connected(dfc,num_final_neurons,activation_fn=None)
    print(final_layer.shape)
    return final_layer


def ttagau_conv(features,keep_prob,num_final_neurons,is_training):
    x = tf.reshape(features,[-1,features.shape[1],1,1])
    for i in range(10):
        channels = int(8*(1.5**i))
        x = conv2d(x,channels,[3,1],is_training,mp=[2,1])

    print(x.shape)
    x = tf.nn.max_pool(x,[1,x.shape[1],1,1],[1,x.shape[1],1,1],"VALID")
    print(x.shape)

    x = tf.contrib.layers.flatten(x)
    x = tf.nn.dropout(x,keep_prob)
    print(x.shape)

    x = tf.contrib.layers.fully_connected(x,150)
    print(x.shape)

    x = tf.nn.dropout(x,keep_prob)

    final_layer = tf.contrib.layers.fully_connected(x,num_final_neurons)

    return final_layer


