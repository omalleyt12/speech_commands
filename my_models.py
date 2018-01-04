import tensorflow as tf
import tensorflow.contrib.slim as slim
import custom_bn as bn
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





def conv2d(x,channels,kernel_size,is_training,strides=[1,1],padding="SAME",mp=None,bn=True):
    """Make sure to update training ops when using this, can run something like:

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    sess.run(...,update_ops)

    Also make sure to use the is_training placeholder
    """
    if bn:
        c = tf.contrib.layers.conv2d(x,channels,kernel_size,strides,padding=padding,activation_fn=None)
        c = tf.contrib.slim.batch_norm(c,is_training=is_training,decay=0.9)
        c = tf.nn.relu(c)
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

def overdrive_bn(features,keep_prob,num_final_neurons,is_training):
    """This is my best so far"""
    fingerprint_4d = tf.reshape(features,[-1,features.shape[1],features.shape[2],1])

    c = conv2d(fingerprint_4d,64,[7,3],is_training,mp=[1,3])
    c = conv2d(c,128,[1,7],is_training,mp=[1,4])

    c = conv2d(c,256,[1,10],is_training,padding="VALID")
    c = conv2d(c,512,[7,1],is_training,mp=[c.shape[1],1])

    c = tf.contrib.layers.flatten(c)

    fc = tf.contrib.layers.fully_connected(c,128)

    final_layer = tf.contrib.layers.fully_connected(fc,num_final_neurons,activation_fn=None)
    print(c.shape)
    return final_layer

def overdrive_full_bn( fffeatures,keep_prob,num_final_neurons,is_training):
    fingerprint_4d = tf.reshape(features,[-1,features.shape[1],features.shape[2],1])

    c = conv2d(fingerprint_4d,64,[7,3],is_training,mp=[1,3])
    c = conv2d(c,128,[1,7],is_training,mp=[1,4])

    c = conv2d(c,256,[1,10],is_training,padding="VALID")
    c = conv2d(c,512,[7,1],is_training,mp=[c.shape[1],1])

    c = tf.contrib.layers.flatten(c)

    fc = tf.contrib.slim.fully_connected(c,256)
    fc = tf.contrib.slim.batch_norm(fc,is_training=is_training,decay=0.9)

    final_layer = tf.contrib.layers.fully_connected(fc,num_final_neurons,activation_fn=None)
    print(c.shape)
    return final_layer, fc

def overdrive_res(features,keep_prob,num_final_neurons,is_training):
    fingerprint_4d = tf.reshape(features,[-1,features.shape[1],features.shape[2],1])

    def res_conv(input_c,channels,kernel,is_training,mp=None):
        x = slim.conv2d(input_c,channels,kernel,activation_fn=None)
        c = slim.batch_norm(x,is_training=is_training,decay=0.9)
        c = tf.nn.relu(c)
        c = slim.conv2d(c,channels,kernel,activation_fn=None)
        res = x + c
        res = slim.batch_norm(res,is_training=is_training,decay=0.9)
        res = tf.nn.relu(res)
        if mp is not None:
            return tf.nn.max_pool(res,[1,mp[0],mp[1],1],[1,mp[0],mp[1],1],"VALID")
        else:
            return res

    c = res_conv(fingerprint_4d,64,[7,3],is_training,mp=[1,3])
    c = res_conv(c,128,[1,7],is_training,mp=[1,4])

    c = slim.conv2d(c,)


def slim_conv2d(input_channel,channels,kernel_size,is_training,padding="SAME",mp=None,l2_penalty=0.0005):
    c = tf.contrib.slim.conv2d(input_channel,channels,kernel_size,activation_fn=None,padding=padding,weights_regularizer=tf.contrib.slim.l2_regularizer(l2_penalty))
    c = tf.contrib.slim.batch_norm(c,is_training=is_training,decay=0.9)
    c = tf.nn.relu(c)
    if mp is not None:
        return tf.nn.max_pool(c,[1,mp[0],mp[1],1],[1,mp[0],mp[1],1],"VALID")
    else:
        return c

def newdrive(features,keep_prob,num_final_neurons,is_training):
    f = tf.reshape(features,[-1,features.shape[1],features.shape[2],1])
    print(f.shape)

    c = slim.conv2d(f,16,[7,1],activation_fn=None)
    c = slim.batch_norm(c,is_training=is_training,decay=0.9)
    c = tf.nn.relu(c)
    print(c.shape)

    c = slim.separable_conv2d(c,32,[1,7],1,activation_fn=None)
    c = slim.batch_norm(c,is_training=is_training,decay=0.9)
    c = tf.nn.relu(c)
    c = tf.nn.max_pool(c,[1,1,3,1],[1,1,3,1],"VALID")
    print(c.shape)

    c = slim.separable_conv2d(c,64,[1,7],1,activation_fn=None)
    c = slim.batch_norm(c,is_training=is_training,decay=0.9)
    c = tf.nn.relu(c)
    c = tf.nn.max_pool(c,[1,1,4,1],[1,1,4,1],"VALID")
    print(c.shape)

    c = slim.separable_conv2d(c,128,[1,10],1,activation_fn=None,padding="VALID")
    c = slim.batch_norm(c,is_training=is_training,decay=0.9)
    c = tf.nn.relu(c)
    print(c.shape)

    c = slim.separable_conv2d(c,256,[7,1],1,activation_fn=None)
    c = slim.batch_norm(c,is_training=is_training,decay=0.9)
    c = tf.nn.relu(c)
    c = tf.nn.max_pool(c,[1,c.shape[1],1,1],[1,c.shape[1],1,1],"VALID")
    c = tf.contrib.layers.flatten(c)
    print(c.shape)

    fc = slim.fully_connected(c,128,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.001))
    fc = slim.batch_norm(fc,is_training=is_training,decay=0.9)
    fc = tf.nn.relu(fc)
    print(fc.shape)

    final_layer = slim.fully_connected(fc,num_final_neurons,activation_fn=None)
    print(final_layer.shape)

    return final_layer, fc






def okconv(features,keep_prob,num_final_neurons,num_full_final_neurons,is_training):
    """More convs for a 40 log mel spectrogram"""
    fingerprint_4d = tf.reshape(features,[-1,features.shape[1],features.shape[2],1])

    c = conv2d(fingerprint_4d,64,[3,3],is_training)
    c = conv2d(c,64,[3,3],is_training)
    c = conv2d(c,64,[3,3],is_training,mp=[1,2])

    c = conv2d(c,128,[3,3],is_training)
    c = conv2d(c,128,[3,3],is_training,mp=[1,2])

    c = conv2d(c,512,[1,10],is_training,padding="VALID")
    c = conv2d(c,512,[1,1],is_training)

    c = conv2d(c,1024,[7,1],is_training)
    c = conv2d(c,1024,[7,1],is_training)

    mp = tf.nn.max_pool(c,[1,c.shape[1],1,1],[1,c.shape[1],1,1],"VALID")
    ap = tf.nn.avg_pool(c,[1,c.shape[1],1,1],[1,c.shape[1],1,1],"VALID")

    flat_conv = tf.concat([
        tf.contrib.layers.flatten(mp),
        tf.contrib.layers.flatten(ap)
    ],axis=1)

    fc = tf.contrib.slim.fully_connected(flat_conv,1024)
    fc = tf.contrib.slim.batch_norm(fc,is_training=is_training,decay=0.95)

    full_fc = tf.contrib.slim.fully_connected(flat_conv,1024)
    full_fc = tf.contrib.slim.batch_norm(full_fc,is_training=is_training,decay=0.95)

    final_layer = tf.contrib.layers.fully_connected(fc,num_final_neurons,activation_fn=None)

    full_final_layer = tf.contrib.layers.fully_connected(full_fc,num_full_final_neurons,activation_fn=None)

    return final_layer, full_final_layer, fc


def full_resdilate(features,keep_prob,num_final_neurons,is_training):
    conv_keep_prob = tf.cond(is_training, lambda: 0.8, lambda: 1.0)
    def cool_layer_bn(input_layer,channels,scope,is_training):
        """
        This allows x to pass freely through the dilation convolutions
        Based on ideas from WaveNet and "Identity Mappings in Deep Residual Networks"
        """
        x = tf.contrib.slim.conv2d(input_layer,channels,[9,1],activation_fn=None)
        c = x
        for dilation in [1,1]:
            c = tf.contrib.slim.batch_norm(c,is_training=is_training,decay=0.9)
            c = tf.nn.relu(c)
            c = tf.contrib.slim.conv2d(c,channels,[9,1],rate=[dilation,1],activation_fn=None,weights_regularizer=slim.l2_regularizer(0.0005))
        res = x + c
        res = tf.contrib.slim.batch_norm(res,is_training=is_training,decay=0.9)
        res = tf.nn.relu(res)
        res = tf.nn.dropout(res,conv_keep_prob)
        mp = tf.nn.max_pool(res,[1,3,1,1],[1,3,1,1],"VALID")
        return mp

    c = tf.reshape(features,[-1,features.shape[1],1,1])
    c = tf.contrib.slim.batch_norm(c,is_training=is_training,decay=0.9)
    for channels in [8,16,32,64,126,256]:
        c = cool_layer_bn(c,channels,str(channels),is_training)
        print(c.shape)
    mp = tf.nn.max_pool(c,[1,c.shape[1],1,1],[1,c.shape[1],1,1],"VALID")
    ap = tf.nn.avg_pool(c,[1,c.shape[1],1,1],[1,c.shape[1],1,1],"VALID")
    flat_conv = tf.concat([
        tf.contrib.layers.flatten(mp),
        tf.contrib.layers.flatten(ap)
    ],axis=1)
    flat_conv = tf.nn.dropout(flat_conv,keep_prob)
    print(flat_conv.shape)

    fc = tf.contrib.slim.fully_connected(flat_conv,512,activation_fn=None,weights_regularizer=tf.contrib.slim.l2_regularizer(0.0005))
    fc = tf.contrib.slim.batch_norm(fc,is_training=is_training,decay=0.9)
    fc = tf.nn.relu(fc)
    fc = tf.nn.dropout(fc,keep_prob)
    print(fc.shape)

    final_layer = tf.contrib.layers.fully_connected(fc,num_final_neurons,activation_fn=None)
    print(final_layer.shape)
    return final_layer, fc


