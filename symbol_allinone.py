import os, sys
import argparse
import find_mxnet
import mxnet as mx
#import mixup


def ConvFactory(name, data, num_filter, kernel, stride=1, pad=0, use_act=True):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter,
                                 kernel=(kernel, kernel),
                                 stride=(stride, stride),
                                 pad=(pad, pad),
                                 name='conv_{}'.format(name))
    bn = mx.symbol.BatchNorm(data=conv, name='bn_{}'.format(name))
    if use_act == True:
        act = mx.sym.Activation(data=bn, act_type='relu',
                                   name='act_{}'.format(name))
        return act
    else:
        return conv


def get_symbol(is_train=True,batch_size=256,lmk_cnt=21,eye_cnt=3,smoke_cnt=2,pose_cnt=21):
    data = mx.sym.Variable("data")
    label = mx.sym.Variable("softmax_label")
    #sharing net
    conv1 = ConvFactory('conv1', data, 96, 3, 2, 1)    
    conv1_2 = ConvFactory('conv1_2', conv1, 96, 3, 2, 1)
    conv2 = ConvFactory('conv2', conv1_2, 256, 3, 1, 1)
    conv3 = ConvFactory('conv3', conv2, 256, 3, 2, 1)    
    conv4 = ConvFactory('conv4', conv3, 256, 1, 1, 0)    
    conv5 = ConvFactory('conv5', conv4, 256, 1, 1, 0)
    #lmk part
    conv6_1 = ConvFactory('conv6_1', conv5, 42, 1, 1, 0)    
    conv6_2 = ConvFactory('conv6_2', conv6_1, 42, 1, 1, 0)
    conv6_3 = ConvFactory('conv6_3', conv6_2, 42, 1, 1, 0, False)
    conv6_4 = mx.symbol.Pooling(data=conv6_3, global_pool=True,pool_type='avg',kernel=(1,1))
    lmk_out = mx.symbol.Flatten(data=conv6_4)
    #pitch part
    conv7_1 = ConvFactory('conv7_1',conv6_2, 7, 1, 1, 0)
    conv7_2 = ConvFactory('conv7_2',conv7_1, 7, 1, 1, 0)
    conv7_3 = ConvFactory('conv7_3',conv7_2, 7, 1, 1, 0, False)
    conv7_4 = mx.symbol.Pooling(data=conv7_3, global_pool=True,pool_type='avg',kernel=(1,1))
    pitch_out = mx.symbol.Flatten(data=conv7_4)
    #yaw part
    conv8_1 = ConvFactory('conv8_1',conv6_2, 7, 1, 1, 0)
    conv8_2 = ConvFactory('conv8_2',conv8_1, 7 ,1, 1, 0)
    conv8_3 = ConvFactory('conv8_3',conv8_2, 7 ,1, 1, 0, False)
    conv8_4 = mx.symbol.Pooling(data=conv8_3, global_pool=True,pool_type='avg',kernel=(1,1))
    yaw_out = mx.symbol.Flatten(data=conv8_4)
    #roll part
    conv9_1 = ConvFactory('conv9_1',conv6_2, 7, 1, 1, 0)
    conv9_2 = ConvFactory('conv9_2',conv9_1, 7, 1, 1, 0)
    conv9_3 = ConvFactory('conv9_3',conv9_2, 7, 1, 1, 0, False)
    conv9_4 = mx.symbol.Pooling(data=conv9_3, global_pool=True,pool_type='avg',kernel=(1,1))
    roll_out = mx.symbol.Flatten(data=conv9_4)    
    #smoke part 
    conv10_1 = ConvFactory('conv10_1',conv5, 64, 1, 1, 0)
    conv10_2 = ConvFactory('conv10_2',conv10_1, 32, 1, 1, 0)
    conv10_3 = ConvFactory('conv10_3',conv10_2, 2, 1, 1, 0)
    conv10_4 = ConvFactory('conv10_4',conv10_3, 2, 1, 1, 0)
    conv10_5 = mx.symbol.Pooling(data=conv10_4,global_pool=True,pool_type='avg',kernel=(1,1))
    smoke_out = mx.symbol.Flatten(data=conv10_5)
    #eye part
    conv11_1 = ConvFactory('conv11_1',conv5, 64, 1, 1, 0)
    conv11_2 = ConvFactory('conv11_2',conv11_1, 32, 1, 1, 0)
    conv11_3 = ConvFactory('conv11_3',conv11_2, 3, 1, 1, 0)
    conv11_4 = ConvFactory('conv11_4',conv11_3, 3, 1, 1, 0)
    conv11_5 = mx.symbol.Pooling(data=conv11_4,global_pool=True,pool_type='avg',kernel=(1,1))
    eye_out = mx.symbol.Flatten(data=conv11_5)
    #out_sum =mx.symbol.concat(lmk_out,eye_out,smoke_out,pitch_out,yaw_out,roll_out,dim=0)
    
    out_sum=mx.symbol.concat(conv6_4,conv7_4,conv8_4,conv9_4,conv10_5,conv11_5,dim=1)
    out_sum=mx.symbol.Flatten(data=out_sum)


    if is_train:
        offset = 0
        # lmk loss
        label_lmk = mx.symbol.slice_axis(data=label[0], axis=1, begin=offset, end=lmk_cnt*2)
        label_lmk = mx.symbol.Flatten(data=label_lmk)
        #lmk_out = mx.symbol.slice_axis(data = lmk_out, axis=1, begin=offset, end=lmk_cnt*2)
        loss_lmk = mx.sym.LinearRegressionOutput(data=lmk_out, label=label_lmk, grad_scale=1)
        offset=offset+lmk_cnt*2
        
        # eye loss
        label_eye = mx.symbol.slice_axis(data=label[0], axis=1, begin=offset, end=offset+3)
        label_eye = mx.symbol.Flatten(data=label_eye)
        #eye_out = mx.symbol.slice_axis(data = conv10_2, axis=1, begin=offset,end=offset+3)
        loss_eye = mx.symbol.SoftmaxOutput(data=eye_out, label=label_eye)
        offset=offset+eye_cnt
        
        # smoke loss
        label_smoke = mx.symbol.slice_axis(data=label[0],axis=1, begin=offset, end=offset+2)
        label_smoke = mx.symbol.Flatten(data=label_smoke)
        #smoke_out = mx.symbol.slice_axis(data=conv10_2, axis=1, begin=offset, end=offset+2)
        loss_smoke = mx.symbol.SoftmaxOutput(data=smoke_out,label=label_smoke)
        offset=offset+smoke_cnt
        
        #pose pitch
        label_pitch = mx.symbol.slice_axis(data=label[0],axis=1, begin=offset, end=offset+7)
        label_pitch = mx.symbol.Flatten(data=label_pitch)
        #pitch_out = mx.symbol.slice_axis(data=conv10_2,axis=1, begin=offset, end=offset+7)
        loss_pitch = mx.symbol.SoftmaxOutput(data=pitch_out,label=label_pitch)
        offset=offset+7
        
        #pose yaw
        label_yaw = mx.symbol.slice_axis(data=label[0],axis=1, begin=offset, end=offset+7)
        label_yaw = mx.symbol.Flatten(data=label_yaw)
        #yaw_out = mx.symbol.slice_axis(data=conv10_2, axis=1, begin=offset, end=offset+7)
        loss_yaw = mx.symbol.SoftmaxOutput(data=yaw_out, label=label_yaw)
        offset=offset+7

        #pose roll
        label_roll = mx.symbol.slice_axis(data=label[0],axis=1, begin=offset, end=offset+7)
        label_roll = mx.symbol.Flatten(data=label_roll)
        #roll_out = mx.symbol.slice_axis(data=conv10_2,axis=1, begin=offset, end=offset+7)
        loss_roll = mx.symbol.SoftmaxOutput(data=roll_out, label=label_roll)
        offset=offset+7
        
        loss_sum = mx.symbol.Group([loss_lmk,loss_eye,loss_smoke,loss_pitch,loss_yaw,loss_roll])
        return loss_sum

    return out_sum


if __name__ == '__main__':
    data_names = ['data']
    label_names = ['label', 'mask']
    data = mx.symbol.Variable(name=data_names[0])
    labels = [mx.symbol.Variable(name=name) for name in label_names]

    # can not be together
    is_train = False
    symbol = get_symbol(is_train=is_train)
    print symbol.list_arguments()
    symbol.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'symbol_%s.json' % ('train' if is_train else 'test')))

    #a = mx.viz.plot_network(symbol, shape={"data":(1, 1, 72, 72)}, node_attrs={"shape":'rect',"fixedsize":'false'})
    #a.render()
