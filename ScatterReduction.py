import os, time, shutil, zipfile
import tensorflow as tf
import numpy as np
from tensorflow.core.protobuf import config_pb2

from random import shuffle
from collections import defaultdict
from numba import cuda

os.environ["CUDA_VISIBLE_DEVICES"]="1"
'''
Folder를 만드는 함수
'''
def mkdir_(path_):
    if not os.path.isdir(path_):
        os.mkdir(path_)
    return
'''
영상을 불러들이는 함수
'''
def imgload(path):
    img = np.fromfile(path,dtype='int16',sep="")
    return img
'''
영상을 normalize 하는 함수
'''
def MinMaxNorm(img,v_max,v_min,shape):
    img = img*1.0
    img = (img-v_min)/(v_max-v_min)
    img = img.reshape(shape)
    return img, v_max, v_min
'''
영상을 denormalize 하는 함수
'''
def MinMaxDenorm(img,v_max,v_min):
    img = img*(v_max-v_min)+v_min
    img = np.int16(img)
    return img
'''
영상을 저장하는 함수
'''
def imgsave(path,img):
    img.astype('int16').tofile(path)
    return
'''
Training에 사용되는 patch를 추출하는 함수
'''
def patch_ext(img,size=64,stride=48):
    x_lim = (img.shape[0]-size)//stride
    y_lim = (img.shape[1]-size)//stride
    patch = []
    for i in range(1,x_lim):
        for j in range(1,y_lim):
            patch.append(img[i*stride:i*stride+size,j*stride:j*stride+size])
    return patch
'''
Network 구성 함수
'''
class Network(object):
    def __init__(self,dtype=tf.float32):
        self.dtype = dtype
        self._counts = defaultdict(lambda:0)
    def _activate(self,x,activation='Linear'):
        if activation == "Linear":
            return x
        func = {
            "ReLU": tf.nn.relu,
            "LReLU":tf.nn.leaky_relu
        }
        return func[activation](x)
    def conv2d(self,x,shape,activation,padding='SAME'):
        in_ch = x.shape[3].value
        kw,kh = shape[0], shape[1]
        out_ch = shape[2]
        idx = self._counts['conv']
        self._counts['conv'] += 1
        with tf.variable_scope('conv'+str(idx)):
            kernel = tf.get_variable('weights',[kw,kh,in_ch,out_ch],dtype=self.dtype,initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2/(kw*kh*in_ch))))
            bias = tf.get_variable('bias',[out_ch],dtype=self.dtype,initializer=tf.constant_initializer(0))
            
            x = tf.nn.conv2d(x,kernel,strides=[1,1,1,1],padding=padding,data_format='NHWC')
            x = tf.nn.bias_add(x,bias)
            x = self._activate(x,activation)
            return x
    def skip_connect(self,x1,x2,func='addition'):
        connect = {
            'concat':lambda x,y: tf.concat([x,y],3),
            'addition': tf.add
        }
        return connect[func](x1,x2)
    def loss_cal(self,x,y,loss='MSE',epsilon=1e-3):
        loss_ ={
            'MSE': lambda x,y: tf.square(tf.subtract(x,y)),
            'MAE': lambda x,y: tf.abs(tf.subtract(x,y)),
            'CHA': lambda x,y: tf.sqrt(tf.square(tf.subtract(x,y))+epsilon**2)
        }
        return tf.reduce_mean(loss_[loss](x,y))

'''
Main 함수

Error, weights, data 등 여러 파일 저장 및 불러오기

Network 학습 진행
'''
def main():
    mkdir_('./tmp/')
    mkdir_('./weights/')
    mkdir_('./data/')
    zipfile.ZipFile('data.zip').extractall('./data')
    
    width, height = 500, 500
    
    '''
    Batch  : 한번에 training 할 영상 개수   []
    lr     : 학습에 사용될 learning rate 
    k      : convolution kernel의 사이즈
    block  : network model에 사용되는 block의 개수
    ch     : 각 layer의 channel 개수
    patch  : 하나의 큰 영상에서 여러 작은 조각 영상을 가져올때 사용되는 조각의 크기
    Epo    : 학습에 사용된 data를 몇 번 학습할지 설정
    '''
    batch = 32
    lr = 1e-4
    k = 3
    block = 3
    ch = 64
    patch = 64
    Epo = 200
    
    # MSE, MAE, CHA 선택 가능
    loss_type = 'MSE'
    
    NumImg = len(os.listdir('./data'))
    NumValid = int(0.2*NumImg//2)
    NumTrain = int(0.8*NumImg//2)
    
    tot, scat = [], []
    for i in range(NumImg//2):
        img = imgload('./data/input%4.4d.raw' % i)
        lab = imgload('./data/label%4.4d.raw' % i)
        
        v_max = np.max(img)
        v_min = np.min(img)
        
        img,_,_ = MinMaxNorm(img,v_max,v_min,[width,height])
        lab,_,_ = MinMaxNorm(lab,v_max,v_min,[width,height])
        
        tot.extend(patch_ext(img,patch))
        scat.extend(patch_ext(lab,patch))
    
    NumTrain = int(0.8*np.shape(tot)[0])
    NumValid = int(0.2*np.shape(tot)[0])
    idx_train, idx_valid = [], []
    
    for i in range(NumTrain//batch):
        idx_train.append(i)
    for i in range(NumValid//batch):
        idx_valid.append(NumTrain//batch+i)
    
    x = tf.placeholder(tf.float32,shape=[batch,patch,patch])
    y = tf.placeholder(tf.float32,shape=[batch,patch,patch])
    
    x_ = tf.reshape(x,[batch,patch,patch,1])
    y_ = tf.reshape(y,[batch,patch,patch,1])
    
    ######### Network model #########

    net = Network()
    tensor = net.conv2d(x_,[k,k,ch],'LReLU')

    tensor1 = tensor

    for i in range(block):
        tensor_ = net.conv2d(tensor,[k,k,ch],'LReLU')
        tensor = net.conv2d(tensor_,[k,k,ch],'LReLU')
        tensor = net.conv2d(tensor,[k,k,ch],'LReLU')
        tensor = net.skip_connect(tensor,tensor_)

    tensor = net.skip_connect(tensor,tensor1)

    output = net.conv2d(tensor,[k,k,1],'Linear')

    #################################

    tot_loss = net.loss_cal(output,y_,loss_type)

    opt = tf.train.AdamOptimizer(lr).minimize(tot_loss)
    saver = tf.train.Saver(tf.global_variables())
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
    
        start = time.time()
        mkdir_('./weights/error')
        for epoch in range(Epo):
            shuffle(idx_train)
            shuffle(idx_valid)
            E_t = open('./weights/error/E_t' + str(epoch) + '.txt','w')
            for idx in range(NumTrain//batch):
                Input = tot[idx_train[idx]*batch:idx_train[idx]*batch+batch][:][:]
                Label = scat[idx_train[idx]*batch:idx_train[idx]*batch+batch][:][:]
                _,l = sess.run([opt,tot_loss],feed_dict={x:Input,y:Label},options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
                e = "%0.8f\n" % l
                E_t.write(e)
                end = time.time()
                print("[Epoch %2d (%6d/%d)] loss %.7f\t %.2f sec" % (epoch,idx,NumTrain//batch,l,end-start))
            E_t.close()
            
            E_v = open('./weights/error/E_v' + str(epoch) + '.txt','w')
            for idx in range(NumValid//batch):
                Input = tot[idx_valid[idx]*batch:idx_valid[idx]*batch+batch][:][:]
                Label = scat[idx_valid[idx]*batch:idx_valid[idx]*batch+batch][:][:]
                l = sess.run(tot_loss,feed_dict={x:Input,y:Label})
                e = "%0.8f\n" % l
                E_v.write(e)
                end = time.time()
                print("[Epoch %2d (%6d/%d)] loss %.7f\t %.2f sec" % (epoch,idx,NumValid//batch,l,end-start))
            E_v.close()
            saver.save(sess,'./tmp/weights.ckpt')
            if os.path.isdir('./weights/' + str(epoch+1)):
                shutil.rmtree('./weights/' + str(epoch+1))
            shutil.copytree('./tmp','./weights/' + str(epoch+1))
        sess.close()
    tf.reset_default_graph()
    cuda.select_device(0)
    cuda.close()
if __name__ == '__main__':
    main()

