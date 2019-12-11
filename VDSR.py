'''

This is VDSR code proposed by Jiwon Kim et al.

Code by Seungwoo Ha

'''

from urllib import request
from PIL import Image
from collections import defaultdict
from random import shuffle
import zipfile, os
import tensorflow as tf
import numpy as np

class PreProcessing(object):
    def __init__(self,patch_size=41):
        self.patch_size = patch_size
    
    def load_train_data(self,path,patch_in,patch_gt):
        patch_size = self.patch_size
        img_list = os.listdir(path)
        for i in img_list:
            img = Image.open(path +'/' + i,'r')
            img = img.convert('YCbCr')
            img_gt = np.asarray(img,dtype='uint8')
            img_gt = img_gt.astype('float32')
            
            width, height,_ = np.shape(img)
            w, h = width//patch_size, height//patch_size

            for s in range(3):
                img_in = img.resize((height//(s+2),width//(s+2)),Image.BICUBIC)
                img_in = img_in.resize((height,width),Image.BICUBIC)    
                img_in = np.asarray(img_in,dtype='uint8')
                img_in = img_in.astype('float32')

                for j in range(h):
                    for i in range(w):
                        tmp = img_in[i*patch_size:i*patch_size+patch_size,j*patch_size:j*patch_size+patch_size,0:1]
                        tmp = self.data_aug(tmp)
                        patch_in.extend(tmp)
                        
                        tmp = img_gt[i*patch_size:i*patch_size+patch_size,j*patch_size:j*patch_size+patch_size,0:1]
                        tmp = self.data_aug(tmp)
                        patch_gt.extend(tmp)
        return patch_in, patch_gt
    
    def data_aug(self,image):
        patch = []
        patch.append(image)
        patch.append(np.fliplr(image))
        for i in range(3):
            tmp = np.rot90(image,i+1)
            patch.append(tmp)
            tmp = np.fliplr(tmp)
            patch.append(tmp)
        return patch
    
class Network(object):
    def __init__(self,dtype=tf.float32,activation='ReLU'):
        self.dtype = dtype
        self.activation = activation
        self._counts = defaultdict(lambda : 0)

    def _activate(self,input,activation=None):
        activation = self.activation
        if activation == 'Linear':
            return input
        activate_func = {
            'ReLU'  : tf.nn.relu,
            'LReLU' : tf.nn.leaky_relu
        }
        return activate_func[activation](input)

    def conv2d(self,input,shape,weights,activation=None,padding='SAME'):
        activation = self.activation
        in_ch = input.shape[3].value
        kw,kh = shape[0],shape[1]
        out_ch = shape[2]
        idx = self._counts['conv2d']
        self._counts['conv2d'] += 1
        with tf.compat.v1.variable_scope('conv2d' + str(idx)):
            kernel = tf.compat.v1.get_variable('weights',[kw,kh,in_ch,out_ch],dtype=self.dtype,initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2/(kw*kh*in_ch))))
            bias = tf.compat.v1.get_variable('bias',[out_ch],dtype=self.dtype,initializer=tf.constant_initializer(0))
            x = tf.nn.conv2d(input,kernel,padding=padding)
            x = tf.nn.bias_add(x,bias)
            x = self._activate(x,activation)
            weights.append(kernel)
            weights.append(bias)
            return x, weights

    def residual(self,input,block):
        return tf.add(input,block)

    def concate2d(self,input,block):
        return tf.concat([input,block],3)

    def VDSR(self,input,weights):
        x, weights = self.conv2d(input,(3,3,64),weights,activation='ReLU',padding='SAME')
        for _ in range(18):
            x ,weights = self.conv2d(x,(3,3,64),weights,activation='ReLU',padding='SAME')
        x ,weights = self.conv2d(x,(3,3,1),weights,activation='Linear',padding='SAME')
        x = self.residual(x,input)
        return x, weights

def main():

    path = './DeepLearning/VDSR'
    print('Check path...')
    if not os.path.isdir(path):
        print('Make directory')
        os.mkdir(path)
        print("load 'train_data.zip")
    url = 'https://cv.snu.ac.kr/research/VDSR/train_data.zip'
    name = 'DeepLearning/VDSR/train_data.zip'
    print('Check files...')
    if not os.path.isfile(name):
        request.urlretrieve(url,name)
        zipfile.ZipFile(name,'r').extractall('DeepLearning/VDSR/train_data')

    batch_size = 64
    patch_size = 41
    epoch = 80

    print('Patch extraction...')
    preprocess = PreProcessing(patch_size=patch_size)
    patch_in,patch_gt = [],[]
    patch_in,patch_gt = preprocess.load_train_data(path + '/train_data/91',patch_in,patch_gt)
    patch_in,patch_gt = preprocess.load_train_data(path + '/train_data/291',patch_in,patch_gt)

    x = tf.compat.v1.placeholder(tf.float32,shape=(batch_size,patch_size,patch_size,1))
    y = tf.compat.v1.placeholder(tf.float32,shape=(batch_size,patch_size,patch_size,1))
    weights = []
    net = Network(dtype=tf.float32,activation='ReLU')
    y_, weights = net.VDSR(x, weights)

    loss = tf.reduce_mean(tf.nn.l2_loss(tf.subtract(y_,y)))
    for weight in weights:
        loss += loss + tf.nn.l2_loss(weight)*1e-4
    tf.compat.v1.summary.scalar('Loss',loss)
    
    global_step = tf.Variable(0, trainable=False)
    lr = tf.compat.v1.train.exponential_decay(0.1,global_step*batch_size,len(patch_in)*20,0.1,staircase=True)
    tf.compat.v1.summary.scalar('Learning rate',lr)

    optTemp = tf.compat.v1.train.MomentumOptimizer(lr,0.9)
    grads_vars = optTemp.compute_gradients(loss)
    clipped_grads_vars = [(tf.clip_by_value(grad,-1.,1.),var) for grad, var in grads_vars]
    opt = optTemp.apply_gradients(clipped_grads_vars)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    idx = []
    for i in range(len(patch_in)//batch_size):
        idx.append(i)
    shuffle(idx)
    print(len(patch_in)//batch_size)
    print('Session start...')
    saver = tf.train.Saver(weights)
    with tf.compat.v1.Session(config=config) as sess:
        if not os.path.isdir(path + '/log'):
            os.mkdir(path + '/log')
        merge = tf.compat.v1.summary.merge_all()
        writer = tf.compat.v1.summary.FileWriter(path + '/log',sess.graph)
        tf.compat.v1.global_variables_initializer().run()

        for e in range(epoch):
            s = 0
            for step in idx:
                s += 1
                tr = patch_in[step*batch_size:step*batch_size+batch_size][:][:][:]
                gt = patch_gt[step*batch_size:step*batch_size+batch_size][:][:][:]
                
                feed_dict = {x:tr,y:gt}
                _,summary = sess.run([opt,merge],feed_dict=feed_dict)
                writer.add_summary(summary,step + len(idx)*e)
                print("Epoch: %d, (%d/%d)" % (e,s,len(idx)) )
        
        if not os.path.isdir(path + "/weights"):
            os.mkdir(path + "/weights")
        saver.save(sess,path + "/weights/VDSR.ckpt",global_step=global_step)
    return 0

if __name__ == '__main__':
    main()


    
        
