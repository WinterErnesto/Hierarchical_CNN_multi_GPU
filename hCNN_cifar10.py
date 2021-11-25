import keras
import keras.backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import np_utils
import argparse

from hCNN_models import hCNN_cifar10
from hCNN_utils import make_parallel, Step
import numpy as np
  

#Auf der Suche nach dem Fehler. Try batchsize 30. (instead of 50): also epochs: instead 0f 320 use 20
'''parser = argparse.ArgumentParser(description='Hierarchical CNN Train Script.')
parser.add_argument("--batch_size", dest="batch_size", default=20, type=int,
                    help='batch size')
parser.add_argument("--num_epochs", dest="num_epochs", default=6, type=int,
                    help="Number of epochs")
parser.add_argument("--num_gpus", dest="num_gpus", default=1, type=int,
                    help="Number of GPUs")
args = parser.parse_args()
'''
def norm_data(data_set):
    mean = np.array([125.3, 123.0, 113.9])
    std = np.array([63.0, 62.1, 66.7])
    data_set -= mean
    data_set /= std
    return data_set

batch_size=10
num_epochs=320
num_gpus=4

#print("Using %i GPUs" %num_gpus)

if __name__ == '__main__':
    print ('Training Hierarchical CNN')
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    
    v_j = 16
    v_1 = 7
    v_2 = 11
    #batch = args.batch_size*args.num_gpus
    batch = batch_size*num_gpus

    #To understand the problem
    #print(batch)
    #print(args.num_gpus)
    #print(args.batch_size)


    model = hCNN_cifar10(v_j, v_1, v_2)
    
    if num_gpus > 1:
        model = make_parallel(model,num_gpus)
    
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print ('Model compiled.')

    (X_train,y_train), (X_test,y_test) = cifar10.load_data()
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    #Reduce size of training and testing set to make 
    #X_test_smaller=X_test[:5000,:,:,:]
    #X_train_smaller=X_train[:10000,:,:,:]
    #y_test_smaller=y_test[:5000,:]
    #y_train_smaller=y_train[:10000,:]
    
    
    
    
    train_set = {}
    test_set = {}
    train_set['data'] = norm_data(X_train)
    test_set['data'] = norm_data(X_test)
    train_set['labels'] = np_utils.to_categorical(y_train)
    test_set['labels'] = np_utils.to_categorical(y_test)
    
    #train_set['data'] = norm_data(X_train_smaller)
    #test_set['data'] = norm_data(X_test_smaller)
    #train_set['labels'] = np_utils.to_categorical(y_train_smaller)
    #test_set['labels'] = np_utils.to_categorical(y_test_smaller)
    
    
    
    print ('Data Loaded and Preprocessed.')

   #nb_epoch = args.num_epochs
    nb_epoch = num_epochs
    callbacks = []
    steps = [40,80,120,160,200,240,260,280,300]
    
    lr_mult = np.array(0.35*np.sqrt(num_gpus)).astype(float)
    #lr_mult = np.array(0.35*np.sqrt(args.num_gpus)).astype(float)
    schedule = Step(steps, lr_mult*[1.0,0.5,0.25, 0.12, 0.06, 0.03, 0.015, 0.007, 0.00035, 0.00017], verbose=1)
    callbacks.append(schedule)
    schedule = None
    name = './results/hCNN_weights'
    data_gen = ImageDataGenerator(horizontal_flip=True,
                                  width_shift_range=0.125,
                                  height_shift_range=0.125,
                                  fill_mode='constant')
    data_iter = data_gen.flow(train_set['data'], train_set['labels'],
                              batch_size=batch, shuffle=True)
    print ('Starting fit_generator.')
    model.fit_generator(data_iter,
                        samples_per_epoch=train_set['data'].shape[0],
                        nb_epoch=nb_epoch,
                        verbose = 1,
                        callbacks=callbacks)
    score = model.evaluate(test_set['data'], test_set['labels'], verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
   # model.save_weights("model.h5"): still need to add a name for folder#
   #model.save_weights(name+'.h5') -- this is the old line 
    model.save_weights('weights_cifar10_9817.h5')
    
    # To load the old model
    #model.load_weights('weights_cifar10_9817.h5')
