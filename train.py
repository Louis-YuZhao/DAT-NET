import os
import numpy as np
from shutil import copyfile
from config import config
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_num']
from Models import Resnet3DBuilder, resnet3d_model, densenet3d_model
from Models.training import get_callbacks
from Models.generator import convertData, preprocess, DataGeneratorNew
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.layers import Activation, Dense, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from evaluate import evaluate
from keras import backend as K
from skimage.transform import resize
from math import ceil
K.image_data_format() == "channels_last"

# parameters
#---------------------------------
kfold = 6
currentFold = 1
num_outputs = 2
class_weight = None

learning_rate = 1e-4
learning_rate_drop = 0.5
learning_rate_patience = 10
early_stopping_patience = 30

IFUseWeight = True # True: wholemodel, False: None

baseModeType = 'resnew' #'resori', 'resnew', 'dense'
optType = 'adam' # 'adam', 'sgd'
epochs = 50
batch_size = num_outputs

dimz = config['dimz']
dimx = config['dimx']
dimy = config['dimy']
channelNum = config['channelNum']
#%%
def get_model_new(baseModeType, optType, reg_factor=1e-4):
    
    if baseModeType == 'resori':
        model = Resnet3DBuilder.build_resnet_18((dimz, dimx, dimy, channelNum), num_outputs, 
                                                reg_factor = reg_factor, ifbase=False)
    elif baseModeType == 'resnew':
        model = resnet3d_model(input_shape=(dimz, dimx, dimy, channelNum), num_outputs=num_outputs, 
                    n_base_filters=64, depth=4, dropout_rate=0.3, kernel_reg_factor = reg_factor, ifbase=False)
    elif baseModeType == 'dense':
        model = densenet3d_model(input_shape=(dimz, dimx, dimy, channelNum), num_outputs=num_outputs, 
                    n_base_filters=64, depth=3, dropout_rate=0.3, kernel_reg_factor = reg_factor, ifbase=False)     
    # Name layers
    for i, layer in enumerate(model.layers):
        layer.name = 'layer_' + str(i)
    
    if optType == 'adam':
        opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optType == 'sgd':
        opt = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)        
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    return model

# train
def train_and_predict(projectRoot):
    
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30) 

    labelDef = config['labelDef']

    inputDict = dict()
    for key in list(labelDef.keys()):
        inputDict[key] = os.path.join(projectRoot, '{}.npy'.format(key))

    features=dict()
    for key in list(labelDef.keys()):
        features[key] = np.load(inputDict[key])
    
    itemNum = int(0)
    for key, value in features.items():
        itemNum = itemNum+value.shape[0]
    steps_per_epoch = ceil(itemNum / batch_size) 
   
    labelPercentage = dict()
    for key in list(labelDef.keys()):
        labelPercentage[key] = 1/float(len(list(labelDef.keys())))

    X_test = np.load(os.path.join(projectRoot,'testFeatures.npy'))
    y_test = np.load(os.path.join(projectRoot,'testLabels.npy'))
    X_test,Y_test = convertData(X_test,y_test, config) 
    validation_data = (X_test,Y_test) 
    TrainData = DataGeneratorNew(features, labelDef, labelPercentage, batch_size)
    
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    #---------------------------------#
    model = get_model_new(baseModeType, optType, reg_factor=1e-4)
    #---------------------------------#
    weightDir = os.path.join(projectRoot, 'Weights.h5')
   
    callbackList = get_callbacks(weightDir, learning_rate, learning_rate_drop, learning_rate_patience, 
                            learning_rate_epochs=None, logging_file="training.log", verbosity=1,
                            early_stopping_patience=early_stopping_patience)
    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)   

    if os.path.exists(weightDir) and IFUseWeight == True:
        model.load_weights(weightDir)
    if class_weight != None:
        train_history = model.fit_generator(TrainData.generator(), 
                                            steps_per_epoch=steps_per_epoch, 
                                            epochs=epochs, 
                                            verbose=1, 
                                            callbacks=callbackList, 
                                            validation_data=validation_data, 
                                            validation_steps=1, 
                                            class_weight=class_weight, 
                                            max_queue_size=10, 
                                            workers=1, 
                                            use_multiprocessing=False)

    else:
        train_history = model.fit_generator(TrainData.generator(), 
                                            steps_per_epoch=steps_per_epoch, 
                                            epochs=epochs, 
                                            verbose=1, 
                                            callbacks=callbackList, 
                                            validation_data=validation_data, 
                                            validation_steps=1, 
                                            max_queue_size=10, 
                                            workers=1, 
                                            use_multiprocessing=False)

    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    np.save(os.path.join(projectRoot,'loss.npy'),loss)
    np.save(os.path.join(projectRoot,'val_loss.npy'),val_loss)
    
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    
    X_test = np.load(os.path.join(projectRoot,'testFeatures.npy'))
    y_test = np.load(os.path.join(projectRoot,'testLabels.npy'))
    X_test, Y_test = convertData(X_test,y_test, config)    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
   
    model.load_weights(weightDir)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    Y_predict = model.predict(X_test, verbose=1)
    np.save(os.path.join(projectRoot,'Y_predict.npy'), Y_predict) 

def value_predict(X_test, baseModeType, load_weight_dir, outputDir=None):   
    
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)  

    X_test = preprocess(X_test,dimz,dimx,dimy,channelNum)
    X_test = X_test.astype('float32')
    
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)

    #---------------------------------#
    model = get_model_new(baseModeType, optType = optType, reg_factor=1e-4)
    #---------------------------------#    
    model.load_weights(load_weight_dir)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    Y_predict = model.predict(X_test, verbose=1)
    if outputDir != None:
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
            np.save(os.path.join(outputDir,'Y_predict.npy'), Y_predict)
    return Y_predict 

if __name__ == '__main__':
    currentOpePath = os.path.realpath(__file__)
    print(currentOpePath)
    currBname = os.path.basename(currentOpePath) 

    projectStoreRoot = '../ProjectFolder'
    projectRoot = os.path.join(projectStoreRoot,'data_{0}_{1}'.format(kfold,currentFold))
    train_and_predict(projectRoot)
    copyfile(currentOpePath, os.path.join(projectRoot,'train.py'))
    evaluate(projectRoot)