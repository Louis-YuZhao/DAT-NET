import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from config import config

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

from matplotlib import pyplot as plt
#%%
#classification_report
target_names = config['target_names']
n_classes =  len(target_names)
#%%
def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.   
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
def calculate_ROCAUC(Y_ture, Y_prob):
    ROCAUC = roc_auc_score(Y_ture, Y_prob) if len(np.unique(Y_ture)) > 1 else 0.0
    return ROCAUC
def plot_ROCAUC(y_test, y_score, n_classes):
    def plot_figure(fpr, tpr, lw, figureTitle):
        plt.figure()
        plt.plot(fpr[lw], tpr[lw], color='darkorange', lw=lw+1, label='ROC curve (area = %0.2f)' % roc_auc[lw])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw+1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic: {}'.format(figureTitle))
        plt.legend(loc="lower right")
        plt.show()
        return True

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    for i in range(n_classes):
        plot_figure(fpr, tpr, lw=1, figureTitle=target_names[i])
def performace_evaluate(groundTruth,predictedResult):
    """
    calculate the TP, FP, TN, FN
    """    
    eplison = 1e-6
    perDict = dict()
    conf_matrix = confusion_matrix(groundTruth, predictedResult)
    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    TN = conf_matrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN+eplison)
    # Specificity or true negative rate
    TNR = TN/(TN+FP+eplison) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP+eplison)
    # Negative predictive value
    NPV = TN/(TN+FN+eplison)
    # Fall out or false positive rate
    FPR = FP/(FP+TN+eplison)
    # False negative rate
    FNR = FN/(TP+FN+eplison)
    # False discovery rate
    FDR = FP/(TP+FP+eplison)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN+eplison)
    
    perDict['FP'] = FP
    perDict['FN'] = FN
    perDict['TP'] = TP
    perDict['TN'] = TN
    perDict['TPR'] = TPR
    perDict['TNR'] = TNR
    perDict['PPV'] = PPV
    perDict['NPV'] = NPV
    perDict['FPR'] = FPR
    perDict['FNR'] = FNR
    perDict['FDR'] = FDR
    perDict['ACC'] = ACC
    
    print('Sensitivity:'+ str(TPR))
    print('Specificity:'+ str(TNR))
    print('PPV        :'+ str(PPV))
    print('NPV        :'+ str(NPV))
    
    return perDict

def evaluate(projectRoot):
    # ground truth
    Y_test = np.load(os.path.join(projectRoot, 'testLabels.npy'))
    Y_test = np.squeeze(Y_test)
    Y_test = np.int64(Y_test)
    nb_classes = len(np.unique(Y_test))
    y_test = to_categorical(Y_test, nb_classes)
    print (Y_test)
    print (Y_test.dtype)

    # predict result
    y_predict = np.load(os.path.join(projectRoot,'Y_predict.npy'))
    # print y_predict
    Y_predict = np.argmax(y_predict, axis=1)
    print (Y_predict)
    print (Y_predict.dtype)

    # ROCAUC
    for item in target_names:
        position = target_names.index(item)
        print('ROCAUC_{}:{}\n'.format(item, calculate_ROCAUC(Y_ture=y_test[:,position], Y_prob = y_predict[:,position])))
    # plot ROCAUC
    plot_ROCAUC(y_test, y_predict, n_classes=n_classes)

    # classification report
    print(classification_report(Y_test, Y_predict, target_names=target_names))
    # Sensitivity, Specificity, PPV, NPV
    perDict = performace_evaluate(Y_test,Y_predict)
    with open(os.path.join(projectRoot,'evaResult.txt'),mode='w') as file_handle:
        for item in target_names:
            position = target_names.index(item)
            file_handle.write('ROCAUC_{}:{}\n'.format(item, calculate_ROCAUC(Y_ture=y_test[:,position], Y_prob = y_predict[:,position])))
        
        file_handle.write(classification_report(Y_test, Y_predict, target_names=target_names))
        
        file_handle.write('Sensitivity:'+ str(perDict['TPR'])+'\n')
        file_handle.write('Specificity:'+ str(perDict['TNR'])+'\n')
        file_handle.write('PPV        :'+ str(perDict['PPV'])+'\n')
        file_handle.write('NPV        :'+ str(perDict['NPV'])+'\n')
        
    # write information to csv
    df = pd.DataFrame({'Label':Y_test, 'Predict':Y_predict, 'PosNC':y_predict[:,0], 'PosPID':y_predict[:,1]})
    csv_dir = os.path.join(projectRoot,'allInformation.csv')
    df.to_csv(csv_dir,index=None)

def multi_fold_evaluate(projectRootList, RecordOutputRoot):  
    def data_read(projectRoot):
        # ground truth
        Y_test = np.load(os.path.join(projectRoot, 'testLabels.npy'))
        Y_test = np.squeeze(Y_test)
        Y_test= np.int64(Y_test)

        # pid
        pid_test_dir = os.path.join(projectRoot, 'testInformation.csv')
        df = pd.read_csv(pid_test_dir)
        pid_test_list = df['IPD'].to_numpy()

        # predict result
        y_predict = np.load(os.path.join(projectRoot,'Y_predict.npy'))
        y_predict = np.dot(y_predict, np.diag((1,1,1)))
        Y_predict = np.argmax(y_predict, axis=1)
        return pid_test_list, Y_test, Y_predict, y_predict

    pid_test_list = list()
    Y_test_list = list()
    y_predict_list = list()
    Y_predict_list = list()
    for projectRoot in projectRootList:
        pid_list, true, prediction, probability = data_read(projectRoot)
        pid_test_list.append(pid_list)
        Y_test_list.append(true)
        Y_predict_list.append(prediction)
        y_predict_list.append(probability)

    Y_PID = np.concatenate(pid_test_list,axis=0)
    Y_test = np.concatenate(Y_test_list,axis=0)
    nb_classes = len(np.unique(Y_test))
    y_test = to_categorical(Y_test, nb_classes)
    Y_predict = np.concatenate(Y_predict_list,axis=0)
    y_predict = np.concatenate(y_predict_list,axis=0)
    print (Y_test)
    print (Y_test.dtype)
    print (Y_predict)
    print (Y_predict.dtype)

    #%%
    # write information to csv
    df = pd.DataFrame({'PID':Y_PID, 'Label':Y_test, 'Predict':Y_predict, 'PosMSA':y_predict[:,0], 'PosPID':y_predict[:,1], 'PosPSP':y_predict[:,2]})
    csv_dir = os.path.join(RecordOutputRoot,'allInformation.csv')
    df.to_csv(csv_dir,index=None)

    # ROCAUC
    for item in target_names:
        position = target_names.index(item)
        print('ROCAUC_{}:{}\n'.format(item, calculate_ROCAUC(Y_ture=y_test[:,position], Y_prob = y_predict[:,position])))
    # plot ROCAUC
    plot_ROCAUC(y_test, y_predict, n_classes=n_classes)
    
    # classification report
    print(classification_report(Y_test, Y_predict, target_names=target_names))
    # Sensitivity, Specificity, PPV, NPV    
    perDict = performace_evaluate(Y_test,Y_predict)
    with open(os.path.join(RecordOutputRoot,'multiFoldEvaResult.txt'),mode='w') as file_handle:
        for item in target_names:
            position = target_names.index(item)
            file_handle.write('ROCAUC_{}:{}\n'.format(item, calculate_ROCAUC(Y_ture=y_test[:,position], Y_prob = y_predict[:,position])))
        
        file_handle.write(classification_report(Y_test, Y_predict, target_names=target_names))
        
        file_handle.write('Sensitivity:'+ str(perDict['TPR'])+'\n')
        file_handle.write('Specificity:'+ str(perDict['TNR'])+'\n')
        file_handle.write('PPV        :'+ str(perDict['PPV'])+'\n')
        file_handle.write('NPV        :'+ str(perDict['NPV'])+'\n')

#%%    
if __name__ == "__main__":
    projectStoreRoot = '../ProjectFolder'
    kfold = 6
    
    currentFold = 5
    projectRoot = os.path.join(projectStoreRoot, 'data_{0}_{1}'.format(kfold,currentFold))
    evaluate(projectRoot)