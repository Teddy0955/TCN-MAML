from scipy import signal
from sklearn.decomposition import PCA as sk_pca
import pywt
import numpy as np
from scipy import ndimage, misc
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from datetime import datetime
import os


def Butterworth_filter(csi,N,rate):
    # N = 1
    # rate = 0.3
    b, a = signal.butter(N, rate, 'lowpass')

    # for idx in range(0,csi[0].__len__()):
    #     csi[:,idx] = signal.filtfilt(b, a, csi[:,idx])
    csi[:] = signal.filtfilt(b, a, csi[:])

def DWT(csi):
    # DWT Input shape : (30, 1500) = (subcarriers, packets)
    cA, cD = pywt.dwt(csi, 'db1')
    return cA, cD
def Median_filter(cA):
    # Median filter Input shape : (30, 80) = (subcarriers, packets)
    # cA is (30, 80)

    # result = []
    # for TR in range(cA.shape[0]):
    # tmp = cA[TR,:,:]
    med = ndimage.median_filter(cA, size=20)
    result = np.transpose(med)
    # result.append(np.transpose(med))

    #result is (80, 30) after transpose
    return np.array(result)

def PCA(result):
    #PCA Input shape : (80, 30) = (packets,subcarriers)
    # result is ( 80, 30)

    # PCA_list = []
    # for TR in range(result.shape[0]):
    #     tmp = np.transpose(result[TR,:,:])
    tmp = result
    my_pca_filter = sk_pca(n_components=6)
    tmp = my_pca_filter.fit(tmp).transform(tmp)
    tmp = tmp[:, 1:]
    #PCA_list.append(tmp)

    # PCA_list is (80, 5) = (TR pairs, packets, subcarriers)
    #return np.array(PCA_list)
    return tmp


def normalize(arr, t_min, t_max):
    # arr = (1500, 30)
    norm_arr = []
    diff = t_max - t_min
    a_min = np.amin(arr)
    a_max = np.amax(arr)
    diff_arr = a_max - a_min
    for packet in range(0,arr.shape[0]):
        norm_tmp = []
        for sub in range(0,arr.shape[1]):
            temp = (((arr[packet][sub] - a_min)*diff)/diff_arr) + t_min
            norm_tmp.append(temp)
        norm_arr.append(norm_tmp)
    return np.array(norm_arr)



def count_class( pre, label, class_acc, class_cnt):
    class_cnt[label] += 1
    if(pre == label):
        class_acc[label] += 1


def print_class_acc(class_cnt,class_acc):
    print("          Total number      Correct number         Accuracy")
    if(len(class_cnt)>9):
        for idx in range(0,9):
            if( class_cnt[idx] != 0):
                print("class %d:      %d                  %d                %.2f%%"
                    % (idx + 1, class_cnt[idx], class_acc[idx], (class_acc[idx] / class_cnt[idx]) * 100))
            else:
                print("class %d:      %d                  %d                %.2f%%"
                    % (idx + 1, class_cnt[idx], class_acc[idx], 0))
        for idx in range(9, class_cnt.__len__()):
            if( class_cnt[idx] != 0):
                print("class %d:     %d                  %d                %.2f%%"
                    % (idx + 1, class_cnt[idx], class_acc[idx], (class_acc[idx] / class_cnt[idx]) * 100))
            else:
                print("class %d:     %d                  %d                %.2f%%"
                    % (idx + 1, class_cnt[idx], class_acc[idx], 0))
    else:
        for idx in range(0,class_cnt.__len__()):
            if( class_cnt[idx] != 0):
                print("class %d:      %d                  %d                %.2f%%"
                    % (idx + 1, class_cnt[idx], class_acc[idx], (class_acc[idx] / class_cnt[idx]) * 100))
            else:
                print("class %d:      %d                  %d                %.2f%%"
                    % (idx + 1, class_cnt[idx], class_acc[idx], 0))

def save_confusion_matrix(y_true, y_pred,filename,args):
    cnf_matrix = confusion_matrix(y_true, y_pred)
    if(args.num_class == 2):
        target_names = ['Kicking with the left leg', 'Kicking with the right leg']
    elif(args.num_class == 12):
        target_names = ['Approaching', 'Departing', 'Handshaking','High five',
                        'Hugging', 'Kicking with the left leg', 'Kicking with the right leg', 'Pointing with the left hand',
                        'Pointing with the right hand', 'Punching with the left hand', 'Punching with the right hand', 'Pushing',
                        ]
    elif(args.num_class == 13):
        target_names = ['Approaching', 'Departing', 'Handshaking','High five',
                        'Hugging', 'Kicking with the left leg', 'Kicking with the right leg', 'Pointing with the left hand',
                        'Pointing with the right hand', 'Punching with the left hand', 'Punching with the right hand', 'Pushing',
                        'Steady']
    plot_confusion_matrix(filename,cnf_matrix, classes=target_names, normalize=True,
                          title=' confusion matrix')


def plot_confusion_matrix(filename, cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    font = {'family' : "Times New Roman",
        'size'   : 15}

    plt.figure(figsize=(15,15))
    cm = np.transpose(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if(cm[i][j]==0 or cm[i][j]==1):
            plt.text(j, i, format(int(cm[i, j]), 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.rc('font', **font)

    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.tight_layout()
    try:
        #plt.savefig(filename + ".jpg")
        #plt.cla()
        #plt.close("all")
        plt.savefig(os.path.join(args.img_path + str(fold+1) + "_confusion_matrix.jpg"))
        plt.cla()
        plt.close("all")
    except Exception as e:
        print(e)


def get_exp_set_name(args):
    exp_set = args.exp_set
    filters = "_"
    # filters += ( str(len(args.channel_size)) + "L")
    # filters += "("
    # for i in range(0,len(args.channel_size)-1):
    #     filters += (str(args.channel_size[i]) + ", ")
    # exp_set += (filters + str(args.channel_size[-1]) + ")" )
    # name = args.data_path.split('/')
    # name = name[1].split('.')
    # exp_set +=  ("_" + name[-1])
    # exp_set += ("_k" + str(args.kernel_size))
    # exp_set += ("_dropout" + str(int(args.dropout_rate*10)))
    now = datetime.now()
    y = datetime.strftime(now, '%Y')
    y = y[2:]
    s = datetime.strftime(now, '%m%d-%H.%M')
    t = y+s
    print(exp_set + "_" + t)
    return exp_set + "_" + t


def find_max_idx(arr):
    val = max(arr)
    for i in range(len(arr)):
        if(val==arr[i]):
            return i

def find_over_90(arr):
    for i in range(len(arr)):
        if(arr[i]>=0.9):
            return i


def print_six_acc(acc):
    print("___________________________________________________________\n")
    print("Before inner update, acc = ",float(int(acc[0]*10000)/100) , "%")
    for i in range(1,6):
        print("   Update ",i , " times, acc = ",float(int(acc[i]*10000)/100) , "%")
    print("___________________________________________________________")

def find_max_idx(arr):
    val = max(arr)
    for i in range(len(arr)):
        if(val==arr[i]):
            return i
