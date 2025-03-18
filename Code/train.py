import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from MyFunction import count_class, print_class_acc, save_confusion_matrix, print_six_acc
from torchstat import stat
from copy import deepcopy
import argparse

def train_one(model, dataloader, args, fold):
    totalloss = 0
    total_num = 0

    class_acc = np.zeros(args.num_class)
    class_cnt = np.zeros(args.num_class)
    '''
    Record six different accuracy
    1. The accuracy before any inner update
    2. The accuracy after 1st inner update
    3. The accuracy after 2nd inner update
    4. The accuracy after 3rd inner update
    5. The accuracy after 4th inner update
    6. The accuracy after 5th inner update
    '''
    all_acc = np.zeros(args.update_step+1)


    # this two list are used to make a confusion matrix
    y_true = []
    y_pred = []

    with tqdm(dataloader, unit='batch', desc='Train') as tqdm_loader:
        for idx, data in enumerate(tqdm_loader):
            '''
            data = (6,  2,   4,   6,   375,    30)
                1st 6 = (number of support set + number of query set) = 5 + 1
                2nd 2 = first dim. for data, second for label
                3rd 4 = the number of task in the batch
                4th 6 = number of TR-pair
                5th 375 = length of packets
                6th 30 = subcarriers
            
            '''
            batch_size = data[0][0].shape[0]
            total_num += batch_size
            
            # reshape the data
            for i in range(batch_size): #batch
                model.train()
                for task_idx in range(len(data)-1):
                    if(not task_idx):
                        
                        x_spt = data[task_idx][0][i,:,:,:].cuda().permute(0,2,1).unsqueeze(0) # original = (0,2,1)
                        y_spt = (data[task_idx][1][i]).cuda().unsqueeze(0)
                    else:
                        x_spt = torch.cat((x_spt,data[task_idx][0][i,:,:,:].cuda().permute(0,2,1).unsqueeze(0)),0) # original = (0,2,1)
                        y_spt = torch.cat((y_spt,data[task_idx][1][i].cuda().unsqueeze(0)),0)

                x_qry = data[-1][0][i,:,:,:].cuda().permute(0,2,1).unsqueeze(0)
                y_qry = data[-1][1][i].cuda().unsqueeze(0)
                if not i:
                    batch_x_spt = x_spt.unsqueeze(0)
                    batch_y_spt = y_spt.unsqueeze(0)
                    batch_x_qry = x_qry.unsqueeze(0)
                    batch_y_qry = y_qry.unsqueeze(0)
                else:
                    batch_x_spt = torch.cat((batch_x_spt,x_spt.unsqueeze(0)),0)
                    batch_y_spt = torch.cat((batch_y_spt,y_spt.unsqueeze(0)),0)
                    batch_x_qry = torch.cat((batch_x_qry,x_qry.unsqueeze(0)),0)
                    batch_y_qry = torch.cat((batch_y_qry,y_qry.unsqueeze(0)),0)
            # print("x_spt = ",batch_x_spt.shape)
            # print("y_spt = ",batch_y_spt.shape)
            # print("x_qry = ",batch_x_qry.shape)
            # print("y_qry = ",batch_y_qry.shape)
            #input()
            acc, loss, pre_label, true_label = model(batch_x_spt, batch_y_spt, batch_x_qry, batch_y_qry)

            loss = loss/batch_size
            totalloss += loss.item()
            all_acc += acc

            avg_loss = totalloss / (idx+1)
            avg_acc  = all_acc[-1] / total_num

            tqdm_loader.set_postfix(loss=loss.item(), avgloss=avg_loss, avgacc=avg_acc)

            for i in range(len(pre_label)):
                count_class(pre_label[i], true_label[i], class_acc, class_cnt)
                y_pred.append(pre_label[i])
                y_true.append(true_label[i])
    print("\nIn Training:")
    print_class_acc(class_cnt,class_acc)
    all_acc/=total_num
    print("___________________________________________________________\n")
    print("Before finetuning, acc = ", float(int(all_acc[0]*10000)/100), "%")
    for i in range(1,len(all_acc)):
        print("After ", str(i)," times update, acc = ", float(int(all_acc[i]*10000)/100), "%")

    if(args.best_acc_train < avg_acc):
        args.best_acc_train = avg_acc
        try:
            save_confusion_matrix(y_true, y_pred, args.img_path + "Confusion Matrix/" + args.exp_set + "_train_MATRIX"+ str(fold), args)
        except Exception as e:
            print(e)

    return avg_loss,avg_acc