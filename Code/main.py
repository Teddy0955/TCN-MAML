import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from train import train_one
from evaluate import eval_one
#from test import test_one
from sklearn.model_selection import KFold
from HHI_meta import HHI_Meta
import numpy as np
import argparse

from FewShot_DataLoader import DA_FewShot_DataLoader
import os
from MyFunction import get_exp_set_name, find_max_idx
from thop import profile
from thop import clever_format
import json
import random

class CSIDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.files = os.listdir(root)
       
    def __getitem__(self, idx):
        
        return self.files[idx]
    def __len__(self):
         return len(self.files)

def main(args):
    target_fold = random.sample([ i for i in range(args.K_fold)], 1)

    random.seed(880614)
    all_file = CSIDataset(args.root)
    test_file = CSIDataset(args.test_root)
    if(args.root.find('Aug')!=-1):
        args.n_task_train *=4
        args.n_task_valid *=4
        args.n_task_test  *=4

    LR_lambda = lambda epoch: 0.988 ** int(epoch)

    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    test_loss_list = []
    test_acc_list = []

    model_structure = []
    for i in range(len(args.channel_size)):
        if(not i):
            model_structure.append(('conv1d',[args.input_channel, args.channel_size[0], args.kernel_size, args.dilation[i]]))
        else:
            model_structure.append(('conv1d', [args.channel_size[i], args.channel_size[i], args.kernel_size, args.dilation[i]]))
        model_structure.append(('relu', [True]))
        model_structure.append(('dropout', [args.dropout_rate]))
        if(i+1 == len(args.channel_size)):
            model_structure.append(('conv1d', [args.channel_size[i], args.channel_size[-1], args.kernel_size, args.dilation[i]]))
        else:
            model_structure.append(('conv1d', [args.channel_size[i], args.channel_size[i+1], args.kernel_size, args.dilation[i]]))
        model_structure.append(('relu', [True]))
        model_structure.append(('dropout', [args.dropout_rate]))

        if(i== len(args.channel_size)-1):
            model_structure.append(('pick_time',-1))
            model_structure.append(('flatten',[]))
            model_structure.append(('linear', [args.num_class, args.channel_size[-1]*args.TR_pairs]))  # (output_channel, input_channel)



    KF = KFold(n_splits=args.K_fold, random_state=2022,shuffle=True)
    for fold,(train_idx,valid_idx) in enumerate(KF.split(all_file)):
        print(fold)
        if(fold != 2):
            continue
        args.exp_set  = get_exp_set_name(args)
        with open('../img/FewShot/MAML/' + args.exp_set +'.txt', 'w') as f:
            json.dump(vars(args), f, indent=2)
        print("\n\n**********  Starting the experiment set: " +  args.exp_set  +  "   **********")
        model = HHI_Meta(args, model_structure)
                 
        DL = DA_FewShot_DataLoader(all_file,
                                support_way= args.n_way, 
                                support_shot = args.k_spt,
                                query_way = 1, 
                                query_shot= args.k_qry,
                                train_idx = train_idx,
                                valid_idx = valid_idx, #0
                                test_idx = np.arange(0,len(test_file)),
                                n_task_train = args.n_task_train,
                                n_task_valid = args.n_task_valid,
                                n_task_test = args.n_task_test,
                                batch_size= args.batch,
                                n_class = args.num_class,
                                root= args.root,
                                test_file=test_file,
                                test_root = args.test_root,
                                num_workers = args.num_workers,
                                target_factor = args.target_factor
                                )
        
        trainDataLoader = DL.get_train_dataloaders()
        validDataLoader = DL.get_valid_dataloaders()
        testDataLoader = DL.get_test_dataloaders()

        
        for epoch in range(args.epoch):
            save_list = []
            print('\nFold {}'.format(fold + 1) + " of " + args.exp_set )
            print('\nEpoch {}'.format(epoch+1))
            print('\nEpoch {}'.format(epoch+1) + "  LR=" + f"{model.meta_optim.param_groups[0]['lr']}")
            train_loss, train_acc = train_one(model, trainDataLoader, args, fold+1)
            #if(epoch%5 == 0):
            valid_loss, valid_acc = eval_one(model, validDataLoader, args, fold+1,"valid")
            print("___________________________________________________")
            print("                 TESTING ON GROUP2                 ")
            print("---------------------------------------------------")

            test_loss, test_acc = eval_one(model,testDataLoader,args,fold+1,"testing") # test on Group2

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

            save_list.append(train_loss_list)
            save_list.append(train_acc_list)
            save_list.append(valid_loss_list)
            save_list.append(valid_acc_list)
            save_list.append(test_loss_list)
            save_list.append(test_acc_list)
            torch.save(save_list,args.img_path + "data/" + args.exp_set + "_PT" + str(fold+1) + "curve.pt")
            print("Best Training accuracy = {}".format(max(train_acc_list)))
            print("Best Validation accuracy = {}".format(max(valid_acc_list)))
            print("Best testing accuracy = {}".format(max(test_acc_list)))

            # plot and save figure
            font = {'family' : "Times New Roman",
                    'size'   : 22}
            plt.rc('font', **font)
            plt.figure(figsize=(13,10))
            plt.plot(train_loss_list, label="train_loss")
            plt.plot(train_acc_list, label="train_acc")
            plt.plot(valid_loss_list, label="valid_loss")
            plt.plot(valid_acc_list, label="valid_acc")
            plt.plot(test_loss_list, label="test_loss")
            plt.plot(test_acc_list, label="test_acc")
            plt.xlabel('Epoch')
            plt.legend()
            plt.tight_layout()
            plt.gcf().subplots_adjust(bottom=0.15)

            plt.text(find_max_idx(train_acc_list), max(train_acc_list) -0.05 ,'%.4f' % max(train_acc_list))
            plt.text(find_max_idx(valid_acc_list), max(valid_acc_list) - 0.05,'%.4f' % max(valid_acc_list))
            plt.text(find_max_idx(test_acc_list), max(test_acc_list) -0.05 ,'%.4f' % max(test_acc_list))
            
            plt.vlines(find_max_idx(train_acc_list),0,max(max(train_loss_list),1),color="orange")
            plt.vlines(find_max_idx(valid_acc_list),0,max(max(train_loss_list),1),color="red")
            plt.vlines(find_max_idx(test_acc_list),0,max(max(train_loss_list),1),color='#734940')
            
            plt.savefig(args.img_path + args.exp_set + "_FIG" + str(fold+1) + ".jpg")
            #plt.show()
            plt.cla()
            plt.close("all")
        train_loss_list = []
        train_acc_list = []
        valid_loss_list = []
        valid_acc_list = []
        test_loss_list = []
        test_acc_list = []

        #print("\nIn TESTING dataset:")
        #__, __ = eval_one(model, testDataLoader, certification)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()    
    argparser.add_argument('--Noting', type=str,
                           help='Some notes that specfices this exp.', 
                           default= 'fix fold to 2\n'+
                                    'Same condition on train and valid' +
                                    'Drop_Ablation Study on raw data: Dropout 0.1'+
                                    'With Focal Loss, alpha = 0.25, gamma = 3\n'+
                                    'With validation\n' +
                                    'target_factorf 3\n'+
                                    'dataset raw+drop\n' +
                                    'outer-learning rate 5e-4,\n' +
                                    'inner-learning rate 1e-4,\n' +
                                    'and focus_lr = 1')
    argparser.add_argument('--dropout_rate', type=float,
                           help='dropout rate in TCN', default=0.1)
    argparser.add_argument('--root', type=str,
                           help='data path', default="../Group1/")
    argparser.add_argument('--test_root', type=str,
                           help='test data path', default="../Group2/")
    argparser.add_argument('--target_factor', type=int,
                           help='To decide how many target sample will be mixed into training. 4 means 4 tasks from source domain and 1 task from target domain \
                            and -1 means no any task in target domain is used',
                             default=3)
    argparser.add_argument('--focus_lr', type=int,
                           help='Multiple the meta_lr with focus_lr while encounter a task from target domain', default=1)
    argparser.add_argument('--meta_lr', type=float,
                           help='meta-level outer learning rate', default=5e-4)
    argparser.add_argument('--update_lr', type=float,
                           help='task-level inner update learning rate', default=5e-4)
    argparser.add_argument('--alpha', type=float,
                           help='alpha in focal loss', default=0.25)
    argparser.add_argument('--gamma', type=float,
                           help='gamma in focal loss', default=3)
    # total is 4800 sample in the raw dataset
    argparser.add_argument('--n_task_train', type=int,
                           help='number of tasks during training (4800*0.9)', default=2500)
    argparser.add_argument('--n_task_valid', type=int,
                           help='number of tasks during validation (4800*0.1)', default=500)
    argparser.add_argument('--n_task_test', type=int,
                           help='number of tasks during testing (4800*0.1)', default=500) 
    
    # Settings of Few-Shot Learning
    argparser.add_argument('--epoch', type=int,
                        #    help='epoch number', default=30)
                            help='epoch number', default=1)
    argparser.add_argument('--num_class', type=int,
                           help='number of classes', default=12)     
    argparser.add_argument('--n_way', type=int,
                           help='n way', default=5)
    argparser.add_argument('--k_spt', type=int,
                           help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int,
                           help='k shot for query set', default=1)
    argparser.add_argument('--update_step', type=int,
                           help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int,
                           help='update steps for finetunning', default=10)
    argparser.add_argument('--num_workers', type=int,
                           help='number of workers', default=1)


    # Model hyper-parameter
    argparser.add_argument('--meta_optim', type=str,
                           help='optimier used in MAML', default='AdamW')

    argparser.add_argument('--kernel_size', type=int,
                           help='kernel size of 1D-CNN in TCN', default=15)   
    argparser.add_argument('--input_channel', type=int,
                           help='Number of channel of input', default=30)
    argparser.add_argument('--TR_pairs', type=int,
                           help='Number of TR pairs', default=6)
    argparser.add_argument('--batch', type=int,
                           help='meta batch size, namely task num', default=32)
    argparser.add_argument('--lr_decay', type=float,
                           help='the rate of decaying of learning rate', default=0.993)  
    argparser.add_argument('--channel_size', type=list,
                           help='channel size of each layer in TCN', default=[50,50,50])                           
    argparser.add_argument('--dilation', type=list,
                           help='dilation_size size of each layer in TCN', default=[1,2,4])                                 
    
    argparser.add_argument('--K_fold', type=int,
                           help='The valud of K in K-fold validation', default=10)   

    argparser.add_argument('--model_save_path', type=str,
                           help='data path', default="./torch.MAML.Model/")       
    argparser.add_argument('--img_path', type=str,
                           help='data path', default="../img/FewShot/MAML/")
    argparser.add_argument('--exp_set', type=str,
                           help='data path', default="MAML_TCN")

    
    argparser.add_argument('--best_acc_train', type=float,
                           help='best training accuracy', default=-1)
    argparser.add_argument('--best_acc_valid', type=float,
                           help='best validation accuracy', default=-1)
    argparser.add_argument('--best_loss_valid', type=float,
                           help='best validation loss ', default=100)

    args = argparser.parse_args()

    main(args)