import torch
import os
import numpy as np
import random
from tqdm import tqdm

class DA_FewShot_Dataset(torch.utils.data.Dataset):
    def __init__(self, support_set, query_set):
        # self.root = root
        # self.all_file = all_file
        self.support_set = support_set
        self.query_set = query_set
        # self.testing_root = testing_root
        # self.test_file = test_file

    def __getitem__(self, idx):
        CSI = []
        for sample_path in self.support_set[idx]:
            # csi = torch.load(os.path.join(self.root, self.all_file[int(sample_idx)]))
            csi = torch.load(sample_path)
            CSI.append(csi)
        ############## query_set stores index:
        #csi = torch.load(os.path.join(self.testing_root, self.test_file[int(self.query_set[idx])]))

        ############## query_set stores direct path of file:
        csi = torch.load(self.query_set[idx])
        CSI.append(csi)
        #self.re_label(CSI)
        return CSI
    def __len__(self):
         return len(self.query_set)

    def re_label(self,CSI):
        for i in range(len(CSI)-1):
            if(CSI[i][1] == CSI[-1][1]):
                CSI[-1][1] = torch.tensor(i)
            CSI[i][1] = torch.tensor(i)


class DA_FewShot_DataLoader(object):
    def __init__(self,  all_file,
                        support_way,
                        support_shot,
                        query_way,
                        query_shot,
                        train_idx,
                        valid_idx,
                        test_idx,
                        n_task_train,
                        n_task_valid,
                        n_task_test,
                        batch_size,
                        n_class,
                        root,
                        test_file,
                        test_root,
                        num_workers,
                        target_factor
                        ):

        self.support_way = support_way
        self.support_shot = support_shot
        self.query_way = query_way
        self.query_shot = query_shot
        self.n_task_train = n_task_train
        self.n_task_valid = n_task_valid
        self.n_task_test = n_task_test
        self.n_class = n_class
        self.root = root
        self.target_factor = target_factor
        self.test_root = test_root
        self.test_file = test_file

        self.list_train = self.sort_file(all_file,  train_idx)
        self.list_valid = self.sort_file(all_file,  valid_idx)
        self.list_test  = self.sort_file(test_file, test_idx)

        self.train_df = self.get_few_shot_df(root,      all_file,  self.list_train, n_task_train, target_factor)
        self.valid_df = self.get_few_shot_df(root,      all_file,  self.list_valid, n_task_valid, target_factor)
        self.test_df  = self.get_few_shot_df(test_root, test_file, self.list_test , n_task_test,  -1)

        self.train_loader = torch.utils.data.DataLoader(self.train_df,batch_size,num_workers=num_workers,drop_last = False)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_df,batch_size,num_workers=num_workers,drop_last = False)
        self.test_loader = torch.utils.data.DataLoader(self.test_df,batch_size,num_workers=num_workers,drop_last = False)


    def __getitem__(self, idx):
        return __loader__
    def __len__(self):
         return __name__
    

    def get_few_shot_df(self,root, file, list, n_task, target_factor):
        cnt = np.zeros((12))
        
        all_label = [ i for i in range(self.n_class)]
        support_set = []
        query_set = []

        for i in tqdm(range(n_task)):
            ################ support set
            random_label = random.sample(all_label, self.support_way)
            support = []
            for label in random_label:
                sample = random.sample(list[int(label)], self.support_shot)
                for k in range(self.support_shot):
                    support.append(root + file[sample[k]])

            support_set.append(support)
            
            query_label = random.sample(random_label, self.query_way)
            cnt[int(query_label[0])]+=1
            if(target_factor >=0 and i%target_factor==0):
                query_idx = random.sample(self.list_test[int(query_label[0])], self.query_shot)
                path = self.test_root + self.test_file[int(query_idx[0])]
            else:
                query_idx = random.sample(list[int(query_label[0])], self.query_shot)
                path = root + file[int(query_idx[0])]
            
            query_set.append(path)
        print(cnt)
        return DA_FewShot_Dataset(support_set,query_set)

    def return_label(self,file_name):
        # get label from file name
        file_name = file_name.split("_")
        file_name = file_name[-2].split("I")
        return int(file_name[1])-1

    def sort_file(self,df, idx):
        class_list = []
        for i in range(12):
            class_list.append([])
        for i in idx:
            label = self.return_label(df[i])
            class_list[label].append(i)
        return class_list

    def get_train_dataloaders(self):
        return self.train_loader
    def get_valid_dataloaders(self):
        return self.valid_loader
    def get_test_dataloaders(self):
        return self.test_loader