import torch
import os
import numpy as np
import random
from tqdm import tqdm

class DA_FewShot_Dataset(torch.utils.data.Dataset):
    def __init__(self, support_set, query_set):
        self.support_set = support_set
        self.query_set = query_set

    def __getitem__(self, idx):
        CSI = []
        for sample_path in self.support_set[idx]:
            csi = torch.load(sample_path)
            CSI.append(csi)
        csi = torch.load(self.query_set[idx])
        CSI.append(csi)
        return CSI

    def __len__(self):
        return len(self.query_set)

class DA_FewShot_DataLoader(object):
    def __init__(self, all_file, support_way, support_shot, query_way, query_shot, train_idx, valid_idx, test_idx, n_task_train, n_task_valid, n_task_test, batch_size, n_class, root, test_file, test_root, num_workers, target_factor):

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

        self.list_train = self.sort_file(all_file, train_idx)
        self.list_valid = self.sort_file(all_file, valid_idx)
        self.list_test = self.sort_file(test_file, test_idx)

        self.train_df = self.get_few_shot_df(root, all_file, self.list_train, n_task_train, target_factor)
        self.valid_df = self.get_few_shot_df(root, all_file, self.list_valid, n_task_valid, target_factor)
        self.test_df = self.get_few_shot_df(test_root, test_file, self.list_test, n_task_test, -1)

        self.train_loader = torch.utils.data.DataLoader(self.train_df, batch_size, num_workers=num_workers, drop_last=False)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_df, batch_size, num_workers=num_workers, drop_last=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_df, batch_size, num_workers=num_workers, drop_last=False)

    def get_few_shot_df(self, root, all_file, list_train, n_task_train, target_factor):
        cnt = np.zeros(12)
        all_label = [i for i in range(self.n_class)]
        support_set = []
        query_set = []

        for i in tqdm(range(n_task_train)):
            random_label = random.sample(all_label, self.support_way)
            support = []
            for label in random_label:
                label_sample = list_train[label]
                while len(label_sample) < self.support_shot:
                    other_labels = list(set(all_label) - set([label]))
                    label_sample.extend(list_train[random.choice(other_labels)])
                sample = random.sample(label_sample, self.support_shot)
                for k in range(self.support_shot):
                    support.append(os.path.join(root, all_file[sample[k]]))
            support_set.append(support)

            query_label = random.sample(random_label, self.query_way)
            cnt[int(query_label[0])] += 1

            if target_factor >= 0 and i % target_factor == 0:
                target_label = int(query_label[0])
                while len(self.list_test[target_label]) < self.query_shot:
                    other_labels = list(set(all_label) - set([target_label]))
                    target_label = random.choice(other_labels)
                query_idx = random.sample(self.list_test[target_label], self.query_shot)
                path = os.path.join(self.test_root, self.test_file[query_idx[0]])
            else:
                target_label = query_label[0]
                while len(list_train[target_label]) < self.query_shot:
                    other_labels = list(set(all_label) - set([target_label]))
                    target_label = random.choice(other_labels)
                query_idx = random.sample(list_train[target_label], self.query_shot)
                path = os.path.join(root, all_file[query_idx[0]])
            query_set.append(path)

        print(cnt)
        return DA_FewShot_Dataset(support_set, query_set)

    def return_label(self, file_name):
        if len(file_name) >= 2 and "I" in file_name[-2]:
            return int(file_name[-2].split("I")[1]) - 1
        else:
            return -1

    def sort_file(self, df, idx):
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

if __name__ == "__main__":
    # 在此添加执行主程序的代码
    pass
