import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from MyFunction import count_class, print_class_acc, save_confusion_matrix, print_six_acc
import os

def eval_one(model, dataloader, args, fold, state):
    totalloss = 0
    totalacc = 0
    total_num = 0

    class_acc = np.zeros(args.num_class)
    class_cnt = np.zeros(args.num_class)

    two_acc = np.zeros(2)

    all_acc = np.zeros(args.update_step_test + 1, dtype=float)

    # this two lists are used to make a confusion matrix
    y_true = []
    y_pred = []

    with tqdm(dataloader, unit='batch', desc='Valid') as tqdm_loader:
        for idx, data in enumerate(tqdm_loader):
            batch_size = data[0][0].shape[0]
            total_num += batch_size

            # reshape the data
            for i in range(batch_size):
                for task_idx in range(len(data) - 1):
                    if not task_idx:
                        x_spt = data[task_idx][0][i, :, :, :].cuda().permute(0, 2, 1).unsqueeze(0)
                        y_spt = (data[task_idx][1][i]).cuda().unsqueeze(0)
                    else:
                        x_spt = torch.cat((x_spt, data[task_idx][0][i, :, :, :].cuda().permute(0, 2, 1).unsqueeze(0)), 0)
                        y_spt = torch.cat((y_spt, data[task_idx][1][i].cuda().unsqueeze(0)), 0)
                x_qry = data[-1][0][i, :, :, :].cuda().permute(0, 2, 1).unsqueeze(0)
                y_qry = data[-1][1][i].cuda().unsqueeze(0)

                if not i:
                    batch_x_spt = x_spt.unsqueeze(0)
                    batch_y_spt = y_spt.unsqueeze(0)
                    batch_x_qry = x_qry.unsqueeze(0)
                    batch_y_qry = y_qry.unsqueeze(0)
                else:
                    batch_x_spt = torch.cat((batch_x_spt, x_spt.unsqueeze(0)), 0)
                    batch_y_spt = torch.cat((batch_y_spt, y_spt.unsqueeze(0)), 0)
                    batch_x_qry = torch.cat((batch_x_qry, x_qry.unsqueeze(0)), 0)
                    batch_y_qry = torch.cat((batch_y_qry, y_qry.unsqueeze(0)), 0)

            acc, loss, pre_label, true_label = model(batch_x_spt, batch_y_spt, batch_x_qry, batch_y_qry)

            loss = loss / batch_size

            all_acc += np.concatenate([acc, np.zeros(all_acc.shape[0] - len(acc))])
            two_acc[0] += acc[0]
            two_acc[1] += acc[-1]
            totalloss += loss.item()
            totalacc += acc[-1]

            avg_loss = totalloss / (idx + 1)
            avg_acc = totalacc / total_num
            tqdm_loader.set_postfix(loss=loss.item(), avgloss=avg_loss, avgacc=avg_acc)

            for i in range(len(pre_label)):
                count_class(pre_label[i], true_label[i], class_acc, class_cnt)
                y_pred.append(pre_label[i])
                y_true.append(true_label[i])

        all_avg_loss = totalloss / len(tqdm_loader)
        if (all_avg_loss <= args.best_loss_valid):
            args.best_loss_valid = all_avg_loss
            if not os.path.exists(args.model_save_path):
                os.makedirs(args.model_save_path)
            torch.save(model.state_dict(), os.path.join(args.model_save_path, args.exp_set + "_MODEL.pt"))

        print("\nIn Validation:")
        print_class_acc(class_cnt, class_acc)
        all_acc /= total_num
        two_acc /= total_num
        print("___________________________________________________________\n")
        print("Before finetuning, acc = ", float(int(all_acc[0] * 10000) / 100), "%")
        for i in range(1, len(all_acc)):
            print("After ", str(i), " times update, acc = ", float(int(all_acc[i] * 10000) / 100), "%")

        if args.best_acc_valid < avg_acc:
            args.best_acc_valid = avg_acc
            
            try:
                save_confusion_matrix(y_true, y_pred, args.img_path + "Confusion Matrix/" + args.exp_set + "_" + state + "_MATRIX" + str(fold), args)
            except:
                print("111111111111111111")

        return avg_loss, avg_acc
