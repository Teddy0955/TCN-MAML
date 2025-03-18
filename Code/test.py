import torch
import argparse
import os
import time
from evaluate import eval_one
from HHI_meta import HHI_Meta
from FewShot_DataLoader import DA_FewShot_DataLoader
from thop import profile
import psutil

def load_model(args, model_path):
    print(f"Loading model from {model_path}")
    model_structure = []
    for i in range(len(args.channel_size)):
        if not i:
            model_structure.append(('conv1d', [args.input_channel, args.channel_size[0], args.kernel_size, args.dilation[i]]))
        else:
            model_structure.append(('conv1d', [args.channel_size[i], args.channel_size[i], args.kernel_size, args.dilation[i]]))
        model_structure.append(('relu', [True]))
        model_structure.append(('dropout', [args.dropout_rate]))
        if i + 1 == len(args.channel_size):
            model_structure.append(('conv1d', [args.channel_size[i], args.channel_size[-1], args.kernel_size, args.dilation[i]]))
        else:
            model_structure.append(('conv1d', [args.channel_size[i], args.channel_size[i + 1], args.kernel_size, args.dilation[i]]))
        model_structure.append(('relu', [True]))
        model_structure.append(('dropout', [args.dropout_rate]))
    model_structure.append(('pick_time', -1))
    model_structure.append(('flatten', []))
    model_structure.append(('linear', [args.num_class, args.channel_size[-1] * args.TR_pairs]))
    
    model = HHI_Meta(args, model_structure)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    
    # 計算模型大小
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2)  # MB
    print(f"Total Model Size: {model_size:.2f} MB")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_root', type=str, default="../Group2/", help='Path to test dataset')
    parser.add_argument('--model_save_path', type=str,default="./torch.MAML.Model/", help='Path where model is saved')
    parser.add_argument('--model_name', type=str, default="MAML_TCN_250309-11.02_MODEL.pt", help='Model filename to load')
    parser.add_argument('--batch', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for DataLoader')
    parser.add_argument('--num_class', type=int, default=12, help='Number of output classes')
    parser.add_argument('--channel_size', type=list, default=[50, 50, 50], help='Channel sizes for TCN')
    parser.add_argument('--dilation', type=list, default=[1, 2, 4], help='Dilation rates for TCN')
    parser.add_argument('--kernel_size', type=int, default=15, help='Kernel size for TCN')
    parser.add_argument('--input_channel', type=int, default=30, help='Input channel size')
    parser.add_argument('--TR_pairs', type=int, default=6, help='Number of TR pairs')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--update_step_test', type=int, default=10, help='Update steps for finetuning')
    parser.add_argument('--update_lr', type=float, default=1e-3, help='Task-level inner update learning rate')
    parser.add_argument('--meta_lr', type=float, default=5e-4, help='Meta-level outer learning rate')
    parser.add_argument('--n_way', type=int, default=5, help='Number of classes per task')
    parser.add_argument('--k_spt', type=int, default=3, help='Number of support examples per class')
    parser.add_argument('--k_qry', type=int, default=3, help='Number of query examples per class')
    parser.add_argument('--update_step', type=int, default=5, help='Number of inner loop updates')
    parser.add_argument('--alpha', type=float, default=0.25, help='Alpha parameter for focal loss')
    parser.add_argument('--gamma', type=float, default=3.0, help='Gamma parameter for focal loss')
    parser.add_argument('--meta_optim', type=str, default='AdamW', help='Optimizer for meta-learning')
    parser.add_argument('--focus_lr', type=float, default=1.0, help='Learning rate scaling for target domain tasks')
    parser.add_argument('--lr_decay', type=float, default=0.993, help='Learning rate decay factor')
    parser.add_argument('--best_acc_valid', type=float, default=-1, help='Best validation accuracy')
    parser.add_argument('--best_loss_valid', type=float, default=100, help='Best validation loss')
    parser.add_argument('--exp_set', type=str, help='data path', default="MAML_TCN")
    args = parser.parse_args()

    test_file_list = os.listdir(args.test_root)
    
    data_loader = DA_FewShot_DataLoader(
        all_file=test_file_list,
        support_way=args.num_class, 
        support_shot=1, 
        query_way=1, 
        query_shot=1, 
        train_idx=[], 
        valid_idx=[], 
        test_idx=range(len(test_file_list)), 
        n_task_train=0, 
        n_task_valid=0, 
        n_task_test=len(test_file_list), 
        batch_size=args.batch, 
        n_class=args.num_class, 
        root=args.test_root, 
        test_file=test_file_list, 
        test_root=args.test_root, 
        num_workers=args.num_workers, 
        target_factor=-1
    )

    test_loader = data_loader.get_test_dataloaders()
    model_path = os.path.join(args.model_save_path, args.model_name)
    model = load_model(args, model_path)
    
    print("Starting evaluation on test set...")
    start_time = time.time()
    gpu_memory_before = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    ram_memory_before = psutil.virtual_memory().used / (1024 ** 2)
    
    avg_loss, avg_acc = eval_one(model, test_loader, args, fold=1, state='testing')
    
    gpu_memory_after = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    ram_memory_after = psutil.virtual_memory().used / (1024 ** 2)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    gpu_memory_used = gpu_memory_after - gpu_memory_before
    ram_memory_used = ram_memory_after - ram_memory_before
    
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.2f}%")
    print(f"Evaluation Time: {elapsed_time:.2f} seconds")
    print(f"GPU Memory Used: {gpu_memory_used:.2f} MB")
    print(f"RAM Used: {ram_memory_used:.2f} MB")

if __name__ == '__main__':
    main()
