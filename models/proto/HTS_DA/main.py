import os
import time
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from pandas import DataFrame
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
from model import TreeLSTM, Backbone
from tree_data import get_tree, convert_tree_to_tensors
from utils import multi_augmented_dataset, get_dataset, get_accuracy
from global_utils import Averager, Mean_confidence_interval, get_outputs_c_h, set_reproducibility

def save_model(model, backbone, args, tag):
    model_path = os.path.join(args.record_folder, ('_'.join([args.model_name, args.train_data, args.test_data,args.backbone,tag]) + '.pt'))
    if args.multi_gpu:
        backbone = backbone.module
    state = {
        'model': model.state_dict(),
        'backbone': backbone.state_dict()
    }
    with open(model_path, 'wb') as f:
        torch.save(state, f)

def save_checkpoint(args, model, backbone, train_log, optimizer, all_task_accout, tag):
    checkpoint_path = os.path.join(args.record_folder, ('_'.join([args.model_name, args.train_data, args.test_data, args.backbone, tag]) + '_checkpoint.pt.tar'))
    if args.multi_gpu:
        backbone = backbone.module
    state = {
        'args': args,
        'model': model.state_dict(),
        'backbone': backbone.state_dict(),
        'train_log': train_log,
        'val_acc': train_log['max_acc'],
        'optimizer': optimizer.state_dict(),
        'all_task_accout': all_task_accout
    }
    with open(checkpoint_path, 'wb') as f:
        torch.save(state, f)

if __name__ =='__main__':
    import argparse

    parser = argparse.ArgumentParser('Prototypical Network')
    parser.add_argument('--model-name', type=str, default='ProtoNet_HTS_DA', help='Name of the model.')
   
    parser.add_argument('--data-folder', type=str, default='./datasets',
        help='Path to the folder the data is downloaded to.')  
    parser.add_argument('--train-data', type=str, default= 'cifar_fs', choices=['cub', 'miniimagenet',
        'tieredimagenet', 'cifar_fs'], help='Name of the dataset.')
    parser.add_argument('--test-data', type=str, default= 'cifar_fs', choices=['cub', 'miniimagenet',
        'tieredimagenet', 'cifar_fs'], help='Name of the dataset.')
    parser.add_argument('--num-shots', type=int, default=1, choices=[1, 5],
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5, 
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--test-shots', type=int, default=15, 
        help='Number of examples per class (k in "k-shot", default: 15).')
    parser.add_argument('--backbone', type=str, default='conv4', choices=['conv4','resnet12','resnet12_wide'], help='The type of model backbone.')
    
    parser.add_argument('--batch-tasks', type=int, default=4,
        help='Number of tasks in a mini-batch of tasks (default: 4).')
    parser.add_argument('--train-tasks', type=int, default=60000,  
        help='Number of tasks the model is trained over (default: 60000).')
    parser.add_argument('--val-tasks', type=int, default=600, 
        help='Number of tasks the model network is validated over (default: 600). ')
    parser.add_argument('--test-tasks', type=int, default=10000, 
        help='Number of tasks the model network is tested over (default: 10000). The final results will be the average of these batches.')
    parser.add_argument('--validation-tasks', type=int, default=1000, 
        help='Number of tasks for each validation (default: 1000).')
 
    parser.add_argument('--lr', type=float, default=0.001,
        help='Initial learning rate (default: 0.001).')
    parser.add_argument('--schedule', type=int, nargs='+', default=[15000, 30000, 45000, 60000], 
        help='Decrease learning rate at these number of tasks.')
    parser.add_argument('--gamma', type=float, default=0.1,
        help='Learning rate decreasing ratio (default: 0.1).')
   
    parser.add_argument('--augment', action='store_true', 
        help='Augment the training dataset (default: True).')
    parser.add_argument('--pretrain', action='store_true',
        help='If backobone network is pretrained.')
    parser.add_argument('--backbone-path', type=str, default=None,
        help='Path to the pretrained backbone.')
    
    parser.add_argument('--pretext-tasks', nargs='+', type=str, default=['rotation2'], 
        help='The type of ssl pretext-tasks.')
    parser.add_argument('--multi-gpu', action='store_true',
        help='True if use multiple GPUs. Else, use single GPU.')
    parser.add_argument('--num-workers', type=int, default=6,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--download', action='store_true',
        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', type=bool, default=True,
        help='Use CUDA if available.')
   
    parser.add_argument('--resume', action='store_true', 
        help='Continue from baseline trained model with largest epoch.')
    parser.add_argument('--resume-folder', type=str, default=None,
        help='Path to the folder the resume is saved to.')
    
    parser.add_argument('--reproduce', action='store_true',
        help='If set reproducibility.')
    parser.add_argument('--seed', type=int, default=0,
        help='Random seed if reproduce.')    

    args = parser.parse_args()

    # make folder and tensorboard writer to save model and results
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    args.record_folder = './{}_{}_{}_{}_{}way_{}shot_{}'.format(args.train_data, args.test_data, args.backbone, args.pretext_tasks, str(args.num_ways), str(args.num_shots), cur_time)
    os.makedirs(args.record_folder, exist_ok=True)
    
    if args.use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    elif args.use_cuda:
        raise RuntimeError('You are using GPU mode, but GPUs are not available!')
    
    # set reproducibility
    if args.reproduce:
        set_reproducibility(args.seed)    
    
    # construct model and optimizer
    args.image_len = 32 if args.train_data == 'cifar_fs' else 84
    args.out_channels, _ = get_outputs_c_h(args.backbone, args.image_len)
    
    backbone = Backbone(args.backbone)
    model = TreeLSTM(args.out_channels)

    if args.use_cuda:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        num_gpus = torch.cuda.device_count()
        if args.multi_gpu:
            backbone = nn.DataParallel(backbone)

        backbone = backbone.cuda()        
        model = model.cuda()
        
    optimizer = torch.optim.Adam([{'params':backbone.parameters()}, 
                        {'params':model.parameters()}], lr=args.lr, weight_decay=0.0005)

    # training from the checkpoint
    if args.resume and args.resume_folder is not None:
        # load checkpoint
        checkpoint_path = os.path.join(args.resume_folder, ('_'.join([args.model_name, args.train_data, args.test_data, args.backbone, 'max_acc']) + '_checkpoint.pt.tar'))    
        state = torch.load(checkpoint_path)
        if args.multi_gpu:
            backbone.module.load_state_dict(state['backbone'])
        else:
            backbone.load_state_dict(state['backbone'])
        
        model.load_state_dict(state['model'])

        train_log = state['train_log']
        optimizer.load_state_dict(state['optimizer'])
        initial_lr = optimizer.param_groups[0]['lr']
        global_task_count = state['global_task_accout']

        print('global_task_count: {}, initial_lr: {}'.format(str(global_task_count), str(initial_lr)))
    # training from scratch
    else:
        train_log = {}
        train_log['args'] = vars(args)
        train_log['train_loss'] = []
        train_log['train_acc'] = []
        train_log['val_loss'] = []
        train_log['val_acc'] = []
        train_log['max_acc'] = 0.0
        train_log['max_acc_i_task'] = 0
        initial_lr = args.lr
        global_task_count = 0
 
        if args.pretrain and args.backbone_path is not None:
            backbone_state = torch.load(args.backbone_path)
            if args.multi_gpu:
                backbone.module.load_state_dict(backbone_state)
            else:
                backbone.load_state_dict(backbone_state)

    # save the args into .json file
    with open(os.path.join(args.record_folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    writer = SummaryWriter(os.path.join(args.record_folder,'tensorboard_log'))
    
    """get train datasets"""
    train_dataset = get_dataset(args, dataset_name=args.train_data, phase='train') 
    train_dataloader = BatchMetaDataLoader(train_dataset, batch_size=args.batch_tasks,
        shuffle=True, num_workers=args.num_workers) 
    """get validation datasets"""
    val_dataset = val_dataset = get_dataset(args, dataset_name=args.test_data, phase='val')
    val_dataloader = BatchMetaDataLoader(val_dataset, batch_size=args.batch_tasks,
        shuffle=True, num_workers=args.num_workers)

    # training
    with tqdm(train_dataloader, total=int(args.train_tasks/args.batch_tasks), initial=int(global_task_count/args.batch_tasks)) as pbar:
        for train_batch_i, train_batch in enumerate(pbar):
            
            if train_batch_i >= args.train_tasks/args.batch_tasks:
                break
            
            # training
            model.train()
            backbone.train()
            # chech if lr should decrease as in schedule
            if (train_batch_i * args.batch_tasks) in args.schedule:
                initial_lr *=args.gamma
                for param_group in optimizer.param_groups:
                    param_group['lr'] = initial_lr

            global_task_count +=args.batch_tasks

            fsl_support_inputs, fsl_support_targets = [_.cuda(non_blocking=True) for _ in train_batch['train']] if args.use_cuda else [_ for _ in train_batch['train']]
            fsl_query_inputs, fsl_query_targets = [_.cuda(non_blocking=True) for _ in train_batch['test']] if args.use_cuda else [_ for _ in train_batch['test']]

            fsl_support_inputs_in = fsl_support_inputs.view(-1, *fsl_support_inputs.shape[2:]) 
            fsl_query_inputs_in = fsl_query_inputs.view(-1, *fsl_query_inputs.shape[2:])
            
            support_shape, ssl_ways, support_inputs = multi_augmented_dataset(args.pretext_tasks, fsl_support_inputs_in)
            query_shape, _, query_inputs = multi_augmented_dataset(args.pretext_tasks, fsl_query_inputs_in)

            support_features = backbone(support_inputs)
            query_features = backbone(query_inputs)

            support_tree = get_tree(ssl_ways, support_shape, support_features)
            query_tree = get_tree(ssl_ways, query_shape, query_features)
   
            support_data = convert_tree_to_tensors(support_tree, fsl_support_inputs.device)
            query_data = convert_tree_to_tensors(query_tree, fsl_query_inputs.device)   

            support_h = model(support_data, fsl_support_inputs.device)
            query_h = model(query_data, fsl_query_inputs.device)
                        
            support_root_mask = support_data['node_order'] == len(args.pretext_tasks)
            support_embeddings = support_h[support_root_mask,:].view(*fsl_support_inputs.shape[:2], -1)
            query_root_mask = query_data['node_order'] == len(args.pretext_tasks)
            query_embeddings = query_h[query_root_mask,:].view(*fsl_query_inputs.shape[:2], -1)

            prototypes = get_prototypes(support_embeddings, fsl_support_targets,
                train_dataset.num_classes_per_task)
            train_loss = prototypical_loss(prototypes, query_embeddings, fsl_query_targets)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_acc = get_accuracy(prototypes, query_embeddings, fsl_query_targets)
                pbar.set_postfix(train_acc='{0:.4f}'.format(train_acc.item()))
                del fsl_support_inputs, fsl_support_targets, fsl_query_inputs, fsl_query_targets
                del support_shape, support_inputs, query_shape, query_inputs, ssl_ways
                del support_features, query_features
                del support_data, query_data, support_tree, query_tree
                del support_embeddings, query_embeddings

                # save tensorboard writer
                if writer is not None and train_batch_i % 100 == 99:
                    writer.add_scalar('Train/train_loss', train_loss.item(), train_batch_i+1)
                    writer.add_scalar('Train/train_acc', train_acc.item(), train_batch_i+1) 
                      
            # validation
            if global_task_count % args.validation_tasks == 0:
                val_loss_avg = Averager()
                val_acc_avg = Mean_confidence_interval()

                model.eval()
                backbone.eval()
                with torch.no_grad():
                    with tqdm(val_dataloader, total=int(args.val_tasks/args.batch_tasks)) as pbar:
                        for val_batch_i, val_batch in enumerate(pbar, 1):

                            if val_batch_i > (args.val_tasks / args.batch_tasks):
                                break
                            
                            fsl_support_inputs, fsl_support_targets = [_.cuda(non_blocking=True) for _ in val_batch['train']] if args.use_cuda else [_ for _ in val_batch['train']]
                            fsl_query_inputs, fsl_query_targets = [_.cuda(non_blocking=True) for _ in val_batch['test']] if args.use_cuda else [_ for _ in val_batch['test']]

                            fsl_support_inputs_in = fsl_support_inputs.view(-1, *fsl_support_inputs.shape[2:]) 
                            fsl_query_inputs_in = fsl_query_inputs.view(-1, *fsl_query_inputs.shape[2:])
                            
                            support_shape, ssl_ways, support_inputs = multi_augmented_dataset(args.pretext_tasks, fsl_support_inputs_in)
                            query_shape, _, query_inputs = multi_augmented_dataset(args.pretext_tasks, fsl_query_inputs_in)

                            support_features = backbone(support_inputs)
                            query_features = backbone(query_inputs)

                            support_tree = get_tree(ssl_ways, support_shape, support_features)
                            query_tree = get_tree(ssl_ways, query_shape, query_features)
                
                            support_data = convert_tree_to_tensors(support_tree, fsl_support_inputs.device)
                            query_data = convert_tree_to_tensors(query_tree, fsl_query_inputs.device)   

                            support_h = model(support_data, fsl_support_inputs.device)
                            query_h = model(query_data, fsl_query_inputs.device)
                    
                            support_root_mask = support_data['node_order'] == len(args.pretext_tasks)
                            support_embeddings = support_h[support_root_mask,:].view(*fsl_support_inputs.shape[:2], -1)
                            query_root_mask = query_data['node_order'] == len(args.pretext_tasks)
                            query_embeddings = query_h[query_root_mask,:].view(*fsl_query_inputs.shape[:2], -1)

                            prototypes = get_prototypes(support_embeddings, fsl_support_targets,
                                val_dataset.num_classes_per_task)
                            val_loss = prototypical_loss(prototypes, query_embeddings, fsl_query_targets)

                            val_acc = get_accuracy(prototypes, query_embeddings, fsl_query_targets)

                            pbar.set_postfix(val_acc='{0:.4f}'.format(val_acc.item()))
                            del fsl_support_inputs, fsl_support_targets, fsl_query_inputs, fsl_query_targets
                            del support_shape, support_inputs, query_shape, query_inputs, ssl_ways
                            del support_features, query_features
                            del support_data, query_data, support_tree, query_tree
                            del support_embeddings, query_embeddings

                            val_loss_avg.add(val_loss.item())
                            val_acc_avg.add(val_acc.item())
                
                # record
                val_acc_mean = val_acc_avg.item()

                print('global_task_count: {}, val_acc_mean: {}'.format(str(global_task_count), str(val_acc_mean)))
                if val_acc_mean > train_log['max_acc']:
                    train_log['max_acc'] = val_acc_mean
                    train_log['max_acc_i_task'] = global_task_count
                    save_model(model, backbone, args, tag='max_acc')

                train_log['train_loss'].append(train_loss.item())
                train_log['train_acc'].append(train_acc.item())
                train_log['val_loss'].append(val_loss_avg.item())
                train_log['val_acc'].append(val_acc_mean)

                save_checkpoint(args, model, backbone, train_log, optimizer, global_task_count, tag='max_acc')
                
                # save tensorboard writer
                if writer is not None:
                    writer.add_scalar('Validation/val_loss_avg', val_loss_avg.item(), train_batch_i+1)
                    writer.add_scalar('Validation/val_acc_avg', val_acc_avg.item(), train_batch_i+1)
                del val_loss_avg, val_acc_avg   

    # testing
    """get test datasets"""
    test_dataset = get_dataset(args, dataset_name=args.test_data, phase='test') 
    test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=args.batch_tasks,
        shuffle=True, num_workers=args.num_workers)  
    
    test_loss_avg = Averager()
    test_acc_avg = Mean_confidence_interval()
    
    model_path = os.path.join(args.record_folder, ('_'.join([args.model_name, args.train_data, args.test_data, args.backbone, 'max_acc']) + '.pt'))
    state = torch.load(model_path)
    if args.multi_gpu:
        backbone.module.load_state_dict(state['backbone'])
    else:
        backbone.load_state_dict(state['backbone'])
    
    model.load_state_dict(state['model'])
 
    model.eval()
    backbone.eval()
    with torch.no_grad():
        with tqdm(test_dataloader, total=int(args.test_tasks/args.batch_tasks)) as pbar:
            for test_batch_i, test_batch in enumerate(pbar, 1): 

                if test_batch_i > (args.test_tasks / args.batch_tasks):
                    break
                
                fsl_support_inputs, fsl_support_targets = [_.cuda(non_blocking=True) for _ in test_batch['train']] if args.use_cuda else [_ for _ in test_batch['train']]
                fsl_query_inputs, fsl_query_targets = [_.cuda(non_blocking=True) for _ in test_batch['test']] if args.use_cuda else [_ for _ in test_batch['test']]
                
                fsl_support_inputs_in = fsl_support_inputs.view(-1, *fsl_support_inputs.shape[2:]) 
                fsl_query_inputs_in = fsl_query_inputs.view(-1, *fsl_query_inputs.shape[2:])
                
                support_shape, ssl_ways, support_inputs = multi_augmented_dataset(args.pretext_tasks, fsl_support_inputs_in)
                query_shape, _, query_inputs = multi_augmented_dataset(args.pretext_tasks, fsl_query_inputs_in)

                support_features = backbone(support_inputs)
                query_features = backbone(query_inputs)

                support_tree = get_tree(ssl_ways, support_shape, support_features)
                query_tree = get_tree(ssl_ways, query_shape, query_features)
    
                support_data = convert_tree_to_tensors(support_tree, fsl_support_inputs.device)
                query_data = convert_tree_to_tensors(query_tree, fsl_query_inputs.device)   

                support_h = model(support_data, fsl_support_inputs.device)
                query_h = model(query_data, fsl_query_inputs.device)
        
                support_root_mask = support_data['node_order'] == len(args.pretext_tasks)
                support_embeddings = support_h[support_root_mask,:].view(*fsl_support_inputs.shape[:2], -1)
                query_root_mask = query_data['node_order'] == len(args.pretext_tasks)
                query_embeddings = query_h[query_root_mask,:].view(*fsl_query_inputs.shape[:2], -1)

                prototypes = get_prototypes(support_embeddings, fsl_support_targets, test_dataset.num_classes_per_task)
                
                test_loss = prototypical_loss(prototypes, query_embeddings, fsl_query_targets)
                test_acc = get_accuracy(prototypes, query_embeddings, fsl_query_targets)

                pbar.set_postfix(test_acc='{0:.4f}'.format(test_acc.item()))
                del fsl_support_inputs, fsl_support_targets, fsl_query_inputs, fsl_query_targets
                del support_shape, support_inputs, query_shape, query_inputs, ssl_ways
                del support_features, query_features
                del support_data, query_data, support_tree, query_tree
                del support_embeddings, query_embeddings

                test_loss_avg.add(test_loss.item())
                test_acc_avg.add(test_acc.item())

    print("Test Acc:", test_acc_avg.item(return_str=True))
    # record
    index_values = [
        'test_acc',
        'best_i_task',    
        'best_train_acc',    
        'best_train_loss',    
        'best_val_acc',
        'best_val_loss'
    ]
    best_index = int(train_log['max_acc_i_task'] / args.validation_tasks) - 1
    test_record = {}
    test_record_data = [
        test_acc_avg.item(return_str=True),
        str(train_log['max_acc_i_task']),
        str(train_log['train_acc'][best_index]),
        str(train_log['train_loss'][best_index]),
        str(train_log['max_acc']),
        str(train_log['val_loss'][best_index]),
    ]
    test_record[args.record_folder] = test_record_data
    test_record_file = os.path.join(args.record_folder, 'record_{}_{}_{}way_{}shot.csv'.format(args.train_data, args.test_data, str(args.num_ways), str(args.num_shots)))
    DataFrame(test_record, index=index_values).to_csv(test_record_file)

         