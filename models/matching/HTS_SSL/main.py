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
from torchmeta.utils.matching import matching_log_probas, matching_loss
from model import TreeLSTM, Backbone, Classifier
from tree_data import get_tree, convert_tree_to_tensors
from utils import multi_augmented_dataset, get_dataset, get_loss, get_ssl_classes
from global_utils import Averager, Mean_confidence_interval, get_outputs_c_h, set_reproducibility

def save_model(model, backbone, classifier_all, args, tag):
    model_path = os.path.join(args.record_folder, ('_'.join([args.model_name, args.train_data, args.test_data,args.backbone,tag]) + '.pt'))
    classifier_state = []
    for i in range(len(args.pretext_tasks)):
        classifier_state.append(classifier_all[i].state_dict())
    if args.multi_gpu:
        backbone = backbone.module
    state = {
        'model': model.state_dict(),
        'backbone': backbone.state_dict(),
        'classifier': classifier_state
    }
    with open(model_path, 'wb') as f:
        torch.save(state, f)

def save_checkpoint(args, model, backbone, classifier_all, train_log, optimizer, classifier_optimizer, global_task_accout, tag):
    checkpoint_path = os.path.join(args.record_folder, ('_'.join([args.model_name, args.train_data, args.test_data, args.backbone, tag]) + '_checkpoint.pt.tar'))
    classifier_state = []
    classifier_optimizer_state = []
    for i in range(len(args.pretext_tasks)):
        classifier_state.append(classifier_all[i].state_dict())
        classifier_optimizer_state.append(classifier_optimizer[i].state_dict())
    if args.multi_gpu:
        backbone = backbone.module
    state = {
        'args': args,
        'model': model.state_dict(),
        'backbone': backbone.state_dict(),
        'classifier': classifier_state,
        'train_log': train_log,
        'val_acc': train_log['max_acc'],
        'optimizer': optimizer.state_dict(),
        'classifier_optimizer': classifier_optimizer_state,
        'global_task_accout': global_task_accout
    }
    with open(checkpoint_path, 'wb') as f:
        torch.save(state, f)

if __name__ =='__main__':
    import argparse

    parser = argparse.ArgumentParser('Matching Network')
    parser.add_argument('--model-name', type=str, default='MatchingNet_HTS_SSL', help='Name of the model.')
   
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
    parser.add_argument('--train-tasks', type=int, default= 60000, 
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
    parser.add_argument('--ssl-trade', type=float, default=0.1,
        help='Trade of ssl loss (default: 0.1).')
   
    parser.add_argument('--augment', action='store_true', default=False,
        help='Augment the training dataset (default: True).')
    parser.add_argument('--pretrain', action='store_true',
        help='If backobone network is pretrained.')
    parser.add_argument('--backbone-path', type=str, default=None,
        help='Path to the pretrained backbone.')
    
    parser.add_argument('--pretext-tasks', nargs='+', type=str, default=['rotation'], help='The type of ssl pretext-tasks.')
    parser.add_argument('--multi-gpu', action='store_true',
        help='True if use multiple GPUs. Else, use single GPU.')
    parser.add_argument('--num-workers', type=int, default=4,
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
    num_classes = get_ssl_classes(args.pretext_tasks)

    backbone = Backbone(args.backbone)
    model = TreeLSTM(args.out_channels)
    classifier_all = []
    for ssl_task in args.pretext_tasks:
        classifier_all.append(Classifier(args.out_channels, num_classes[ssl_task]))
        
    if args.use_cuda:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        num_gpus = torch.cuda.device_count()
        if args.multi_gpu:
            backbone = nn.DataParallel(backbone)

        backbone = backbone.cuda()        
        model = model.cuda()
        for i in range(len(args.pretext_tasks)):
            classifier_all[i].cuda()

    optimizer = torch.optim.Adam([{'params':backbone.parameters()}, 
                        {'params':model.parameters()}], lr=args.lr, weight_decay=0.0005)
    classifier_optimizer = []
    for i in range(len(args.pretext_tasks)):
        classifier_optimizer.append(torch.optim.Adam(classifier_all[i].parameters(), lr=args.lr, weight_decay=0.0005))

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
        for i in range(len(args.pretext_tasks)):
            classifier_all[i].load_state_dict(state['classifier'][i])

        train_log = state['train_log']
        optimizer.load_state_dict(state['optimizer'])
        for i in range(len(args.pretext_tasks)):
            classifier_optimizer[i].load_state_dict(state['classifier_optimizer'][i])
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
                for i in range(len(args.pretext_tasks)):
                    for classifier_param_group in classifier_optimizer[i].param_groups:
                        classifier_param_group['lr'] = initial_lr

            global_task_count +=args.batch_tasks

            fsl_support_inputs, fsl_support_targets = [_.cuda(non_blocking=True) for _ in train_batch['train']] if args.use_cuda else [_ for _ in train_batch['train']]
            fsl_query_inputs, fsl_query_targets = [_.cuda(non_blocking=True) for _ in train_batch['test']] if args.use_cuda else [_ for _ in train_batch['test']]

            fsl_support_inputs_in = fsl_support_inputs.view(-1, *fsl_support_inputs.shape[2:]) 
            fsl_query_inputs_in = fsl_query_inputs.view(-1, *fsl_query_inputs.shape[2:])
            
            support_shape, ssl_ways, support_inputs, ssl_support_targets = multi_augmented_dataset(args.pretext_tasks, fsl_support_inputs_in, fsl_support_inputs.shape)
            query_shape, _, query_inputs, ssl_query_targets = multi_augmented_dataset(args.pretext_tasks, fsl_query_inputs_in, fsl_query_inputs.shape)

            support_features = backbone(support_inputs)
            query_features = backbone(query_inputs)

            support_tree = get_tree(ssl_ways, support_shape, support_features)
            query_tree = get_tree(ssl_ways, query_shape, query_features)
   
            support_data = convert_tree_to_tensors(support_tree, fsl_support_inputs.device)
            query_data = convert_tree_to_tensors(query_tree, fsl_query_inputs.device)   

            support_h = model(support_data, fsl_support_inputs.device)
            query_h = model(query_data, fsl_query_inputs.device)

            support_root_mask = support_data['node_order'] == len(args.pretext_tasks)
            fsl_support_embeddings = support_h[support_root_mask,:].view(*fsl_support_inputs.shape[:2], -1)
            query_root_mask = query_data['node_order'] == len(args.pretext_tasks)
            fsl_query_embeddings = query_h[query_root_mask,:].view(*fsl_query_inputs.shape[:2], -1)

            train_loss = matching_loss(fsl_support_embeddings, fsl_support_targets,
                fsl_query_embeddings, fsl_query_targets, args.num_ways)
            
            train_ssl_loss = torch.tensor(0., device=fsl_support_inputs.device)
            for i in range(len(args.pretext_tasks)):
                classifier_all[i].train()
                ssl_support_mask = support_data['node_order'] == len(args.pretext_tasks)-(i+1)
                ssl_support_embeddings = support_h[ssl_support_mask,:].view(args.batch_tasks, int(support_shape[i+1]/args.batch_tasks), -1)
                ssl_query_mask = query_data['node_order'] == len(args.pretext_tasks)-(i+1)
                ssl_query_embeddings = query_h[ssl_query_mask,:].view(args.batch_tasks, int(query_shape[i+1]/args.batch_tasks), -1)
                ssl_embeddings = torch.cat((ssl_support_embeddings, ssl_query_embeddings), 1)
                ssl_targets = torch.cat((ssl_support_targets[i], ssl_query_targets[i]), 1)
                ssl_predictions = classifier_all[i](ssl_embeddings)
                level_loss = get_loss(ssl_predictions, ssl_targets)
                train_ssl_loss +=level_loss
            
            train_loss_main = train_loss + args.ssl_trade * train_ssl_loss
            optimizer.zero_grad()
            for i in range(len(args.pretext_tasks)):
                classifier_optimizer[i].zero_grad()
            train_loss_main.backward()
            optimizer.step()
            for i in range(len(args.pretext_tasks)):
                classifier_optimizer[i].step()
            
            with torch.no_grad():
                train_log_probas = matching_log_probas(fsl_support_embeddings, fsl_support_targets, 
                    fsl_query_embeddings, args.num_ways)
                train_predictins = torch.argmax(train_log_probas, dim=1)
                train_acc = torch.mean((train_predictins == fsl_query_targets).float())
                pbar.set_postfix(train_acc='{0:.4f}'.format(train_acc.item()))
                
                del fsl_support_inputs, fsl_support_targets, fsl_query_inputs, fsl_query_targets
                del support_shape, support_inputs, query_shape, query_inputs, ssl_support_embeddings, ssl_query_embeddings 
                del support_h, query_h, ssl_ways, support_data, query_data, support_tree, query_tree
                del fsl_support_embeddings, fsl_query_embeddings, ssl_support_targets, ssl_query_targets

                # save tensorboard writer
                if writer is not None and train_batch_i % 100 == 99:
                    writer.add_scalar('Train/train_loss', train_loss.item(), train_batch_i+1)
                    writer.add_scalar('Train/train_ssl_loss', train_ssl_loss.item(), train_batch_i+1)
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
                            
                            support_shape, ssl_ways, support_inputs, ssl_support_targets = multi_augmented_dataset(args.pretext_tasks, fsl_support_inputs_in, fsl_support_inputs.shape)
                            query_shape, _, query_inputs, ssl_query_targets = multi_augmented_dataset(args.pretext_tasks, fsl_query_inputs_in, fsl_query_inputs.shape)
            
                            support_features = backbone(support_inputs)
                            query_features = backbone(query_inputs)

                            support_tree = get_tree(ssl_ways, support_shape, support_features)
                            query_tree = get_tree(ssl_ways, query_shape, query_features)
                
                            support_data = convert_tree_to_tensors(support_tree, fsl_support_inputs.device)
                            query_data = convert_tree_to_tensors(query_tree, fsl_query_inputs.device)   

                            support_h = model(support_data, fsl_support_inputs.device)
                            query_h = model(query_data, fsl_query_inputs.device)

                            support_root_mask = support_data['node_order'] == len(args.pretext_tasks)
                            fsl_support_embeddings = support_h[support_root_mask,:].view(*fsl_support_inputs.shape[:2], -1)
                            query_root_mask = query_data['node_order'] == len(args.pretext_tasks)
                            fsl_query_embeddings = query_h[query_root_mask,:].view(*fsl_query_inputs.shape[:2], -1)

                            val_loss = matching_loss(fsl_support_embeddings, fsl_support_targets,
                                fsl_query_embeddings, fsl_query_targets, args.num_ways)
                            
                            val_ssl_loss = torch.tensor(0., device=fsl_support_inputs.device)
                            for i in range(len(args.pretext_tasks)):
                                classifier_all[i].train()
                                ssl_support_mask = support_data['node_order'] == len(args.pretext_tasks)-(i+1)
                                ssl_support_embeddings = support_h[ssl_support_mask,:].view(args.batch_tasks, int(support_shape[i+1]/args.batch_tasks), -1)
                                ssl_query_mask = query_data['node_order'] == len(args.pretext_tasks)-(i+1)
                                ssl_query_embeddings = query_h[ssl_query_mask,:].view(args.batch_tasks, int(query_shape[i+1]/args.batch_tasks), -1)
                                ssl_embeddings = torch.cat((ssl_support_embeddings, ssl_query_embeddings), 1)
                                ssl_targets = torch.cat((ssl_support_targets[i], ssl_query_targets[i]), 1)
                                ssl_predictions = classifier_all[i](ssl_embeddings)
                                level_loss = get_loss(ssl_predictions, ssl_targets)
                                val_ssl_loss +=level_loss
                            
                            val_loss_main = val_loss + args.ssl_trade * val_ssl_loss

                            val_log_probas = matching_log_probas(fsl_support_embeddings, fsl_support_targets, 
                                fsl_query_embeddings, args.num_ways)
                            val_predictins = torch.argmax(val_log_probas, dim=1)
                            val_acc = torch.mean((val_predictins == fsl_query_targets).float())
                            pbar.set_postfix(val_acc='{0:.4f}'.format(val_acc.item()))
                            del fsl_support_inputs, fsl_support_targets, fsl_query_inputs, fsl_query_targets
                            del support_shape, support_inputs, query_shape, query_inputs, ssl_support_embeddings, ssl_query_embeddings 
                            del support_h, query_h, ssl_ways, support_data, query_data, support_tree, query_tree
                            del fsl_support_embeddings, fsl_query_embeddings, ssl_support_targets, ssl_query_targets

                            val_loss_avg.add(val_loss_main.item())
                            val_acc_avg.add(val_acc.item())
                
                # record
                val_acc_mean = val_acc_avg.item()

                print('global_task_count: {}, val_acc_mean: {}'.format(str(global_task_count), str(val_acc_mean)))
                if val_acc_mean > train_log['max_acc']:
                    train_log['max_acc'] = val_acc_mean
                    train_log['max_acc_i_task'] = global_task_count
                    save_model(model, backbone, classifier_all, args, tag='max_acc')

                train_log['train_loss'].append(train_loss_main.item())
                train_log['train_acc'].append(train_acc.item())
                train_log['val_loss'].append(val_loss_avg.item())
                train_log['val_acc'].append(val_acc_mean)

                save_checkpoint(args, model, backbone, classifier_all, train_log, optimizer, classifier_optimizer, global_task_count, tag='max_acc')
                
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
    for i in range(len(args.pretext_tasks)):
        classifier_all[i].load_state_dict(state['classifier'][i])
 
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
                
                support_shape, ssl_ways, support_inputs, ssl_support_targets = multi_augmented_dataset(args.pretext_tasks, fsl_support_inputs_in, fsl_support_inputs.shape)
                query_shape, _, query_inputs, ssl_query_targets = multi_augmented_dataset(args.pretext_tasks, fsl_query_inputs_in, fsl_query_inputs.shape)

                support_features = backbone(support_inputs)
                query_features = backbone(query_inputs)

                support_tree = get_tree(ssl_ways, support_shape, support_features)
                query_tree = get_tree(ssl_ways, query_shape, query_features)
    
                support_data = convert_tree_to_tensors(support_tree, fsl_support_inputs.device)
                query_data = convert_tree_to_tensors(query_tree, fsl_query_inputs.device)   

                support_h = model(support_data, fsl_support_inputs.device)
                query_h = model(query_data, fsl_query_inputs.device)

                support_root_mask = support_data['node_order'] == len(args.pretext_tasks)
                fsl_support_embeddings = support_h[support_root_mask,:].view(*fsl_support_inputs.shape[:2], -1)
                query_root_mask = query_data['node_order'] == len(args.pretext_tasks)
                fsl_query_embeddings = query_h[query_root_mask,:].view(*fsl_query_inputs.shape[:2], -1)

                test_loss = matching_loss(fsl_support_embeddings, fsl_support_targets,
                    fsl_query_embeddings, fsl_query_targets, args.num_ways)
                
                test_ssl_loss = torch.tensor(0., device=fsl_support_inputs.device)
                for i in range(len(args.pretext_tasks)):
                    classifier_all[i].train()
                    ssl_support_mask = support_data['node_order'] == len(args.pretext_tasks)-(i+1)
                    ssl_support_embeddings = support_h[ssl_support_mask,:].view(args.batch_tasks, int(support_shape[i+1]/args.batch_tasks), -1)
                    ssl_query_mask = query_data['node_order'] == len(args.pretext_tasks)-(i+1)
                    ssl_query_embeddings = query_h[ssl_query_mask,:].view(args.batch_tasks, int(query_shape[i+1]/args.batch_tasks), -1)
                    ssl_embeddings = torch.cat((ssl_support_embeddings, ssl_query_embeddings), 1)
                    ssl_targets = torch.cat((ssl_support_targets[i], ssl_query_targets[i]), 1)
                    ssl_predictions = classifier_all[i](ssl_embeddings)
                    level_loss = get_loss(ssl_predictions, ssl_targets)
                    test_ssl_loss +=level_loss
                
                test_loss_main = test_loss + args.ssl_trade * test_ssl_loss
                test_log_probas = matching_log_probas(fsl_support_embeddings, fsl_support_targets, 
                    fsl_query_embeddings, args.num_ways)
                test_predictins = torch.argmax(test_log_probas, dim=1)
                test_acc = torch.mean((test_predictins == fsl_query_targets).float())
                pbar.set_postfix(test_acc='{0:.4f}'.format(test_acc.item()))
                
                del fsl_support_inputs, fsl_support_targets, fsl_query_inputs, fsl_query_targets
                del support_shape, support_inputs, query_shape, query_inputs, ssl_support_embeddings, ssl_query_embeddings 
                del support_h, query_h, ssl_ways, support_data, query_data, support_tree, query_tree
                del fsl_support_embeddings, fsl_query_embeddings, ssl_support_targets, ssl_query_targets

                test_loss_avg.add(test_loss_main.item())
                test_acc_avg.add(test_acc.item())

    # record
    index_values = [
        'test_acc',
        'best_i_task',    # the best_i_task of the highest val_acc
        'best_train_acc',    # the train_acc according to the best_i_task of the highest val_acc
        'best_train_loss',    # the train_loss according to the best_i_task of the highest val_acc
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










            










                
                









            
            
            

            