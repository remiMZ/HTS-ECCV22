import torch
import numpy as np
import pretext_tasks
import torch.nn.functional as F
from torchvision import transforms

def get_augmented_dataset_targets(ssl_task, inputs, inputs_shape):
    transform, ssl_ways = pretext_tasks.__dict__[ssl_task]()
    ssl_inputs = transform(inputs)
    ssl_targets = torch.arange(ssl_ways, device=inputs.device).view(1, ssl_ways)
    ssl_targets = ssl_targets.repeat(1, inputs_shape[0]*inputs_shape[1]).view(inputs_shape[0],-1)

    return ssl_ways, ssl_inputs, ssl_targets

def multi_augmented_dataset(ssl_tasks, inputs, inputs_shape):
    inputs_all = []
    inputs_all.append(inputs)
    targets_all = []
    shape = []
    shape.append(inputs.shape[0])
    ssl_ways = []
    for i, name in enumerate(ssl_tasks):
        if i == 0:
            ssl_way, ssl_inputs, ssl_targets = get_augmented_dataset_targets(name, inputs, inputs_shape)
            inputs_all.append(ssl_inputs)
            targets_all.append(ssl_targets)
            shape.append(shape[i]*ssl_way)
            ssl_ways.append(ssl_way)
            del ssl_way, ssl_inputs, ssl_targets
        else:
            ssl_way, ssl_inputs, ssl_targets = get_augmented_dataset_targets(name, inputs_all[i], inputs_shape)
            inputs_all.append(ssl_inputs)
            targets_all.append(ssl_targets)
            shape.append(shape[i]*ssl_way)
            ssl_ways.append(ssl_way)
            del ssl_way, ssl_inputs, ssl_targets

    inputs_all = torch.cat(inputs_all, 0)
    
    return shape, ssl_ways, inputs_all, targets_all
    
def get_dataset(args, dataset_name, phase):
    if dataset_name == 'cub':
        from torchmeta.datasets.helpers import cub as dataset_helper
        image_size = 84
        padding_len = 8
    elif dataset_name == 'miniimagenet':
        from torchmeta.datasets.helpers import miniimagenet as dataset_helper
        image_size = 84
        padding_len = 8
    elif dataset_name == 'tieredimagenet':
        from torchmeta.datasets.helpers import tieredimagenet as dataset_helper
        image_size = 84
        padding_len = 8
    elif dataset_name == 'cifar_fs':
        from torchmeta.datasets.helpers import cifar_fs as dataset_helper
        image_size = 32
        padding_len = 8
    else:
        raise ValueError('Non-supported Dataset.')

    if dataset_name == 'tieredimagenet':
        if args.augment and phase == 'train':
            transforms_list = [
                transforms.RandomCrop(image_size, padding=padding_len),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transforms_list = [
                transforms.ToTensor(),
            ]

    else:
        if args.augment and phase == 'train':
            transforms_list = [
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transforms_list = [
                transforms.Resize(image_size+padding_len),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]

    # pre-processing 
    if args.backbone == 'resnet12':
        transforms_list = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])

    else:
        transforms_list = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])

    # get datasets
    dataset = dataset_helper(args.data_folder,
                            shots=args.num_shots,
                            ways=args.num_ways,
                            shuffle=(phase == 'train'),
                            test_shots=args.test_shots,
                            meta_split=phase,
                            download=args.download,
                            transform=transforms_list)

    return dataset

def get_loss(predictions, targets):
    predictions = predictions.view(-1,predictions.shape[-1])
    targets = targets.view(-1)
    return F.cross_entropy(predictions, targets)

def get_one_hot_targets(targets, num_classes):
    batch_size, num_examples = targets.size(0), targets.size(-1)
    one_hot_targets = torch.zeros(batch_size, num_examples, num_classes, device=targets.device)
    targets.unsqueeze_(2)
    ones = torch.ones(batch_size, num_examples, 1, device=targets.device)
    one_hot_targets.scatter_add_(2, targets, ones)
    return one_hot_targets

def get_ssl_classes(ssl_tasks):
    ssl_classes = {}
    for ssl_task in ssl_tasks:
        _, ssl_ways = pretext_tasks.__dict__[ssl_task]()
        ssl_classes[ssl_task] = ssl_ways
    
    return ssl_classes








