import torch
import numpy as np
import pretext_tasks
from torchvision import transforms

def get_augmented_dataset(ssl_task, inputs):
    transform, ssl_way = pretext_tasks.__dict__[ssl_task]()
    ssl_inputs = transform(inputs)
    return ssl_way, ssl_inputs

def multi_augmented_dataset(ssl_tasks, inputs):
    inputs_all = []
    inputs_all.append(inputs)
    shape = []
    shape.append(inputs.shape[0])
    ssl_ways = []
    for i, name in enumerate(ssl_tasks):
        if i == 0:
            ssl_way, ssl_inputs = get_augmented_dataset(name, inputs)
            inputs_all.append(ssl_inputs)
            shape.append(shape[i]*ssl_way)
            ssl_ways.append(ssl_way)
            del ssl_way, ssl_inputs
        else:
            ssl_way, ssl_inputs = get_augmented_dataset(name, inputs_all[i])
            inputs_all.append(ssl_inputs)
            shape.append(shape[i]*ssl_way)
            ssl_ways.append(ssl_way)
            del ssl_way, ssl_inputs

    inputs_all = torch.cat(inputs_all, 0)
    
    return shape, ssl_ways, inputs_all

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

    dataset = dataset_helper(args.data_folder,
                            shots=args.num_shots,
                            ways=args.num_ways,
                            shuffle=(phase == 'train'),
                            test_shots=args.test_shots,
                            meta_split=phase,
                            download=args.download,
                            transform=transforms_list)

    return dataset


def get_accuracy(prototypes, embeddings, targets):
    sq_distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)
    _, predictions = torch.min(sq_distances, dim=-1)
    return torch.mean(predictions.eq(targets).float())










