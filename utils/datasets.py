import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def get_test_loader(args):
    if args.dataset == 'SVHN':
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        test_dataset = datasets.SVHN(args.dataset_path, split='test', download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=args.batch_size,
            shuffle=False, drop_last=False, num_workers=args.workers,
            pin_memory=False)


    elif args.dataset == 'CIFAR10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        test_dataset = datasets.CIFAR10(args.dataset_path, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=args.batch_size,
            shuffle=False, drop_last=False, num_workers=args.workers,
            pin_memory=False)


    elif args.dataset == "Omniglot":
        task_file = args.pretrained_models_path.replace("model", "task")
        test_dataset = torch.load(task_file)
        xs = torch.cat([test_dataset[0], test_dataset[2]])
        ys = torch.cat([test_dataset[1], test_dataset[3]])
        #test_loader = zip(xs, ys)
        # list with 1 element
        test_loader = [(xs, ys)]

    else:
        raise NotImplementedError

    return test_loader
