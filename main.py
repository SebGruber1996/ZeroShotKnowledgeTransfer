import torch
from solver import *
from utils.helpers import *
from models.selector import *

def main(args):
    """
    Run the experiment as many times as there
    are seeds given, and write the mean and std
    to as an empty file's name for cleaner logging
    """

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    teacher_model = select_model(
        dataset=args.dataset,
        model_name=args.teacher_architecture,
        pretrained=True,
        pretrained_models_path=args.pretrained_models_path
    ).to(args.device)

    student_model = select_model(
        dataset=args.dataset,
        model_name=args.student_architecture,
        pretrained=False,
        pretrained_models_path=args.pretrained_models_path
    ).to(args.device)

    generator_model = Generator(
        z_dim=args.z_dim,
        out_channels=args.x_channels,
        out_dim=args.x_dim
    ).to(device=args.device)

    if len(args.seeds) > 1:
        test_accs = []
        base_name = args.experiment_name
        for seed in args.seeds:
            print('\n\n----------- SEED {} -----------\n\n'.format(seed))
            set_torch_seeds(seed)
            args.experiment_name = os.path.join(base_name, base_name+'_seed'+str(seed))
            solver = ZeroShotKTSolver(args, teacher_model, student_model, generator_model)
            test_acc = solver.run()
            test_accs.append(test_acc)
        mu = np.mean(test_accs)
        sigma = np.std(test_accs)
        print('\n\nFINAL MEAN TEST ACC: {:02.2f} +/- {:02.2f}'.format(mu, sigma))
        file_name = "mean_final_test_acc_{:02.2f}_pm_{:02.2f}".format(mu, sigma)
        with open(os.path.join(args.log_directory_path, base_name, file_name), 'w+') as f:
            f.write("NA")
    else:
        set_torch_seeds(args.seeds[0])
        solver = ZeroShotKTSolver(args, teacher_model, student_model, generator_model)
        test_acc = solver.run()
        print('\n\nFINAL TEST ACC RATE: {:02.2f}'.format(test_acc))
        file_name = "final_test_acc_{:02.2f}".format(test_acc)
        with open(os.path.join(args.log_directory_path, args.experiment_name, file_name), 'w+') as f:
            f.write("NA")

if __name__ == "__main__":
    import argparse
    import numpy as np
    from utils.helpers import str2bool
    print('Running...')

    parser = argparse.ArgumentParser(description='Welcome to the future')

    parser.add_argument('--dataset', type=str, default='SVHN', choices=['SVHN', 'CIFAR10', "Omniglot"])
    parser.add_argument('--total_n_pseudo_batches', type=float, default=1000)
    parser.add_argument('--n_generator_iter', type=int, default=1, help='per batch, for few and zero shot')
    parser.add_argument('--n_student_iter', type=int, default=7, help='per batch, for few and zero shot')
    parser.add_argument('--batch_size', type=int, default=128, help='for few and zero shot')
    parser.add_argument('--z_dim', type=int, default=100, help='for few and zero shot')
    parser.add_argument('--x_channels', type=int, default=3, help='image data channel amount; probably 3 or 1')
    parser.add_argument('--x_dim', type=int, default=32, help='image data x- and y-axis size; must be multiple of 4')
    parser.add_argument('--student_learning_rate', type=float, default=2e-3)
    parser.add_argument('--generator_learning_rate', type=float, default=1e-3)
    parser.add_argument('--teacher_architecture', type=str, default='LeNet')
    parser.add_argument('--student_architecture', type=str, default='LeNet')
    parser.add_argument('--KL_temperature', type=float, default=1, help='>1 to smooth probabilities in divergence loss, or <1 to sharpen them')
    parser.add_argument('--AT_beta', type=float, default=250, help='beta coefficient for AT loss')

    parser.add_argument('--pretrained_models_path', nargs="?", type=str, default='/home/paul/Pretrained/')
    parser.add_argument('--datasets_path', type=str, default="/home/paul/Datasets/Pytorch/")
    parser.add_argument('--log_directory_path', type=str, default="/home/paul/git/ZeroShotKnowledgeTransfer/logs/")
    parser.add_argument('--save_final_model', type=str2bool, default=0)
    parser.add_argument('--save_n_checkpoints', type=int, default=0)
    parser.add_argument('--save_model_path', type=str, default="/home/paul/git/FewShotKT/logs/")
    parser.add_argument('--seeds', nargs='*', type=int, default=[0, 1])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--device', type=str, default="cpu")
    args = parser.parse_args()

    args.total_n_pseudo_batches = int(args.total_n_pseudo_batches)
    if args.AT_beta > 0: assert args.student_architecture[:3] in args.teacher_architecture
    args.log_freq = max(1, int(args.total_n_pseudo_batches / 100))
    args.dataset_path = os.path.join(args.datasets_path, args.dataset)
    args.device = torch.device(args.device)
    args.experiment_name = 'ZeroShotKnowledgeTransfer_{}_{}_{}_gi{}_si{}_zd{}_plr{}_slr{}_bs{}_T{}_beta{}'.format(args.dataset, args.teacher_architecture,  args.student_architecture, args.n_generator_iter, args.n_student_iter, args.z_dim, args.generator_learning_rate, args.student_learning_rate, args.batch_size, args.KL_temperature, args.AT_beta)

    print('\nTotal data batches: {}'.format(args.total_n_pseudo_batches))
    print('Logging results every {} batch'.format(args.log_freq))
    print('\nRunning on device: {}'.format(args.device))

    main(args)
