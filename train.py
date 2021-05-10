from datareader import DBreader_Vimeo90k, BVIDVC, Sampler
from torch.utils.data import DataLoader
import argparse
from torchvision import transforms
import torch
from TestModule import Middlebury_other
from trainer import Trainer
from loss import Loss
import datetime
from os.path import join
from models.cdfi_adacof import CDFI_adacof

parser = argparse.ArgumentParser(description='AdaCoF-Pytorch')

# parameters
# Hardware Setting
parser.add_argument('--gpu_id', type=int, default=0)

# Directory Setting
parser.add_argument('--data_dir', type=str, default='D:\\')
parser.add_argument('--out_dir', type=str, default='./output_cdfi_train')
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--test_input', type=str, default='./test_input/middlebury_others/input')
parser.add_argument('--gt', type=str, default='./test_input/middlebury_others/gt')

# Learning Options
parser.add_argument('--epochs', type=int, default=100, help='Max Epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--patch_size', type=int, default=224, help='Patch size')

# Optimization specifications
parser.add_argument('--loss', type=str, default='1*Charb+0.01*g_Spatial')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=int, default=20, help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAMax', choices=('SGD', 'ADAM', 'RMSprop', 'ADAMax'), help='optimizer to use (SGD | ADAM | RMSprop | ADAMax)')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# Model
parser.add_argument('--kernel_size', type=int, default=11)
parser.add_argument('--dilation', type=int, default=2)

transform = transforms.Compose([transforms.ToTensor()])

def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    vimeo90k_train = DBreader_Vimeo90k(join(args.data_dir, 'vimeo_triplet'), random_crop=(args.patch_size, args.patch_size))
    bvidvc_2k = BVIDVC(join(args.data_dir, 'bvidvc'), res='2k', crop_sz=(args.patch_size,args.patch_size))
    bvidvc_1080 = BVIDVC(join(args.data_dir, 'bvidvc'), res='1080', crop_sz=(args.patch_size,args.patch_size))
    bvidvc_960 = BVIDVC(join(args.data_dir, 'bvidvc'), res='960', crop_sz=(args.patch_size,args.patch_size))
    bvidvc_480 = BVIDVC(join(args.data_dir, 'bvidvc'), res='480', crop_sz=(args.patch_size,args.patch_size))
    datasets_train = [vimeo90k_train] + 64*[bvidvc_2k] + 16*[bvidvc_1080] + 4*[bvidvc_960, bvidvc_480] 
    train_sampler = Sampler(datasets_train, iter=True)

    TestDB = Middlebury_other(args.test_input, args.gt)
    train_loader = DataLoader(dataset=train_sampler, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    model = CDFI_adacof(args).cuda()
    
    loss = Loss(args)

    start_epoch = 0
    if args.load is not None:
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    my_trainer = Trainer(args, train_loader, TestDB, model, loss, start_epoch)

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    with open(args.out_dir + '/config.txt', 'a') as f:
        f.write(now + '\n\n')
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
        f.write('\n')

    while not my_trainer.terminate():
        my_trainer.train()
        if my_trainer.current_epoch % 10 == 0:
            my_trainer.test()

    my_trainer.close()


if __name__ == "__main__":
    main()