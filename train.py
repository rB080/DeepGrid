from Dataloaders import isic
from Models import deepgrid
import train_utils as Tr
import torch
import argparse
import os
import json

loaders = {'isic2017':isic}

models = {'deepgrid':deepgrid}

def get_args_parser():
    parser = argparse.ArgumentParser()
    #Training Hyperparameters
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--BCE_weight', default=1, type=float)
    parser.add_argument('--IOU_weight', default=1, type=float)
    parser.add_argument('--DSC_weight', default=1, type=float)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lr_red1', default=50, type=int)
    parser.add_argument('--lr_red2', default=80, type=int)
    parser.add_argument('--lr_red_factor', default=10, type=int)

    #Model and input
    parser.add_argument('--model', default='deepgrid', type=str, help='Name of model to train')
    parser.add_argument('--d1', default=4, type=int, help='Streams (if applicable)')
    parser.add_argument('--d2', default=3, type=int, help='Columns (if applicable)')
    parser.add_argument('--input_dim', default=128, type=int, help='Input Dimension of Image')

    #Paths
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_save_dir', type=str, required=True)
    parser.add_argument('--model_load_path', type=str, required=True)
    parser.add_argument('--log_path', type=str, required=True)

    #Flags
    parser.add_argument('--resume', default='n', type=str, help='Resume Training: [y/n]')
    parser.add_argument('--val', default='n', type=str, help='Validation: [y/n]')

    args = parser.parse_args()
    return args

#Get Root Paths
args = get_args_parser()
cont = args.resume
assert cont in ['y', 'Y', 'n', 'N'], 'Invalid resume parameter. Can only be y/n.'

#Init Training Device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Training device:', device)

#Init Models
if cont == 'n' or cont == 'N':
    model_name = args.model + '_' + args.dataset + '.pth'
    model_save_path = os.path.join(args.model_save_dir, model_name)
    model = models[args.model].give_model(device, args.d1, args.d2)
    log = [[], []]
    
else:
    print('Loading pre-trained model to resume training')
    model_save_path = args.model_load_path
    model = models[args.model].give_model(device, args.d1, args.d2, args.model_load_path)
    with open(args.log_path, "rb") as file:
        log = json.load(file)


#Load Data
v = args.val
assert v in ['y', 'Y', 'n', 'N'], 'Invalid validation parameter. Can only be y/n.'
if v == 'n' or v == 'N':
    train_loader, train_set_size = loaders[args.dataset].getTrainLoader(args.dataset_path,
                                                                    args.batch_size,
                                                                    (args.input_dim, args.input_dim))
else:
    train_loader, train_set_size = loaders[args.dataset].getValLoader(args.dataset_path,
                                                                    args.batch_size,
                                                                    (args.input_dim, args.input_dim))

test_loader, test_set_size = loaders[args.dataset].getTestLoader(args.dataset_path,
                                                            args.batch_size,
                                                            (args.input_dim, args.input_dim))

#Training Hyperparameters
learning_rate = args.lr
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.0001)
epochs = args.epochs
W = [args.BCE_weight, args.IOU_weight, args.DSC_weight]
L1, L2, factor = args.lr_red1, args.lr_red2, args.lr_red_factor

#Training
print('Beginning Training of', args.model, 'for', args.epochs, 'epochs.')
print('Training being resumed:', args.resume)
print('Validation training:', args.val)
for epoch in range(1, epochs):
    print('Epoch:', epoch)
    print('Training phase---------------------')
    trl = Tr.train_one_epoch(epoch, model, train_loader, train_set_size, optimizer, device, W)
    print('-----------------------------------')
    print('Testing phase---------------------')
    tel = Tr.test_one_epoch(epoch, model, test_loader, test_set_size, device, W)
    print('-----------------------------------')

    #Learning rate change
    if epoch == L1 or epoch == L2:
        print('Learning rate update:', learning_rate, '=>', learning_rate / factor)
        learning_rate /= factor
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.0001)

    #Save Model after Training epoch
    torch.save(model.state_dict(), model_save_path)
    print('Model Saved')

    #Log training and testing losses
    log[0].append(trl)
    log[1].append(tel)
    with open(args.log_path, "w") as file:
        file.write(json.dumps(log))
print('Training complete. Check logfile for training analysis.')
