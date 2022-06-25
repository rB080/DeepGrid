import torch
from torch.nn import functional as F
from torch import nn
from tqdm import tqdm
from Metrics import losses

def train_one_epoch(epoch,
                    model,
                    train_loader,
                    train_set_size,
                    optimizer,
                    device,
                    loss_weights=[1,1,1],
                    log_interval=100):

    model.train()
    train_loss = 0
    num_samples = 0
    train_set = tqdm(enumerate(train_loader), total=train_set_size)
    l_func = losses.loss_function()
    for batch_idx, (data, labels) in train_set:
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(data)
        pred = F.sigmoid(logits)
        loss = l_func.forward(labels, pred, loss_weights)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print('\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                       100. * batch_idx / len(train_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / num_samples))
    return train_loss / num_samples

def test_one_epoch(epoch,
                    model,
                    test_loader,
                    test_set_size,
                    device,
                    loss_weights=[1,1,1]):

    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        test_set = tqdm(enumerate(test_loader), total=test_set_size)
        l_func = losses.loss_function()
        for i, (data, labels) in test_set:
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            pred = F.sigmoid(logits)
            loss += l_func.forward(labels, pred, loss_weights).item()
            num_samples += len(data)

    print('======> Test set loss for epoch '+str(epoch)+': {:.4f}'.format(
        loss / num_samples))
    return loss / num_samples