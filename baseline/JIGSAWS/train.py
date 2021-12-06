from .dataset import video_train_loader, video_test_loader
import argparse
import torch
import time
import copy
import torch.nn as nn
import torch.optim as optim
from .model.Models import SDConv
from .model.Optim import ScheduledOptim
import numpy as np
import sklearn.metrics as metrics


# visualise the video segmentation result
def eval_pred(model, test_loader, device, n_class):
    was_training = model.training
    model.eval()

    current_video = 0
    overall_acc = 0.0
    f1_score = 0

    ground_truth = []
    predicted_result = []

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            _, input_len, _ = inputs.size()
            idx = torch.arange(1, input_len + 1)
            attn = torch.arange(1, (input_len // 4) + 1)
            idx = idx.unsqueeze(0).to(device)
            attn = attn.unsqueeze(0).to(device)
            inputs = inputs.to(torch.float32)
            idx = idx.to(torch.float32)
            outputs = model(inputs, idx, attn)
            outputs = torch.squeeze(outputs).to(torch.float32)
            labels = torch.squeeze(labels).to(torch.float32)
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)

            running_corrects = (predicted == labels).sum()
            running_acc = running_corrects.double() / labels.size(0)
            overall_acc += running_acc

            # transfer the torch tensor to numpy array first, then normalized to p[0,1]
            predicted = predicted.cpu().numpy()
            labels = labels.cpu().numpy()

            ground_truth.extend(np.array(labels.data))
            predicted_result.extend(np.array(predicted.data))

            f1_score += metrics.f1_score(y_true=labels, y_pred=predicted, average='micro')

            current_video += 1

        eval_acc = overall_acc / len(test_loader)
        print('Evaluation Accuray: {:.4f}'.format(eval_acc))
        print('F1: {:.4f}'.format(f1_score/len(test_loader)))


        model.train(mode=was_training)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for i, data in enumerate(train_loader, 0):
        # Iterate over data
        inputs, labels = data[0].to(device), data[1].to(device)
        _, input_len, _ = inputs.size()
        idx = torch.arange(1, input_len + 1)
        attn = torch.arange(1, (input_len // 4) + 1)
        idx = idx.unsqueeze(0).to(device)
        attn = attn.unsqueeze(0).to(device)
        inputs = inputs.to(torch.float32)
        idx = idx.to(torch.float32)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        outputs = model(inputs, idx, attn)
        outputs = torch.squeeze(outputs).to(torch.float32)
        labels = torch.squeeze(labels).to(torch.float32)
        _, predicted = torch.max(outputs, 1)

        loss = criterion(torch.softmax(outputs,1), labels)

        loss.backward()
        optimizer.step_and_update_lr()
        running_loss += loss.item()
        # count the correct result
        _, labels = torch.max(labels, 1)
        running_corrects = (predicted == labels).sum()
        # calculate the accuray each step and accumulate
        running_acc += running_corrects.double() / labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = running_acc / len(train_loader)

    return train_loss, train_accuracy


def test_epoch(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            _, input_len, _ = inputs.size()
            idx = torch.arange(1, input_len + 1)
            attn = torch.arange(1, (input_len // 4) + 1)
            idx = idx.unsqueeze(0).to(device)
            attn = attn.unsqueeze(0).to(device)
            inputs = inputs.to(torch.float32)
            idx = idx.to(torch.float32)
            outputs = model(inputs, idx, attn)
            outputs = torch.squeeze(outputs).to(torch.float32)
            labels = torch.squeeze(labels).to(torch.float32)

            _, predicted = torch.max(outputs, 1)
            loss = criterion(torch.softmax(outputs,1), labels)
            _, labels = torch.max(labels, 1)
            running_loss += loss.item()
            running_corrects = (predicted == labels).sum()
            running_acc += running_corrects.double() / labels.size(0)

    test_loss = running_loss / len(test_loader)
    test_accuracy = running_acc / len(test_loader)

    return test_loss, test_accuracy


def train(model, train_loader, test_loader, criterion1, optimizer, device, config):
    since = time.time()
    best_acc = 0.0

    for epoch in range(config.epoch):
        print('Epoch {}/{}'.format(epoch, config.epoch - 1))
        print('-' * 10)

        train_loss, train_accu = train_epoch(
            model, train_loader, criterion1, optimizer, device
        )
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(
            train_loss, train_accu))

        test_loss, test_accu = test_epoch(
            model, test_loader, criterion1, device
        )
        print('Test Loss: {:.4f} Acc: {:.4f}'.format(
            test_loss, test_accu))

        if test_accu > best_acc:
            best_acc = test_accu
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def main(config):
    # ========= Loading Dataset =========#
    train_loader = video_train_loader(config)
    test_loader = video_test_loader(config)

    # Using Cuda
    device = torch.device("cuda" if config.cuda else "cpu")
    if config.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')

    # ========= Preparing Model =========#
    sdConv = SDConv(
        n_position=config.n_position,
        n_classes=config.n_classes,
        n_dlayers=config.n_dlayers,
        num_f_maps=config.num_f_maps,
        d_model=config.d_model,
        d_inner=config.d_inner_hid,
        n_layers=config.n_layers,
        n_head=config.n_head,
        d_k=config.d_k,
        d_v=config.d_v,
        dropout=config.dropout
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, sdConv.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        config.d_model, config.n_warmup_steps)

    if config.train:
    	model = train(sdConv, train_loader, test_loader, criterion, optimizer, device, config)
    	# save the best model
    	torch.save(model, config.save_path + '.pth')
   
    if config.test:
    	net = torch.load(config.checkpoint, device)
    	eval_pred(net, test_loader, device, config.n_classes)


if __name__ == '__main__':
    main(config)
