import torch
from torch.utils.data import DataLoader
import numpy as np
import torchmetrics

import cfg


def train(model,
          train_set,
          val_set,
          device,
          optimizer,
          criterion):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                           T_max=cfg.num_epochs,
                                                           eta_min=1e-6)
    train_loader = DataLoader(train_set,
                              batch_size=cfg.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_set,
                            batch_size=1,
                            shuffle=False)
    train_dice = torchmetrics.Dice(cfg.num_classes,
                                   ignore_index=0).to(device)
    eval_dice = torchmetrics.Dice(cfg.num_classes,
                                  ignore_index=0).to(device)
    epoch = cfg.start_epoch
    train_loss_file = open('output\\losses\\train loss.txt', 'w')
    eval_loss_file = open('output\\losses\\eval loss.txt', 'w')
    print("Training start...")
    while epoch < cfg.num_epochs:
        model.train()
        train_losses = []
        for batch_id, data in enumerate(train_loader):
            imgs, labels = data['image'].to(device), data['mask'].to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = torch.argmax(out, dim=1)
            train_dice.update(preds, labels)
            train_losses.append(loss.item())
        train_mean_loss = np.mean(train_losses).item()
        train_epoch_dice = train_dice.compute()
        train_dice.reset()

        model.eval()
        eval_losses = []
        for batch_id, data in enumerate(val_loader):
            imgs, labels = data['image'].to(device), data['mask'].to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            preds = torch.argmax(out, dim=1)
            eval_dice.update(preds, labels)
            eval_losses.append(loss.item())
        eval_mean_loss = np.mean(eval_losses).item()
        eval_epoch_dice = eval_dice.compute()
        eval_dice.reset()
        scheduler.step()
        print("Epoch[{}/{}]:".format(epoch + 1, cfg.num_epochs))
        print("train loss: {:.6f}, eval loss: {:.6f}".format(train_mean_loss, eval_mean_loss))
        print("train dice: {:.4f}, eval dice: {:.4f}\n".format(train_epoch_dice, eval_epoch_dice))
        torch.save(model.state_dict(), 'output\\weights\\epoch{}_loss{:.6f}.pth'.format(epoch + 1, eval_mean_loss))
        train_loss_file.write(str(train_mean_loss) + '\n')
        eval_loss_file.write(str(eval_mean_loss) + '\n')
        epoch = epoch + 1
    train_loss_file.close()
    eval_loss_file.close()
