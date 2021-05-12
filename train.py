import time
import os
import torch
from config import device


def train(epoch, vis, train_loader, model, criterion, optimizer, scheduler, opts):

    print('Training of epoch [{}]'.format(epoch))
    tic = time.time()
    model.train()

    for idx, data in enumerate(train_loader):

        # burn in process
        if opts.warm_up is not None:
            warm_up_idx = idx + epoch * len(train_loader)
            if warm_up_idx < opts.warm_up:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = opts.lr * ((warm_up_idx + 1) / opts.warm_up) ** 4

        optimizer.zero_grad()

        input = data['input'].to(device)                         # [B, 64]
        target = data['target'].to(device)                       # [B, 64]
        input_mask = data['input_mask'].to(device)               # [B, 1, 64]
        target_mask = data['target_mask'].to(device)             # [B, 64, 64]

        # print(input.size())
        # print(target.size())
        # print(input_mask.size())
        # print(target_mask.size())

        logits = model(input, target, input_mask, target_mask)  # [B, 64, 110000]

        shift_logits = logits[..., :-1, :].contiguous()         # [B, 63, 110000] - [eos]/[CLS] 제거
        shift_labels = target[..., 1:].contiguous()             # [B, 63, 110000] - [sos]/[SEP] 제거
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # sgd
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        toc = time.time()

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # for each steps
        if idx % opts.vis_step == 0 or idx == len(train_loader) - 1:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'Loss: {loss:.4f}\t'
                  'Learning rate: {lr:.7f} s \t'
                  'Time : {time:.4f}\t'
                  .format(epoch, idx, len(train_loader),
                          loss=loss,
                          lr=lr,
                          time=toc - tic))

            if vis is not None:
                # loss plot
                vis.line(X=torch.ones((1, 1)).cpu() * idx + epoch * train_loader.__len__(),  # step
                         Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
                         win='train_loss',
                         update='append',
                         opts=dict(xlabel='step',
                                   ylabel='Loss',
                                   title='training loss',
                                   legend=['Total Loss']))

    # # 각 epoch 마다 저장
    if not os.path.exists(opts.save_path):
        os.mkdir(opts.save_path)

    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict()}

    torch.save(checkpoint, os.path.join(opts.save_path, opts.save_file_name + '.{}.pth.tar'.format(epoch)))



