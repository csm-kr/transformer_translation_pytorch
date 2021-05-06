import time
import torch
import os
import math
import model
import tqdm


def train(self, epochs, train_dataset, eval_dataset, optimizer, scheduler):
    self.model.train()
    total_loss = 0.
    global_steps = 0
    start_time = time.time()
    losses = {}
    best_val_loss = float("inf")
    best_model = None
    start_epoch = 0
    start_step = 0
    train_dataset_length = len(train_dataset)

    self.model.to(self.device)
    if os.path.isfile(f'{self.checkpoint_path}/{self.model_name}.pth'):
        checkpoint = torch.load(f'{self.checkpoint_path}/{self.model_name}.pth', map_location=self.device)
        start_epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        global_steps = checkpoint['train_step']
        start_step = global_steps if start_epoch == 0 else (global_steps % train_dataset_length) + 1

        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()

        pb = tqdm(enumerate(train_dataset),
                  desc=f'Epoch-{epoch} Iterator',
                  total=train_dataset_length,
                  bar_format='{l_bar}{bar:10}{r_bar}'
                  )
        pb.update(start_step)
        for i, data in pb:
            if i < start_step:
                continue
            """
            doc={
            "input":input,                                              # input
            "input_mask": (input != pad_token_idx).unsqueeze(-2),       # input_mask
            "target": target,                                           # target,
            "target_mask": self.make_std_mask(target, pad_token_idx),   # target_mask
            "token_num": (target[...,1:] != pad_token_idx).data.sum()   # token_num
          }
            """
            input = data['input'].to(self.device)
            target = data['target'].to(self.device)
            input_mask = data['input_mask'].to(self.device)
            target_mask = data['target_mask'].to(self.device)

            optimizer.zero_grad()
            generator_logit, loss = self.model.forward(input, target, input_mask, target_mask, labels=target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            losses[global_steps] = loss.item()
            total_loss += loss.item()
            log_interval = 1
            save_interval = 500

            global_steps += 1

            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                # print('| epoch {:3d} | {:5d}/{:5d} batches | '
                #       'lr {:02.2f} | ms/batch {:5.2f} | '
                #       'loss {:5.2f} | ppl {:8.2f}'.format(
                #         epoch, i, len(train_dataset), scheduler.get_lr()[0],
                #         elapsed * 1000 / log_interval,
                #         cur_loss, math.exp(cur_loss)))
                pb.set_postfix_str('| epoch {:3d} | {:5d}/{:5d} batches | '
                                   'lr {:02.2f} | ms/batch {:5.2f} | '
                                   'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, i, len(train_dataset), scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
                # self.save(epoch, self.model, optimizer, losses, global_steps)
                if i % save_interval == 0:
                    self.save(epoch, self.model, optimizer, losses, global_steps)
        val_loss = self.evaluate(eval_dataset)
        self.model.train()
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
        start_step = 0
        scheduler.step()