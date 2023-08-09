import time
import logging

import torch

def train(args, train_loader, model, optimizer, epoch, loss_func, log_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()

    end = time.time()
    
    for i, (X,y, feats) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        y = y.to(args.device)
        X = X.to(args.device)
        # compute output
        output = model(X)

        loss = loss_func(output, y)
       
        losses.update(loss)

        avg_acc = calculate_accuracy(torch.argmax(output.cpu().detach(),dim = 1),y,feats)

        acc.update(avg_acc)
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_steps == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=X.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logging.info(msg)

            writer = log_writer['writer']
            global_steps = log_writer['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            log_writer['train_global_steps'] = global_steps + 1

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def validate(args, val_loader, model, loss_func, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (X,y,feats) in enumerate(val_loader):
            # measure data loading time

            target = target.to(args.device)
            target_weight = target_weight.to(args.device)
          
            # compute output
            output = model(X)

            loss = loss_func(output, y)

            # measure accuracy and record loss
            losses.update(loss.item(), X.size(0))

            classes_,all = calculate_accuracy(torch.argmax(output.cpu().detach(),dim = 1),y,feats,True)

            acc.update(all)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if i % args.print_steps == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logging.info(msg)

    
        model_name = args.model_name
           
        _print_name_value(classes_, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )

            writer.add_scalars(
                    'valid',
                    classes_,
                    global_steps
            )
            writer_dict['valid_global_steps'] = global_steps + 1

    return loss

# markdown format output


def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logging.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logging.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logging.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )

def calculate_accuracy(y_pred, y_real, feats, each_classes = False):

    with torch.no_grad():
        all = float(torch.sum((y_real == y_pred)/len(y_pred)))
        if each_classes:
            classes_ = {int(key):0 for key in y_real}
            for i in range(len(y_pred)):
                classes_[y_real.tolist()[i]] += int(y_real[i] == y_pred[i])
      
            classes_ = {feats['real_label'][key] if key != 0 else 'others':float(classes_[int(key)]/torch.sum(y_real==key)) for key in y_real}
            return classes_, all
        return all