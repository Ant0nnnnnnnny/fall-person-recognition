import time
import logging
import os
import numpy as np
import torch
from .evaluate import accuracy, get_final_preds
from .vis import save_debug_images


def train(args, train_loader, model, optimizer, epoch, loss_func, log_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()

    end = time.time()
    
    for i, (x, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(args.device)
        target_weight = target_weight.to(args.device)
        x = x.to(args.device)
        # compute output
        outputs = model(x, target, target_weight)

        if isinstance(outputs, list):
            loss = loss_func(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += loss_func(output, target, target_weight)
        else:
            output = outputs
            loss = loss_func(output, target, target_weight)

        loss = loss_func(output, target, target_weight)
        _, avg_acc, cnt, pred = accuracy(output.cpu().detach().numpy(),
                                         target.cpu().numpy())
        losses.update(loss)
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
                      speed=x.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logging.info(msg)

            writer = log_writer['writer']
            global_steps = log_writer['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            log_writer['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(args.debug_dir, 'train'), i)
            save_debug_images(args, x, meta, target, pred*4, output,
                              prefix)


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


def validate(args, val_loader, val_dataset, model, loss_func, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, args.num_keypoints, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (x, target, target_weight, meta) in enumerate(val_loader):
            # measure data loading time

            target = target.to(args.device)
            target_weight = target_weight.to(args.device)
            num_images = x.size(0)
            # compute output
            outputs = model(x, target, target_weight)

            if isinstance(outputs, list):
                loss = loss_func(outputs[0], target, target_weight)
                for output in outputs[1:]:
                    loss += loss_func(output, target, target_weight)
            else:
                output = outputs
                loss = loss_func(output, target, target_weight)

            loss = loss_func(output, target, target_weight)
            # measure accuracy and record loss
            losses.update(loss.item(), x.size(0))

            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                args, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % args.print_steps == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logging.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(args.output_dir, 'val'), i
                )
                save_debug_images(args, x, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            args, all_preds, args.output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = args.model_name
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

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
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator, loss

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
