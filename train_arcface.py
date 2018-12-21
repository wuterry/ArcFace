from __future__ import division

import argparse, time, os
import logging
import numpy as np

import mxnet as mx
from mxnet import gluon
from mxnet import profiler

from mxnet import autograd as ag
from mxnet.test_utils import get_mnist_iterator
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
from metric import CrossEntropy
from model import ArcFace
from data import *


# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset to use. options are mnist, cifar10, caltech101, imagenet and dummy.')
parser.add_argument('--data-dir', type=str, default='',
                    help='training directory of imagenet images, contains train/val subdirs.')
parser.add_argument('--num-worker', '-j', dest='num_workers', default=4, type=int,
                    help='number of workers for dataloader')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--gpus', type=str, default='',
                    help='ordinates of gpus to use, can be "0,1,2" or empty for cpu only.')
parser.add_argument('--epochs', type=int, default=120,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.1.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. Default=123.')
parser.add_argument('--mode', type=str,
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')
parser.add_argument('--model', type=str, default='vgg16',
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--use_thumbnail', action='store_true',
                    help='use thumbnail or not in resnet. default is false.')
parser.add_argument('--batch-norm', action='store_true',
                    help='enable batch normalization or not in vgg. default is false.')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--prefix', default='models/arcface', type=str,
                    help='path to checkpoint prefix, default is current working dir')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='starting epoch, 0 for fresh training, > 0 to resume')
parser.add_argument('--resume', type=str, default='',
                    help='path to saved weight where you want resume')
parser.add_argument('--lr-factor', default=0.1, type=float,
                    help='learning rate decay ratio')
parser.add_argument('--lr-steps', default='50,80,100', type=str,
                    help='list of learning rate decay epochs as in str')
parser.add_argument('--dtype', default='float32', type=str,
                    help='data type, float32 or float16 if applicable')
parser.add_argument('--save-frequency', default=10, type=int,
                    help='epoch frequence to save model, best model will always be saved')
parser.add_argument('--save-type', default='mxnet', type=str,
                    help='type of saving, gluon of mxnet.')
parser.add_argument('--kvstore', type=str, default='device',
                    help='kvstore to use for trainer/module.')
parser.add_argument('--log-interval', type=int, default=50,
                    help='Number of batches to wait before logging.')
parser.add_argument('--profile', action='store_true',
                    help='Option to turn on memory profiling for front-end, ' \
                         'and prints out the memory usage by python function at the end.')
parser.add_argument('--builtin-profiler', type=int, default=0, help='Enable built-in profiler (0=off, 1=on)')
opt = parser.parse_args()

# global variables
mx.random.seed(opt.seed)
model_name = opt.model
dataset_classes = {'mnist': 10, 'cifar10': 10, 'caltech101': 101, 'imagenet': 1000, 'dummy': 1000}
batch_size, dataset, classes = opt.batch_size, opt.dataset, dataset_classes[opt.dataset]
context = [mx.gpu(int(i)) for i in opt.gpus.split(',')] if opt.gpus.strip() else [mx.cpu()]
num_gpus = len(context)
batch_size *= max(1, num_gpus)
lr_steps = [int(x) for x in opt.lr_steps.split(',') if x.strip()]
metric = CompositeEvalMetric([Accuracy(), TopKAccuracy(5), CrossEntropy()])
kv = mx.kv.create(opt.kvstore)


def get_data_iters(dataset, batch_size, opt):
    """get dataset iterators"""
    if dataset == 'mnist':
        train_data, val_data = get_mnist_iterator(batch_size, (1, 28, 28),
                                                  num_parts=kv.num_workers, part_index=kv.rank)
    elif dataset == 'cifar10':
        train_data, val_data = get_cifar10_iterator(batch_size, (3, 32, 32),
                                                    num_parts=kv.num_workers, part_index=kv.rank)
    elif dataset == 'imagenet':
        shape_dim = 299 if model_name == 'inceptionv3' else 224

        if not opt.data_dir:
            raise ValueError('Dir containing raw images in train/val is required for imagenet.'
                             'Please specify "--data-dir"')

        train_data, val_data = get_imagenet_iterator(opt.data_dir, batch_size,
                                                     opt.num_workers, shape_dim, opt.dtype)
    elif dataset == 'caltech101':
        train_data, val_data = get_caltech101_iterator(batch_size, opt.num_workers, opt.dtype)
    elif dataset == 'dummy':
        shape_dim = 299 if model_name == 'inceptionv3' else 224
        train_data, val_data = dummy_iterator(batch_size, (3, shape_dim, shape_dim))
    return train_data, val_data


def test(net, ctx, val_data):
    metric.reset()
    val_data.reset()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch.data[0].astype(opt.dtype, copy=False),
                                          ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0].astype(opt.dtype, copy=False),
                                           ctx_list=ctx, batch_axis=0)

        outputs = [net(X, y)[-1] for X, y in zip(data, label)]
        metric.update(label, outputs)
    return metric.get()


def update_learning_rate(lr, trainer, epoch, ratio, steps):
    """Set the learning rate to the initial value decayed by ratio every N epochs."""
    new_lr = lr * (ratio ** int(np.sum(np.array(steps) < epoch)))
    trainer.set_learning_rate(new_lr)
    return trainer


def save_checkpoint(net, epoch, top1, best_acc, save_type='mxnet'):
    save_prefix = f'{opt.prefix}/{opt.model}'

    if opt.save_frequency and (epoch + 1) % opt.save_frequency == 0:
        if save_type == 'gluon':
            fname = os.path.join(save_prefix, '_%d_acc_%.4f.params' % (epoch, top1))
            net.save_parameters(fname)
        else:
            net.export(save_prefix, epoch)
            logger.info('[Epoch %d] Saving checkpoint with Accuracy: %.4f', epoch, top1)

    if top1 > best_acc[0]:
        best_acc[0] = top1
        fname = os.path.join(opt.prefix, '%s_best.params' % (opt.model))
        if save_type == 'gluon':
            net.save_parameters(fname)
        else:
            net.export(f'{save_prefix}-best', 0)
        logger.info('[Epoch %d] Saving checkpoint to %s with Accuracy: %.4f', epoch, fname, top1)


def train(net, opt, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    train_data, val_data = get_data_iters(dataset, batch_size, opt)
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            optimizer_params={'learning_rate': opt.lr,
                                              'wd': opt.wd,
                                              'momentum': opt.momentum,
                                              'multi_precision': True},
                            kvstore=kv)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    total_time = 0
    num_epochs = 0
    best_acc = [0]
    for epoch in range(opt.start_epoch, opt.epochs):
        trainer = update_learning_rate(opt.lr, trainer, epoch, opt.lr_factor, lr_steps)
        tic = time.time()
        train_data.reset()
        metric.reset()
        btic = time.time()
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch.data[0].astype(opt.dtype), ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0].astype(opt.dtype), ctx_list=ctx, batch_axis=0)
            outputs = []
            Ls = []
            outputs_eval = []
            with ag.record():
                for x, y in zip(data, label):
                    z, ze = net(x, y)
                    L = loss(z, y)
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(z)
                    outputs_eval.append(ze)
                ag.backward(Ls)
            trainer.step(batch.data[0].shape[0])
            metric.update(label, outputs_eval)
            if opt.log_interval and not (i + 1) % opt.log_interval:
                name, indices = metric.get()
                logger.info('Epoch[{:3d}] Batch [{}]\t Speed: {:.0f} samples/sec\t {}={:.6f}\t{}={:.6f}\t{}={:.6f}'
                            .format(epoch, i, batch_size / (time.time() - btic),
                                    name[0], indices[0], name[1], indices[1], name[2], indices[2]))
            btic = time.time()

        epoch_time = time.time() - tic

        # First epoch will usually be much slower than the subsequent epics,
        # so don't factor into the average
        if num_epochs > 0:
            total_time = total_time + epoch_time
        num_epochs = num_epochs + 1

        name, indices = metric.get()
        logger.info('[Epoch %d] training: %s=%.6f, %s=%.6f, %s=%.6f'
                    % (epoch, name[0], indices[0], name[1], indices[1], name[2], indices[2]))
        logger.info('[Epoch %d] time cost: %f' % (epoch, epoch_time))
        name, val_indices = test(net, ctx, val_data)
        logger.info('[Epoch %d] validation: %s=%.6f, %s=%.6f, %s=%.6f'
                    % (epoch, name[0], val_indices[0], name[1], val_indices[1], name[2], val_indices[2]))

        # save model if meet requirements
        save_checkpoint(net, epoch, val_indices[0], best_acc)
    if num_epochs > 1:
        print('Average epoch time: {}'.format(float(total_time) / (num_epochs - 1)))


def main():
    net = ArcFace(backbone_output='vgg0_dense1_relu_fwd_output', num_class=classes, ctx=context, opt=opt)
    # net.collect_params().initialize()

    if opt.builtin_profiler > 0:
        profiler.set_config(profile_all=True, aggregate_stats=True)
        profiler.set_state('run')
    if opt.mode == 'symbolic':
        data = mx.sym.var('data')
        if opt.dtype == 'float16':
            data = mx.sym.Cast(data=data, dtype=np.float16)
        out = net(data)
        if opt.dtype == 'float16':
            out = mx.sym.Cast(data=out, dtype=np.float32)
        softmax = mx.sym.SoftmaxOutput(out, name='softmax')
        mod = mx.mod.Module(softmax, context=context)
        train_data, val_data = get_data_iters(dataset, batch_size, opt)
        mod.fit(train_data,
                eval_data=val_data,
                num_epoch=opt.epochs,
                kvstore=kv,
                batch_end_callback=mx.callback.Speedometer(batch_size, max(1, opt.log_interval)),
                epoch_end_callback=mx.callback.do_checkpoint('image-classifier-%s' % opt.model),
                optimizer='sgd',
                optimizer_params={'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum, 'multi_precision': True},
                initializer=mx.init.Xavier(magnitude=2))
        mod.save_parameters('%s/image-classifier-%s-%d-final.params' % (save_dir, opt.model, opt.epochs))
    else:
        if opt.mode == 'hybrid':
            print('Hybrid Network.')
            net.hybridize()
        train(net, opt, context)
    if opt.builtin_profiler > 0:
        profiler.set_state('stop')
        print(profiler.dumps())


if __name__ == '__main__':
    save_dir = 'models/arcface'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file = '{}/train_{}.log'.format(save_dir, time.strftime('%Y%m%d%H'))
    if os.path.exists(log_file):
        os.remove(log_file)

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    logger.addHandler(fh)

    if opt.profile:
        import hotshot, hotshot.stats

        prof = hotshot.Profile('image-classifier-%s-%s.prof' % (opt.model, opt.mode))
        prof.runcall(main)
        prof.close()
        stats = hotshot.stats.load('image-classifier-%s-%s.prof' % (opt.model, opt.mode))
        stats.strip_dirs()
        stats.sort_stats('cumtime', 'calls')
        stats.print_stats()
    else:
        main()
