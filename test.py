import os
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon.model_zoo import vision as models
from data import get_cifar10_iterator

# ---------------------------------
ctx           = mx.gpu(0)
batch_size    = 128
height, width = 32, 32
# ---------------------------------


def get_new_arguments(net, arg_params, aux_params):
    arg_names = net.list_arguments()
    new_args, new_auxs = dict(), dict()
    for k, v in arg_params.items():
        if k in arg_names:
            new_args[k] = v
    for k, v in aux_params.items():
            new_auxs[k] = v
    return new_args, new_auxs


def convert_model(prefix, epoch, save_prefix):
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    net = all_layers['arcface0_fwd_output']
    new_args, new_auxs = get_new_arguments(net, arg_params, aux_params)
    save_callback = mx.callback.do_checkpoint(save_prefix)
    save_callback(-1, net, new_args, new_auxs)
    print("Convert to deploy model done.")


def load_model(prefix, epoch, ctx):
    deploy_prefix = f'{prefix}-deploy'
    convert_model(prefix, epoch, save_prefix=deploy_prefix)
    sym, arg_params, aux_params = mx.model.load_checkpoint(f'{prefix}-deploy', 0)
    mod = mx.mod.Module(symbol=sym, context=ctx, data_names=('data0',), label_names=None)
    mod.bind(for_training=False,
             data_shapes=[('data0', (batch_size, 3, height, width))])
    mod.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True)
    return sym, mod



def main():
    _, data = get_cifar10_iterator(batch_size, (3, height, width))
    sym, mod = load_model(f'models/arcface/vgg16-best', 0, ctx=ctx)

    labels = []
    pred_labels  = []
    for preds, ibatch, batch in mod.iter_predict(data):
        nsample = len(preds[0])

        # get ground truth labels
        labels.extend(list(batch.label[0].asnumpy())[0:nsample])

        # calculate predict
        prob = mx.ndarray.softmax(preds[0], axis=1)
        pred_labels.extend(list(mx.ndarray.argmax(prob, axis=1).asnumpy()))

        print('batch:', ibatch)

    pred_labels = np.array(pred_labels)
    labels = np.array(labels)
    acc = sum(1.0 * (pred_labels == labels)) / len(labels)
    print('accuracy:', acc)

main()