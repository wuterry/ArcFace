import math
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models


def load_model(ctx, classes, opt):
    """Model initialization."""
    kwargs = {'ctx': ctx, 'pretrained': opt.use_pretrained, 'classes': classes}
    if opt.model.startswith('resnet'):
        kwargs['thumbnail'] = opt.use_thumbnail
    elif opt.model.startswith('vgg'):
        kwargs['batch_norm'] = opt.batch_norm

    net = models.get_model(opt.model, **kwargs)
    if opt.resume:
        net.load_parameters(opt.resume)
    elif not opt.use_pretrained:
        if opt.model in ['alexnet']:
            net.initialize(mx.init.Normal())
        else:
            net.initialize(mx.init.Xavier(magnitude=2))
    net.cast(opt.dtype)
    return net


def get_backbone(output_layer_name, classes, ctx, opt):
    net = load_model(ctx, classes, opt)
    data = mx.sym.var('data')
    internals = net(data).get_internals()
    output_list = [internals[output_layer_name]]
    net = mx.gluon.SymbolBlock(output_list, data, params=net.collect_params())
    return net


class L2Normalization(nn.HybridBlock):
    def __init__(self):
        super(L2Normalization, self).__init__()

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.L2Normalization(x)


class ArcFace(nn.HybridBlock):
    def __init__(self, backbone_output, num_class, ctx, opt, margin_s=64, margin_m=0, easy_margin=False):
        super(ArcFace, self).__init__()
        self.mode = opt.mode
        self.num_class = num_class
        self.m = margin_m
        self.s = margin_s
        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.embedding = get_backbone(backbone_output, num_class, ctx, opt)

        # with self.name_scope():
        self._units = num_class
        self._in_units = 4096
        self.fc_weight = self.params.get('fc_weight', shape=(self._units, self._in_units),
                                         init=mx.initializer.Xavier(), dtype='float32')
        self.fc_weight.initialize(mx.initializer.Xavier())

    def hybrid_forward(self, F, x, label, fc_weight):
        embedding = F.L2Normalization(self.embedding(x))
        weight    = F.L2Normalization(fc_weight, mode='instance')
        fc = F.FullyConnected(embedding, weight, no_bias=True, num_hidden=self._units, flatten=True, name='fwd')
        cos_theta = fc
        sin_theta = F.sqrt(1 - cos_theta ** 2)
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m

        if self.easy_margin:
            phi = F.where(cos_theta > 0, phi, cos_theta)
        else:
            phi = F.where(cos_theta > self.th, phi, cos_theta - self.mm)
        # convert label to one-hot
        one_hot = F.one_hot(label, depth=self.num_class)
        # where(out_i = {x_i if condition_i else y_i)
        output = one_hot * phi + (1.0 - one_hot) * cos_theta
        output = output * self.s
        return output, fc

