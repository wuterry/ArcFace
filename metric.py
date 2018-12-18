import numpy as np
import mxnet as mx
from mxnet import nd, ndarray
from mxnet.metric import EvalMetric, check_label_shapes


class TopKAccuracy(EvalMetric):

    def __init__(self, top_k=1, name='top_k_accuracy',
                 output_names=None, label_names=None):
        super(TopKAccuracy, self).__init__(
            name, top_k=top_k,
            output_names=output_names, label_names=label_names)
        self.top_k = top_k
        assert(self.top_k > 1), 'Please use Accuracy if top_k is no more than 1'
        self.name += '_%d' % self.top_k

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        check_label_shapes(labels, preds)

        for label, pred_label in zip(labels, preds):
            assert(len(pred_label.shape) <= 2), 'Predictions should be no more than 2 dims'
            pred_label = np.argsort(pred_label.asnumpy().astype('float32'), axis=1)
            label = label.asnumpy().astype('int32')
            check_label_shapes(label, pred_label)
            num_samples = pred_label.shape[0]
            num_dims = len(pred_label.shape)
            if num_dims == 1:
                self.sum_metric += (pred_label.flat == label.flat).sum()
            elif num_dims == 2:
                num_classes = pred_label.shape[1]
                top_k = min(num_classes, self.top_k)
                for j in range(top_k):
                    self.sum_metric += (pred_label[:, num_classes - 1 - j].flat == label.flat).sum()
            self.num_inst += num_samples


class MAE(EvalMetric):
    def __init__(self, name='mae',
                 output_names=None, label_names=None):
        super(MAE, self).__init__(
            name, output_names=output_names, label_names=label_names)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            pred = nd.argmax(pred, axis=1)
            label = label.asnumpy()
            pred = pred.asnumpy()

            # if len(label.shape) == 1:
            #     label = label.reshape(label.shape[0], 1)

            self.sum_metric += np.abs(label - pred).mean()
            self.num_inst += 1


class Accuracy(EvalMetric):
    def __init__(self, axis=1, name='accuracy',
                 output_names=None, label_names=None):
        super(Accuracy, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names)
        self.axis = axis

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data with class indices as values, one per sample.

        preds : list of `NDArray`
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')
            # flatten before checking shapes to avoid shape miss match
            label = label.flat
            pred_label = pred_label.flat

            check_label_shapes(label, pred_label)

            self.sum_metric += (pred_label == label).sum()
            self.num_inst += len(pred_label)


class MultiAccuracy(EvalMetric):
    """Calculate accuracies of multi label"""

    def __init__(self, num=None):
        self.num = num
        super(MultiAccuracy, self).__init__('acc')

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0 if self.num is None else [0] * self.num
        self.sum_metric = 0.0 if self.num is None else [0.0] * self.num

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        labels = nd.transpose(labels[0])
        preds = preds[0]

        for i in range(len(labels)):
            # print(i, preds[i].shape)
            pred_label = mx.nd.argmax_channel(preds[i], axis=1).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            mx.metric.check_label_shapes(label, pred_label)

            if self.num is None:
                self.sum_metric += (pred_label.flat == label.flat).sum()
                self.num_inst += len(pred_label.flat)
            else:
                self.sum_metric[i] += (pred_label.flat == label.flat).sum()
                self.num_inst[i] += len(pred_label.flat)

    def get(self):
        """Gets the current evaluation result.
        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num is None:
            return super(MultiAccuracy, self).get()
        else:
            return zip(*(('task%d-%s' % (i, self.name), float('nan') if self.num_inst[i] == 0 \
                else self.sum_metric[i] / self.num_inst[i]) for i in range(self.num)))

    def get_name_value(self):
        """Returns zipped name and value pairs.
        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        if self.num is None:
            return super(MultiAccuracy, self).get_name_value()
        name, value = self.get()
        return list(zip(name, value))


class MultiCELoss(EvalMetric):
    """Calculate accuracies of multi label"""

    def __init__(self, num=None):
        self.eps = 1e-12
        self.num = num
        super(MultiCELoss, self).__init__('loss')

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0 if self.num is None else [0] * self.num
        self.sum_metric = 0.0 if self.num is None else [0.0] * self.num

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        labels = nd.transpose(labels[0])
        preds = preds[0]

        for i in range(len(labels)):
            pred = mx.ndarray.softmax(preds[i], axis=1).asnumpy()
            label = labels[i].asnumpy()

            mx.metric.check_label_shapes(label, pred)

            label = label.ravel()
            assert label.shape[0] == pred.shape[0]

            prob = pred[np.arange(label.shape[0]), np.int64(label)]

            if self.num is None:
                self.sum_metric += (-np.log(prob + self.eps)).sum()
                self.num_inst += label.shape[0]
            else:
                self.sum_metric[i] += (-np.log(prob + self.eps)).sum()
                self.num_inst[i] += label.shape[0]

    def get(self):
        """Gets the current evaluation result.
        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num is None:
            return super(MultiAccuracy, self).get()
        else:
            return zip(*(('task%d-%s' % (i, self.name), float('nan') if self.num_inst[i] == 0 \
                else self.sum_metric[i] / self.num_inst[i]) for i in range(self.num)))

    def get_name_value(self):
        """Returns zipped name and value pairs.
        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        if self.num is None:
            return super(MultiCELoss, self).get_name_value()
        name, value = self.get()
        return list(zip(name, value))


class CrossEntropy(EvalMetric):
    """Computes Cross Entropy loss.

    The cross entropy over a batch of sample size :math:`N` is given by

    .. math::
       -\\sum_{n=1}^{N}\\sum_{k=1}^{K}t_{nk}\\log (y_{nk}),

    where :math:`t_{nk}=1` if and only if sample :math:`n` belongs to class :math:`k`.
    :math:`y_{nk}` denotes the probability of sample :math:`n` belonging to
    class :math:`k`.

    Parameters
    ----------
    eps : float
        Cross Entropy loss is undefined for predicted value is 0 or 1,
        so predicted values are added with the small constant.
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> ce = mx.metric.CrossEntropy()
    >>> ce.update(labels, predicts)
    >>> print ce.get()
    ('cross-entropy', 0.57159948348999023)
    """
    def __init__(self, eps=1e-12, name='cross-entropy',
                 output_names=None, label_names=None):
        super(CrossEntropy, self).__init__(
            name, eps=eps,
            output_names=output_names, label_names=label_names)
        self.eps = eps

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred in zip(labels, preds):
            pred = nd.softmax(pred, axis=1)
            label = label.asnumpy()
            pred = pred.asnumpy()

            label = label.ravel()
            assert label.shape[0] == pred.shape[0]
            prob = pred[np.arange(label.shape[0]), np.int64(label)]
            self.sum_metric += (-np.log(prob + self.eps)).sum()
            self.num_inst += label.shape[0]
