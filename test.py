


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