# Arcface Gluon

This is an implementation of Additive Angular Margin Loss with Gluon (MXNet).
Reference: ["ArcFace: Additive Angular Margin Loss for Deep Face Recognition"](https://arxiv.org/abs/1801.07698)

## How to use
```
python train_arcface.py --batch-size=256 --log-interval=20 --gpus=0 --lr=0.001 --mode=hybrid
```

## Experiments
### settings
- Test networks: VGG16
- Test dataset: cifar10

### results

|   loss   | softmax-loss | arcface-loss |
|:--------:|:------------:|:------------:|
| accuracy |    0.8247    |    0.8118    |

From the results of the compared losses, the accuracy of arcface-loss is lower than softmax-loss.
