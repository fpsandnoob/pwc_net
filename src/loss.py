import mindspore.nn as nn
from mindspore.nn.loss.loss import LossBase
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.ops import operations as P

class PyramidEPE(LossBase):
    def __init__(self):
        super(PyramidEPE, self).__init__()
        self.scale_weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    @staticmethod
    def downsample2d_as(input, target_shape_tensor):
        _, _, h1, _ = P.Shape()(target_shape_tensor)
        _, _, h2, _ = P.Shape()(input)
        resize = h2 // h1
        return nn.AvgPool2d(1, stride=(resize, resize))(input) * (1.0 / resize)

    @staticmethod
    def elementwise_epe(input1, input2):
        return nn.Norm(axis=1, keep_dims=True)(input1 - input2)

    def construct(self, prediction, target):
        if self.training:
            target = target * 0.05
            total_loss = 0
            for i, pred in enumerate(prediction):
                _target = self.downsample2d_as(target, pred)
                total_loss += self.elementwise_epe(_target, pred).sum() * self.scale_weights[i]
            return total_loss / P.Shape()(target)[0]
        else:
            loss = self.elementwise_epe(target, prediction)
            total_loss = loss.mean()
            return total_loss.sum()



class MultiStepLR(LearningRateSchedule):
    def __init__(self, lr, milestones, gamma):
        super().__init__()
        self.lr = lr
        self.milestones = milestones
        self.gamma = gamma
    
    def construct(self, global_step):
        lr = self.lr
        for milestone in self.milestones:
            if global_step >= milestone:
                lr *= self.gamma
        return lr