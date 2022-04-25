import numpy as np

"""
xgb loss: https://zhuanlan.zhihu.com/p/278060736
"""
def xgbCateLoss(pred, dtrain):

    penalty, label = 1.0, dtrain.get_label()
    sigmoid_pred   = 1.0 / (1.0 + np.exp(-pred))

    grad = -(penalty**label) * (label-sigmoid_pred)
    hess = (penalty**label) * sigmoid_pred * (1.0-sigmoid_pred)

    return grad, hess

def xgbSucceLoss(pred, dtrain):
    penalty, label = 1.0, dtrain.get_label()
    residual = (label - pred).astype("float")

    grad = np.where(residual<0, -2*(residual)/(label+1), -10*2*(residual)/(label+1))
    hess = np.where(residual<0, 2 / (label + 1), 10 * 2 / (label + 1))

    return grad, hess