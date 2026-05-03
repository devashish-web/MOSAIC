import numpy as np
import ot
import torch as th
from ot.bregman import sinkhorn
from ot.utils import dist, UndefinedParameter, list_to_array
from ot.optim import cg
from ot.lp import emd_1d, emd
from ot.utils import check_random_state
from ot.backend import get_backend
from ot.gromov import init_matrix, gwloss, gwggrad

    


def parallel_gromov_wasserstein2(C1, C2, p, q, loss_fun='square_loss', log=False, armijo=False, G0=None, **kwargs):
    p, q = list_to_array(p, q)

    p0, q0, C10, C20 = p, q, C1, C2
    nx = get_backend(p0, q0, C10, C20)

    p = nx.to_numpy(p)
    q = nx.to_numpy(q)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)
    
    if G0 is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = nx.to_numpy(G0)
        np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-04)
        np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-04)

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    T, log_gw = cg(p, q, 0, 1, f, df, G0, log=True, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)


    gp = nx.from_numpy(log_gw['u'] - log_gw['u'].mean())
    gq = nx.from_numpy(log_gw['v'] - log_gw['v'].mean())
    

    if loss_fun == 'square_loss':
        
        gC1 = nx.from_numpy(2 * C1 * (p[:, None] * p[None, :]) - 2 * T.dot(C2).dot(T.T))
        gC2 = nx.from_numpy(2 * C2 * (q[:, None] * q[None, :]) - 2 * T.T.dot(C1).dot(T))
    return nx.from_numpy(gwloss(constC, hC1, hC2, T), type_as=C10), gp, gq, gC1, gC2


def parallel_fused_gromov_wasserstein2_learnablealpha( C1, C2, F1, F2, M, p, q, loss_fun='square_loss', alpha=0.5, compute_gradients=True, learn_alpha=False, armijo=False, log=False, G0=None, **kwargs):
    p, q = list_to_array(p, q)

    p0, q0, C10, C20, F10, F20, M0, alpha0 = p, q, C1, C2, F1, F2, M, alpha
    nx = get_backend(p0, q0, C10, C20, F10, F20, M0, alpha0)

    p = nx.to_numpy(p0)
    q = nx.to_numpy(q0)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)
    F1 = nx.to_numpy(F10)
    F2 = nx.to_numpy(F20)
    M = nx.to_numpy(M0)
    alpha = nx.to_numpy(alpha0)
    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    if G0 is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = nx.to_numpy(G0)
        # Check marginals of G0
        np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-04)
        np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-04)

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    T, log_fgw = cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True, **kwargs)

    fgw_dist = nx.from_numpy(log_fgw['loss'][-1], type_as=C10)
    if not compute_gradients:
        return fgw_dist
    else:
    
        if loss_fun == 'square_loss':
            gC1 = nx.from_numpy(2 * C1 * (p[:, None] * p[None, :]) - 2 * T.dot(C2).dot(T.T))
            gC2 = nx.from_numpy(2 * C2 * (q[:, None] * q[None, :]) - 2 * T.T.dot(C1).dot(T))
            if learn_alpha:
                gwloss_ = gwloss(constC, hC1, hC2, T)
                galpha = nx.from_numpy(gwloss_ - (M*T).sum(), type_as=C10)
            else:
                galpha = None
        gp = nx.from_numpy(log_fgw['u'] - log_fgw['u'].mean())
        gq = nx.from_numpy(log_fgw['v'] - log_fgw['v'].mean())
        gF1 = nx.from_numpy(2 * F1 * p[:, None] - 2 * T.dot(F2))
        gF2 = nx.from_numpy(2 * F2 * q[:, None] - 2 * (T.T).dot(F1))
        
        gp = gp.to(fgw_dist.device)
        gq = gq.to(fgw_dist.device)
        gC1 = gC1.to(fgw_dist.device)
        gC2 = gC2.to(fgw_dist.device)
        gF1 = gF1.to(fgw_dist.device)
        gF2 = gF2.to(fgw_dist.device)
        
        return fgw_dist, gp, gq, alpha0 * gC1, alpha0 * gC2, (1. - alpha0) * gF1, (1. - alpha0) * gF2, galpha
    

from torch.autograd import Function
class ValFunction(Function):

    @staticmethod
    def forward(ctx, val, grads, *inputs):
        ctx.grads = grads
        return val

    @staticmethod
    def backward(ctx, grad_output):
        # the gradients are grad
        return (None, None) + tuple(g * grad_output for g in ctx.grads)

    

def set_gradients(Func, val, inputs, grads):

    res = Func.apply(val, grads, *inputs)

    return res



def probability_simplex_projection(x):
    descending_idx = th.argsort(x, descending=True)
    u = x[descending_idx]
    rho= 0.
    lambda_= 1.
    for i in range(u.shape[0]):
        value = u[i] + (1- u[:(i+1)].sum())/(i+1)
        if value>0:
            rho+=1
            lambda_-=u[i]
        else:
            break
    return th.max(x + lambda_/rho, th.zeros_like(x))

