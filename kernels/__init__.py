import math
import torch

def select_kernel(args):
    if args.kernel == 'linear':
        return linear_kernel
    if args.kernel == 'laplace':
        return laplace
    if args.kernel == 'gaussian':
        return gaussian
    elif args.kernel == 'ntk':
        return gram_ntk
    else:
        raise ValueError("Please, specify a valid kernel.")

def k0(phi):
    return (math.pi - phi) / (math.pi)

def k1(phi):
    return (torch.sin(phi) + (math.pi - phi) * torch.cos(phi)) / math.pi

def gram_ntk(X, Y, depth=1):
    '''
    X, Y tensors of shape (n, d)
    '''
    assert Y.size(1) == X.size(1), 'data have different dimension!'

    XdY = X @ Y.t()
    XnYn = X.norm(dim=1, keepdim=True) @ Y.norm(dim=1, keepdim=True).t()
    ntk = torch.clamp( XdY / XnYn, min=-1., max=1.)
    rfk = torch.clone( ntk)

    for _ in range(depth):
        ntk.mul_( k0( rfk))
        rfk = k1( rfk)
        ntk.add_( rfk)
#         ntk /= 2

    return ntk

def linear_kernel(X, Y):
    '''
    X, Y tensors of shape (n, d)
    '''
    assert Y.size(1) == X.size(1), 'data have different dimension!'

    return X @ Y.t()

def laplace(X, Y=None, c=None):
    '''
    X, Y tensors of shape (n, d)
    '''
    if c is None:
        c = 1 / X.shape[1]
    if Y is None:
        Y = X
    return (-c * (X[:, None] - Y[None]).pow(2).sum(dim=-1).sqrt()).exp()

def gaussian(X, Y=None, c=None):
    '''
    X, Y tensors of shape (n, d)
    '''
    if c is None:
        c = 1 / X.shape[1]
    if Y is None:
        Y = X
    return (-c / 2 * (X[:, None] - Y[None]).pow(2).sum(dim=-1)).exp()
