import torch

def keops_formula(X1, X2, v):
    theta = 'two * Atan2(Norm2(Norm2(Y) * X - Norm2(X) * Y), Norm2(Norm2(Y) * X + Norm2(X) * Y))'
    norm_xy = '(Norm2(X) * Norm2(Y))'

    j01 = f'({norm_xy} * (Sin({theta}) + (one + variance) * (pi - {theta}) * Cos({theta})))'
    formula = f'({j01} / pi) * v'
    aliases = [
        'X = Vi(%d)' % (X1.shape[1]),
        'Y = Vj(%d)' % (X2.shape[1]),
        'v = Vj(%d)' % (v.shape[1]),
        'pi = Pm(1)',
        'variance = Pm(1)',
        'one = Pm(1)',
        'two = Pm(1)'
    ]
    other_vars = [torch.tensor([np.pi]).to(dtype=X1.dtype, device=X1.device),
                  torch.tensor([self.variance]).to(dtype=X1.dtype, device=X1.device),
                  torch.tensor([1.0]).to(dtype=X1.dtype, device=X1.device),
                  torch.tensor([2.0]).to(dtype=X1.dtype, device=X1.device)]

