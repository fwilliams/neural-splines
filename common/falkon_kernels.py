import functools
from abc import ABC
from abc import abstractmethod
from typing import Optional

import numpy as np
import torch

from falkon.kernels import Kernel, KeopsKernelMixin
from falkon.options import FalkonOptions
from falkon.sparse.sparse_tensor import SparseTensor


def extract_float(d):
    if isinstance(d, torch.Tensor):
        try:
            # tensor.item() works if tensor is a scalar, otherwise it throws
            # a value error.
            return d.item()
        except ValueError:
            raise ValueError("Item is not a scalar")
    else:
        try:
            return float(d)
        except TypeError:
            raise TypeError("Item must be a scalar or a tensor.")


def k_spherical_laplace(x, xp, gamma, alpha):
    x = x.unsqueeze(1)  # [s1, 1, d]
    xp = xp.unsqueeze(0)  # [1, s2, d]
    x_dot_xp = (x * xp).sum(-1)  # [s1, s2]
    norm_x = torch.norm(x, dim=-1)  # [s1, 1]
    norm_xp = torch.norm(xp, dim=-1)  # [1, s2]
    norm_x_xp = norm_x * norm_xp  # [s2, s1]

    cos_alpha_x = torch.clamp(x_dot_xp / norm_x_xp, -1.0, 1.0)  # [s1, s2]

    return norm_x_xp * torch.exp(alpha * torch.pow(1.0 - cos_alpha_x, gamma)).squeeze(-1)


def k_arccos(x, xp):
    # x and xp have shape [s1, d] and [s2, d] respectively

    x = x.unsqueeze(1)  # [s1, 1, d]
    xp = xp.unsqueeze(0)  # [1, s2, d]
    x_dot_xp = (x * xp).sum(-1)  # [s1, s2]
    norm_x = torch.norm(x, dim=-1)  # [s1, 1]
    norm_xp = torch.norm(xp, dim=-1)  # [1, s2]
    norm_x_xp = norm_x * norm_xp  # [s2, s1]

    cos_alpha_x = torch.clamp(x_dot_xp / norm_x_xp, -1.0, 1.0)  # [s1, s2]
    alpha_x = torch.acos(cos_alpha_x)  # [s1, s2]
    sin_alpha_x = torch.sin(alpha_x)  # [s2, s1]

    K = norm_x_xp * (sin_alpha_x + (np.pi - alpha_x) * cos_alpha_x)  # [s1, s2]
    return K / np.pi


def k_ntk(x, xp, variance=1.0):
    # x and xp have shape [s1, d] and [s2, d] respectively

    x = x.unsqueeze(1)  # [s1, 1, d]
    xp = xp.unsqueeze(0)  # [1, s2, d]
    x_dot_xp = (x * xp).sum(-1)  # [s1, s2]
    norm_x = torch.norm(x, dim=-1)  # [s1, 1]
    norm_xp = torch.norm(xp, dim=-1)  # [1, s2]
    norm_x_xp = norm_x * norm_xp  # [s2, s1]

    cos_alpha_x = torch.clamp(x_dot_xp / norm_x_xp, -1.0, 1.0)  # [s1, s2]
    alpha_x = torch.acos(cos_alpha_x)  # [s1, s2]
    sin_alpha_x = torch.sin(alpha_x)  # [s2, s1]

    K = norm_x_xp * (sin_alpha_x + (1.0 + variance) * (np.pi - alpha_x) * cos_alpha_x)  # [s1, s2]
    return K / np.pi


class DirectKernelSolver:
    def __init__(self, kernel, penalty):
        self.penalty = penalty
        self.alpha_ = None
        self.x_ = None
        self.kernel = kernel

    def fit(self, x, y):
        Kxx = self.kernel.direct_kernel(x, x)
        self.x_ = x
        self.alpha_ = torch.inverse(Kxx + self.penalty * torch.eye(Kxx.shape[0], Kxx.shape[1])) @ y

    def predict(self, x):
        Kxxp = self.kernel.direct_kernel(x, self.x_)
        return Kxxp @ self.alpha_


class DirectKernelMixin:
    @abstractmethod
    def direct_kernel(self, X1, X2):
        pass


class ArcCosineKernel(Kernel, KeopsKernelMixin, ABC, DirectKernelMixin):
    kernel_type = "dot-product"

    def __init__(self, opt: Optional[FalkonOptions] = None):
        super().__init__("ArcCosine", self.kernel_type, opt)
        self.debug = opt.debug if opt is not None else False
        # self.alpha = torch.tensor(extract_float(alpha), dtype=torch.float64)

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt):
        if self.debug:
            print(f"ArcCosineKernel._keops_mmv_impl(X1={X1.shape}, X2={X2.shape}, v, kernel, out, opt)")

        norm_xy = '(Norm2(X) * Norm2(Y))'
        cos_theta = '((Normalize(X) | Normalize(Y)))'
        theta = 'Acos(' + cos_theta + ')'

        j1 = '(Sin({theta}) + (pi - {theta}) * {cos_theta})'.format(theta=theta, cos_theta=cos_theta)
        formula = '(({norm_xy} / pi) * {j1}) * v'.format(norm_xy=norm_xy, j1=j1)
        aliases = [
            'X = Vi(%d)' % (X1.shape[1]),
            'Y = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'pi = Pm(1)',
        ]
        other_vars = [torch.tensor([np.pi]).to(dtype=X1.dtype, device=X1.device)]

        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def _decide_mmv_impl(self, X1, X2, v, opt):
        if self.keops_can_handle_mmv(X1, X2, v, opt):
            return self._keops_mmv_impl
        else:
            return super()._decide_mmv_impl(X1, X2, v, opt)

    def _decide_dmmv_impl(self, X1, X2, v, w, opt):
        if self.keops_can_handle_dmmv(X1, X2, v, w, opt):
            return functools.partial(self.keops_dmmv_helper, mmv_fn=self._keops_mmv_impl)
        else:
            return super()._decide_dmmv_impl(X1, X2, v, w, opt)

    def _prepare(self, X1, X2, **kwargs):
        if self.debug:
            print(f"ArcCosineKernel._prepare(X1={X1.shape}, X2={X2.shape}, *kwargs)")
        return [torch.norm(X1, dim=-1).unsqueeze(1), torch.norm(X2, dim=-1).unsqueeze(0)]

    def _prepare_sparse(self, X1: SparseTensor, X2: SparseTensor):
        raise NotImplementedError("ArcCosineKernel does not implement sparse prepare")

    def _apply(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor):
        out.addmm_(X1, X2)

    def _apply_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor):
        raise NotImplementedError("ArcCosineKernel does not implement sparse apply")

    def _finalize(self, A: torch.Tensor, d):
        if self.debug:
            print(f"ArcCosineKernel._finalize(A={A.shape}, d)")
        n1, n2 = [_.to(A) for _ in d]
        A.div_(n1)
        A.div_(n2)
        A.clamp_(-1.0, 1.0)  # cos(theta)

        B = torch.acos(A)  # theta
        C = torch.sin(B)  # sin(theta)
        B.mul_(-1.0)  # -theta
        B.add_(np.pi)  # pi - theta
        A.mul_(B)  # (pi - theta) * cos(theta)
        A.add_(C)  # sin(theta) + (pi - theta) * cos(theta) = J_1
        del B, C

        A.mul_(n1)  # |X| * J_1
        A.mul_(n2)  # |X| * |Y| * J_1
        A.div_(np.pi)  # |X| * |Y| * J_1 / pi
        return A

    def direct_kernel(self, X1, X2):
        return k_arccos(X1, X2)

    def __str__(self):
        return f"ArcCosineKernel()"

    def __repr__(self):
        return self.__str__()


class NeuralTangentKernel(Kernel, KeopsKernelMixin, ABC, DirectKernelMixin):
    kernel_type = "dot-product"

    def __init__(self, variance: float = 1.0, opt: Optional[FalkonOptions] = None):
        super().__init__("NeuralTangentKernel", self.kernel_type, opt)
        self.debug = opt.debug if opt is not None else False
        self.variance = variance

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt):
        if self.debug:
            print(f"NeuralTangentKernel._keops_mmv_impl(X1={X1.shape}, X2={X2.shape}, v, kernel, out, opt)")

        norm_xy = '(Norm2(X) * Norm2(Y))'
        cos_theta = '((Normalize(X) | Normalize(Y)))'
        theta = 'Acos(' + cos_theta + ')'

        j01 = f'({norm_xy} * (Sin({theta}) + (one + variance) * (pi - {theta}) * {cos_theta}))'
        formula = f'({j01} / pi) * v'
        aliases = [
            'X = Vi(%d)' % (X1.shape[1]),
            'Y = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'pi = Pm(1)',
            'variance = Pm(1)',
            'one = Pm(1)'
        ]
        other_vars = [torch.tensor([np.pi]).to(dtype=X1.dtype, device=X1.device),
                      torch.tensor([self.variance]).to(dtype=X1.dtype, device=X1.device),
                      torch.tensor([1.0]).to(dtype=X1.dtype, device=X1.device)]

        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def _decide_mmv_impl(self, X1, X2, v, opt):
        if self.keops_can_handle_mmv(X1, X2, v, opt):
            return self._keops_mmv_impl
        else:
            return super()._decide_mmv_impl(X1, X2, v, opt)

    def _decide_dmmv_impl(self, X1, X2, v, w, opt):
        if self.keops_can_handle_dmmv(X1, X2, v, w, opt):
            return functools.partial(self.keops_dmmv_helper, mmv_fn=self._keops_mmv_impl)
        else:
            return super()._decide_dmmv_impl(X1, X2, v, w, opt)

    def _prepare(self, X1, X2, **kwargs):
        if self.debug:
            print(f"NeuralTangentKernel._prepare(X1={X1.shape}, X2={X2.shape}, *kwargs)")
        return [torch.norm(X1, dim=-1).unsqueeze(1), torch.norm(X2, dim=-1).unsqueeze(0)]

    def _prepare_sparse(self, X1: SparseTensor, X2: SparseTensor):
        raise NotImplementedError("NeuralTangentKernel does not implement sparse prepare")

    def _apply(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor):
        out.addmm_(X1, X2)

    def _apply_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor):
        raise NotImplementedError("NeuralTangentKernel does not implement sparse apply")

    def _finalize(self, A: torch.Tensor, d):
        if self.debug:
            print(f"NeuralTangentKernel._finalize(A={A.shape}, d)")
        n1, n2 = [_.to(A) for _ in d]
        A.div_(n1)
        A.div_(n2)
        A.clamp_(-1.0, 1.0)  # cos(theta)

        B = torch.acos(A)  # theta
        C = torch.sin(B)  # sin(theta)
        B.mul_(-1.0)  # -theta
        B.add_(np.pi)  # pi - theta
        A.mul_(B)  # (pi - theta) * cos(theta)
        A.mul_(1.0 + self.variance)  # (1 + sigma) * (pi - theta) * cos(theta)
        A.add_(C)  # sin(theta) + (1 + sigma) * (pi - theta) * cos(theta) = J_1 + J_0
        del B, C

        A.mul_(n1)  # |X| * J_1
        A.mul_(n2)  # |X| * |Y| * J_1
        A.div_(np.pi)  # |X| * |Y| * J_1 / pi
        return A

    def direct_kernel(self, X1, X2):
        return k_arccos(X1, X2)

    def __str__(self):
        return f"ArcCosineKernel()"

    def __repr__(self):
        return self.__str__()


class LaplaceKernelSphere(Kernel, KeopsKernelMixin, DirectKernelMixin, ABC):
    kernel_type = "dot-product"

    def __init__(self, alpha, gamma, opt: Optional[FalkonOptions] = None):
        super().__init__("LaplaceKernelSphere", self.kernel_type, opt)
        self.debug = opt.debug if opt is not None else False
        self.alpha = torch.tensor(extract_float(alpha), dtype=torch.float64)
        self.gamma = torch.tensor(extract_float(gamma), dtype=torch.float64)

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt):
        if self.debug:
            print("LaplaceKernelSphere._keops_mmv_impl(X1, X2, v, kernel, out, opt)")
        # formula = 'Norm2(X) * Norm2(Y) * Exp(alpha * Sqrt(one - Clamp11(Normalize(X) | Normalize(Y)))) * v'
        formula = 'Norm2(X) * Norm2(Y) * Exp(alpha * Powf(one - Clamp11((Normalize(X) | Normalize(Y))), gamma)) * v'
        aliases = [
            'X = Vi(%d)' % (X1.shape[1]),
            'Y = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'alpha = Pm(1)',
            'gamma = Pm(1)',
            'one = Pm(1)',
        ]
        other_vars = [torch.tensor([self.alpha]).to(dtype=X1.dtype, device=X1.device),
                      torch.tensor([self.gamma]).to(dtype=X1.dtype, device=X1.device),
                      torch.tensor([1.0]).to(dtype=X1.dtype, device=X1.device)]

        return self.keops_mmv(X1, X2, v, out, formula, aliases, other_vars, opt)

    def _decide_mmv_impl(self, X1, X2, v, opt):
        if self.keops_can_handle_mmv(X1, X2, v, opt):
            return self._keops_mmv_impl
        else:
            return super()._decide_mmv_impl(X1, X2, v, opt)

    def _decide_dmmv_impl(self, X1, X2, v, w, opt):
        if self.keops_can_handle_dmmv(X1, X2, v, w, opt):
            return functools.partial(self.keops_dmmv_helper, mmv_fn=self._keops_mmv_impl)
        else:
            return super()._decide_dmmv_impl(X1, X2, v, w, opt)

    def _prepare(self, X1, X2, **kwargs):
        if self.debug:
            print("LaplaceKernelSphere._prepare(X1, X2, *kwargs)")
        return [torch.norm(X1, dim=-1).unsqueeze(1), torch.norm(X2, dim=-1).unsqueeze(0)]

    def _prepare_sparse(self, X1: SparseTensor, X2: SparseTensor):
        raise NotImplementedError("LaplaceKernelSphere does not implement sparse prepare")

    def _apply(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor):
        if self.debug:
            print("LaplaceKernelSphere._apply(X1, X2, out)")
        out.addmm_(X1, X2)

    def _apply_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor):
        raise NotImplementedError("LaplaceKernelSphere does not implement sparse apply")

    def _finalize(self, A, d):
        if self.debug:
            print("LaplaceKernelSphere._finalize(A, d)")
        alpha = self.alpha.to(A)
        gamma = self.gamma.to(A)
        n1, n2 = [_.to(A) for _ in d]
        A.div_(n1)
        A.div_(n2)
        A.clamp_(-1.0, 1.0)
        A.mul_(-1.0)
        A.add_(1.0)
        A.pow_(gamma)
        A.mul_(alpha)
        A.exp_()
        A.mul_(n1)
        A.mul_(n2)
        return A

    def direct_kernel(self, X1, X2):
        return k_spherical_laplace(X1, X2, self.gamma, self.alpha)

    def __str__(self):
        return f"LaplaceKernelSphere(alpha={self.alpha})"

    def __repr__(self):
        return self.__str__()