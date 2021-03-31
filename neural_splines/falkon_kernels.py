import functools
from abc import ABC
from typing import Optional

import cupy as cp
import numpy as np
import torch
from falkon.kernels import Kernel, KeopsKernelMixin
from falkon.options import FalkonOptions
from falkon.sparse.sparse_tensor import SparseTensor
from torch.utils.dlpack import to_dlpack


def _extract_float(d):
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


class NeuralSplineKernel(Kernel, KeopsKernelMixin, ABC):
    kernel_type = "angle"

    def __init__(self, variance: float = 1.0, opt: Optional[FalkonOptions] = None):
        super().__init__("NeuralSpline", self.kernel_type, opt)
        self.debug = opt.debug if opt is not None else False
        self.variance = _extract_float(variance)

    def extra_mem(self):
        return {
            # We transpose X2 in _apply
            'nd': 0,
            'md': 1,
            # Norm results in prepare
            'm': 0,
            'n': 0,
            # We do a copy in _apply
            'nm': 1,
        }

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt):
        if self.debug:
            print(f"NeuralSpline._keops_mmv_impl(X1={X1.shape}, X2={X2.shape}, v, kernel, out, opt)")

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
            print(f"NeuralSpline._prepare(X1={X1.shape}, X2={X2.shape}, *kwargs)")
        return []

    def _prepare_sparse(self, X1: SparseTensor, X2: SparseTensor):
        raise NotImplementedError("NeuralSpline does not implement sparse prepare")

    def _apply(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor):
        if self.debug:
            print(f"NeuralSpline._apply(X1={X1.shape}, X2={X2.shape}, out={out.shape})")

        kernel_code = r'''
        #define PI (DTYPE) (3.1415926535897932384626433832795028841971693993751058209749445923078164062)
        #define ONE (DTYPE) (1.0)
        extern "C" __global__
        void stable_kernel(const DTYPE* x1, const DTYPE* x2, DTYPE* out, const double variance, 
                           const int N, int M, int D) {
            const int I = (blockIdx.x * blockDim.x) + threadIdx.x;
            const int J = (blockIdx.y * blockDim.y) + threadIdx.y;

            if (I >= N || J >= M) {
                return;
            }

            DTYPE norm_x = (DTYPE) 0.0; //normf(D, &x1[I*D]);
            DTYPE norm_y = (DTYPE) 0.0; //normf(D, &x2[J*D]);

            #pragma unroll 
            for (int k = 0; k < D; k += 1) {
                norm_x = fma(x1[I * D + k], x1[I * D + k], norm_x);
                norm_y = fma(x2[J * D + k], x2[J * D + k], norm_y);
            }
            norm_x = sqrt(norm_x);
            norm_y = sqrt(norm_y);

            DTYPE arg1 = (DTYPE) 0.0;
            DTYPE arg2 = (DTYPE) 0.0;

            #pragma unroll
            for (int k = 0; k < D; k += 1) {
                DTYPE x1_ik = x1[I * D + k];
                DTYPE x2_jk = x2[J * D + k];
                DTYPE a1 = norm_y * x1_ik - norm_x * x2_jk;
                DTYPE a2 = norm_y * x1_ik + norm_x * x2_jk;

                arg1 = fma(a1, a1, arg1);
                arg2 = fma(a2, a2, arg2);
            }
            arg1 = sqrt(arg1);
            arg2 = sqrt(arg2);

            DTYPE angle = 2.0 * atan2(arg1, arg2);

            DTYPE norm_xy = norm_x * norm_y;
            DTYPE cos_angle = cos(angle);
            DTYPE sin_angle = sin(angle);
            DTYPE opv = ONE + (DTYPE)(variance);
            DTYPE K = norm_xy * (sin_angle + opv * (PI - angle) * cos_angle) / PI;
            out[I * M + J] = K;
        }
        '''
        assert X1.dtype == X2.dtype == out.dtype, "X1, X2, and out don't have the same dtype"
        assert X1.device == X2.device == out.device, "X1, X2, and out are not on the same device"
        assert out.device.index is not None, "None device index"

        if X1.dtype == torch.float32:
            str_dtype = "float"
            cupy_dtype = cp.float32
        elif X1.dtype == torch.float64:
            str_dtype = "double"
            cupy_dtype = cp.float64
        else:
            raise ValueError("Invalid dtype must be float32 or float64")

        kernel_code = kernel_code.replace("DTYPE", str_dtype)
        kernel = cp.RawKernel(kernel_code, 'stable_kernel')

        # The .contiguous should be a no-op in both these cases, but add them in for good measure
        X1 = X1.contiguous()
        X2 = X2.T.contiguous()

        # Convert X1 and X2 to CuPy arrays.
        x1cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(X1))
        x2cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(X2))
        with cp.cuda.Device(out.device.index):
            outcp = cp.zeros((out.shape[0], out.shape[1]), dtype=cupy_dtype)

        # Run the CUDA kernel to build the matrix K
        pt_dim = int(X1.shape[1])
        dims = int(X1.shape[0]), int(X2.shape[0])
        threads_per_block = (16, 16)  # TODO: Maybe hardcoding this is bad
        blocks_per_grid = tuple((dims[i] + threads_per_block[i] - 1) // threads_per_block[i] for i in range(2))
        kernel(blocks_per_grid, threads_per_block, (x1cp, x2cp, outcp, self.variance, dims[0], dims[1], pt_dim))
        cp.cuda.stream.get_current_stream().synchronize()  # Need to synchronize so we can copy to PyTorch

        # print("COPYING CUPY OUT TO PYTORCH")
        # print("OUT CUPY\n", outcp)
        # Copy the kernel back into the output PyTorch tensor
        outcp_dlpack = outcp.toDlpack()
        out_dlpack = torch.utils.dlpack.from_dlpack(outcp_dlpack)
        out.copy_(out_dlpack)
        # print("OUT PYTORCH\n", out)

        # rand_idx_i, rand_idx_j = np.random.randint(X1.shape[0]), np.random.randint(X2.shape[0])
        # xi, xj = X1[rand_idx_i].detach().cpu().numpy(), X2[rand_idx_j].detach().cpu().numpy()
        # nxi, nxj = np.linalg.norm(xi), np.linalg.norm(xj)
        # angle1, angle2 = np.linalg.norm(nxj * xi - nxi * xj), np.linalg.norm(nxj * xi + nxi * xj)
        # angle = 2.0 * np.arctan2(angle1, angle2)
        # kij = nxi * nxj * (np.sin(angle) + (1.0 + self.variance) * (np.pi - angle) * np.cos(angle)) / np.pi
        # print(np.abs(kij - out[rand_idx_i, rand_idx_j].item()))

    def _apply_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor):
        raise NotImplementedError("NeuralSpline does not implement sparse apply")

    def _finalize(self, A: torch.Tensor, d):
        if self.debug:
            print(f"NeuralSpline._finalize(A={A.shape}, d)")
        return A

    def __str__(self):
        return f"NeuralSplineKernel()"

    def __repr__(self):
        return self.__str__()


class LaplaceKernelSphere(Kernel, KeopsKernelMixin, ABC):
    kernel_type = "angle"

    def __init__(self, alpha, gamma, opt: Optional[FalkonOptions] = None):
        super().__init__("LaplaceKernelSphere", self.kernel_type, opt)
        self.debug = opt.debug if opt is not None else False
        self.alpha = _extract_float(alpha)
        self.gamma = _extract_float(gamma)

    def extra_mem(self):
        return {
            # We transpose X2 in _apply
            'nd': 0,
            'md': 1,
            # Norm results in prepare
            'm': 0,
            'n': 0,
            # We do a copy in _apply
            'nm': 1,
        }

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt):
        if self.debug:
            print("LaplaceKernelSphere._keops_mmv_impl(X1, X2, v, kernel, out, opt)")

        theta = 'two * Atan2(Norm2(Norm2(Y) * X - Norm2(X) * Y), Norm2(Norm2(Y) * X + Norm2(X) * Y))'
        norm_xy = '(Norm2(X) * Norm2(Y))'
        j01 = f'({norm_xy} * (Exp(alpha * Powf(one - Cos({theta}), gamma))))'
        formula = f'({j01}) * v'
        aliases = [
            'X = Vi(%d)' % (X1.shape[1]),
            'Y = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'alpha = Pm(1)',
            'gamma = Pm(1)',
            'one = Pm(1)',
            'two = Pm(1)',
        ]
        other_vars = [torch.tensor([self.alpha]).to(dtype=X1.dtype, device=X1.device),
                      torch.tensor([self.gamma]).to(dtype=X1.dtype, device=X1.device),
                      torch.tensor([1.0]).to(dtype=X1.dtype, device=X1.device),
                      torch.tensor([2.0]).to(dtype=X1.dtype, device=X1.device)]

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
        return []

    def _prepare_sparse(self, X1: SparseTensor, X2: SparseTensor):
        raise NotImplementedError("LaplaceKernelSphere does not implement sparse prepare")

    def _apply(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor):
        if self.debug:
            print("LaplaceKernelSphere._apply(X1, X2, out)")
        kernel_code = r'''
         #define PI (DTYPE) (3.1415926535897932384626433832795028841971693993751058209749445923078164062)
         #define ONE (DTYPE) (1.0)
         extern "C" __global__
         void stable_kernel(const DTYPE* x1, const DTYPE* x2, DTYPE* out, const double alpha, double gamma, 
                            const int N, int M, int D) {
             const int I = (blockIdx.x * blockDim.x) + threadIdx.x;
             const int J = (blockIdx.y * blockDim.y) + threadIdx.y;

             if (I >= N || J >= M) {
                 return;
             }

             DTYPE norm_x = (DTYPE) 0.0; //normf(D, &x1[I*D]);
             DTYPE norm_y = (DTYPE) 0.0; //normf(D, &x2[J*D]);

             #pragma unroll 
             for (int k = 0; k < D; k += 1) {
                 norm_x = fma(x1[I * D + k], x1[I * D + k], norm_x);
                 norm_y = fma(x2[J * D + k], x2[J * D + k], norm_y);
             }
             norm_x = sqrt(norm_x);
             norm_y = sqrt(norm_y);

             DTYPE arg1 = (DTYPE) 0.0;
             DTYPE arg2 = (DTYPE) 0.0;

             #pragma unroll
             for (int k = 0; k < D; k += 1) {
                 DTYPE x1_ik = x1[I * D + k];
                 DTYPE x2_jk = x2[J * D + k];
                 DTYPE a1 = norm_y * x1_ik - norm_x * x2_jk;
                 DTYPE a2 = norm_y * x1_ik + norm_x * x2_jk;

                 arg1 = fma(a1, a1, arg1);
                 arg2 = fma(a2, a2, arg2);
             }
             arg1 = sqrt(arg1);
             arg2 = sqrt(arg2);

             DTYPE angle = 2.0 * atan2(arg1, arg2);

             DTYPE norm_xy = norm_x * norm_y;
             DTYPE cos_angle = cos(angle);
             DTYPE K = norm_xy * exp((DTYPE) alpha * pow(ONE - cos_angle, (DTYPE) gamma));
             out[I * M + J] = K;
         }
         '''
        assert X1.dtype == X2.dtype == out.dtype, "X1, X2, and out don't have the same dtype"
        assert X1.device == X2.device == out.device, "X1, X2, and out are not on the same device"
        assert out.device.index is not None, "None device index"

        if X1.dtype == torch.float32:
            str_dtype = "float"
            cupy_dtype = cp.float32
        elif X1.dtype == torch.float64:
            str_dtype = "double"
            cupy_dtype = cp.float64
        else:
            raise ValueError("Invalid dtype must be float32 or float64")

        kernel_code = kernel_code.replace("DTYPE", str_dtype)
        kernel = cp.RawKernel(kernel_code, 'stable_kernel')

        # The .contiguous should be a no-op in both these cases, but add them in for good measure
        X1 = X1.contiguous()
        X2 = X2.T.contiguous()

        # Convert X1 and X2 to CuPy arrays.
        x1cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(X1))
        x2cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(X2))
        with cp.cuda.Device(out.device.index):
            outcp = cp.zeros((out.shape[0], out.shape[1]), dtype=cupy_dtype)

        # Run the CUDA kernel to build the matrix K
        pt_dim = int(X1.shape[1])
        dims = int(X1.shape[0]), int(X2.shape[0])
        threads_per_block = (16, 16)  # TODO: Maybe hardcoding this is bad
        blocks_per_grid = tuple((dims[i] + threads_per_block[i] - 1) // threads_per_block[i] for i in range(2))
        kernel(blocks_per_grid, threads_per_block,
               (x1cp, x2cp, outcp, self.alpha, self.gamma, dims[0], dims[1], pt_dim))
        cp.cuda.stream.get_current_stream().synchronize()  # Need to synchronize so we can copy to PyTorch

        # Copy the kernel back into the output PyTorch tensor
        outcp_dlpack = outcp.toDlpack()
        out_dlpack = torch.utils.dlpack.from_dlpack(outcp_dlpack)
        out.copy_(out_dlpack)

    def _apply_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor):
        raise NotImplementedError("LaplaceKernelSphere does not implement sparse apply")

    def _finalize(self, A, d):
        if self.debug:
            print("LaplaceKernelSphere._finalize(A, d)")
        return A

    def __str__(self):
        return f"LaplaceKernelSphere(alpha={self.alpha})"

    def __repr__(self):
        return self.__str__()


class LinearAngleKernel(Kernel, KeopsKernelMixin, ABC):
    kernel_type = "angle"

    def __init__(self, multiply_norm=False, opt: Optional[FalkonOptions] = None):
        super().__init__("LinearAngleKernel", self.kernel_type, opt)
        self.debug = opt.debug if opt is not None else False
        self.multiply_norm = multiply_norm

    def extra_mem(self):
        return {
            # We transpose X2 in _apply
            'nd': 0,
            'md': 1,
            # Norm results in prepare
            'm': 0,
            'n': 0,
            # We do a copy in _apply
            'nm': 1,
        }

    def _keops_mmv_impl(self, X1, X2, v, kernel, out, opt):
        if self.debug:
            print(f"LinearAngleKernel._keops_mmv_impl(X1={X1.shape}, X2={X2.shape}, v, kernel, out, opt)")

        theta = 'two * Atan2(Norm2(Norm2(Y) * X - Norm2(X) * Y), Norm2(Norm2(Y) * X + Norm2(X) * Y))'
        if self.multiply_norm:
            norm_xy = '(Norm2(X) * Norm2(Y))'
            j01 = f'({norm_xy} * (pi - {theta}))'
        else:
            j01 = f'(pi - {theta})'
        formula = f'({j01} / pi) * v'
        aliases = [
            'X = Vi(%d)' % (X1.shape[1]),
            'Y = Vj(%d)' % (X2.shape[1]),
            'v = Vj(%d)' % (v.shape[1]),
            'pi = Pm(1)',
            'two = Pm(1)'
        ]
        other_vars = [torch.tensor([np.pi]).to(dtype=X1.dtype, device=X1.device),
                      torch.tensor([2.0]).to(dtype=X1.dtype, device=X1.device)]

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
            print(f"LinearAngleKernel._prepare(X1={X1.shape}, X2={X2.shape}, *kwargs)")
        return []

    def _prepare_sparse(self, X1: SparseTensor, X2: SparseTensor):
        raise NotImplementedError("LinearAngleKernel does not implement sparse prepare")

    def _apply(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor):
        if self.debug:
            print(f"LinearAngleKernel._apply(X1={X1.shape}, X2={X2.shape}, out={out.shape})")

        kernel_code = r'''
        #define PI (DTYPE) (3.1415926535897932384626433832795028841971693993751058209749445923078164062)
        #define ONE (DTYPE) (1.0)
        __MUL_NORM_DEFINE__
        
        extern "C" __global__
        void stable_kernel(const DTYPE* x1, const DTYPE* x2, DTYPE* out,
                           const int N, int M, int D) {
            const int I = (blockIdx.x * blockDim.x) + threadIdx.x;
            const int J = (blockIdx.y * blockDim.y) + threadIdx.y;

            if (I >= N || J >= M) {
                return;
            }

            DTYPE norm_x = (DTYPE) 0.0; //normf(D, &x1[I*D]);
            DTYPE norm_y = (DTYPE) 0.0; //normf(D, &x2[J*D]);

            #pragma unroll 
            for (int k = 0; k < D; k += 1) {
                norm_x = fma(x1[I * D + k], x1[I * D + k], norm_x);
                norm_y = fma(x2[J * D + k], x2[J * D + k], norm_y);
            }
            norm_x = sqrt(norm_x);
            norm_y = sqrt(norm_y);

            DTYPE arg1 = (DTYPE) 0.0;
            DTYPE arg2 = (DTYPE) 0.0;

            #pragma unroll
            for (int k = 0; k < D; k += 1) {
                DTYPE x1_ik = x1[I * D + k];
                DTYPE x2_jk = x2[J * D + k];
                DTYPE a1 = norm_y * x1_ik - norm_x * x2_jk;
                DTYPE a2 = norm_y * x1_ik + norm_x * x2_jk;

                arg1 = fma(a1, a1, arg1);
                arg2 = fma(a2, a2, arg2);
            }
            arg1 = sqrt(arg1);
            arg2 = sqrt(arg2);

            DTYPE angle = 2.0 * atan2(arg1, arg2);

            #ifdef MULTIPLY_NORM
            DTYPE norm_xy = norm_x * norm_y;
            DTYPE K = norm_xy * (PI - angle) / PI;
            #else
            DTYPE K = (PI - angle) / PI;
            #endif
            
            out[I * M + J] = K;
        }
        '''
        assert X1.dtype == X2.dtype == out.dtype, "X1, X2, and out don't have the same dtype"
        assert X1.device == X2.device == out.device, "X1, X2, and out are not on the same device"
        assert out.device.index is not None, "None device index"

        if X1.dtype == torch.float32:
            str_dtype = "float"
            cupy_dtype = cp.float32
        elif X1.dtype == torch.float64:
            str_dtype = "double"
            cupy_dtype = cp.float64
        else:
            raise ValueError("Invalid dtype must be float32 or float64")

        kernel_code = kernel_code.replace("DTYPE", str_dtype)
        if self.multiply_norm:
            kernel_code = kernel_code.replace("__MUL_NORM_DEFINE__", "#define MULTIPLY_NORM\n")
        else:
            kernel_code = kernel_code.replace("__MUL_NORM_DEFINE__", "\n")
        kernel = cp.RawKernel(kernel_code, 'stable_kernel')

        # The .contiguous should be a no-op in both these cases, but add them in for good measure
        X1 = X1.contiguous()
        X2 = X2.T.contiguous()

        # Convert X1 and X2 to CuPy arrays.
        x1cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(X1))
        x2cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(X2))
        with cp.cuda.Device(out.device.index):
            outcp = cp.zeros((out.shape[0], out.shape[1]), dtype=cupy_dtype)

        # Run the CUDA kernel to build the matrix K
        pt_dim = int(X1.shape[1])
        dims = int(X1.shape[0]), int(X2.shape[0])
        threads_per_block = (16, 16)  # TODO: Maybe hardcoding this is bad
        blocks_per_grid = tuple((dims[i] + threads_per_block[i] - 1) // threads_per_block[i] for i in range(2))
        kernel(blocks_per_grid, threads_per_block, (x1cp, x2cp, outcp, dims[0], dims[1], pt_dim))
        cp.cuda.stream.get_current_stream().synchronize()  # Need to synchronize so we can copy to PyTorch

        # print("COPYING CUPY OUT TO PYTORCH")
        # print("OUT CUPY\n", outcp)
        # Copy the kernel back into the output PyTorch tensor
        outcp_dlpack = outcp.toDlpack()
        out_dlpack = torch.utils.dlpack.from_dlpack(outcp_dlpack)
        out.copy_(out_dlpack)
        # print("OUT PYTORCH\n", out)

        # rand_idx_i, rand_idx_j = np.random.randint(X1.shape[0]), np.random.randint(X2.shape[0])
        # xi, xj = X1[rand_idx_i].detach().cpu().numpy(), X2[rand_idx_j].detach().cpu().numpy()
        # nxi, nxj = np.linalg.norm(xi), np.linalg.norm(xj)
        # angle1, angle2 = np.linalg.norm(nxj * xi - nxi * xj), np.linalg.norm(nxj * xi + nxi * xj)
        # angle = 2.0 * np.arctan2(angle1, angle2)
        # kij = nxi * nxj * (np.sin(angle) + (1.0 + self.variance) * (np.pi - angle) * np.cos(angle)) / np.pi
        # print(np.abs(kij - out[rand_idx_i, rand_idx_j].item()))

    def _apply_sparse(self, X1: SparseTensor, X2: SparseTensor, out: torch.Tensor):
        raise NotImplementedError("LinearAngleKernel does not implement sparse apply")

    def _finalize(self, A: torch.Tensor, d):
        if self.debug:
            print(f"LinearAngleKernel._finalize(A={A.shape}, d)")
        return A

    def __str__(self):
        return f"NeuralSplineKernel()"

    def __repr__(self):
        return self.__str__()
