import cupy as cp
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import torch

print("CuPy Stream", cp.cuda.get_current_stream())
print("Pytorch Stream", torch.cuda.current_stream())
X1 = torch.rand(25_000, 4)
X2 = torch.rand(25_000, 4)
out = torch.rand(25_000, 25_000)


kernel_code = r'''
#define PI (DTYPE) (3.1415926535897932384626433832795028841971693993751058209749445923078164062)
#define ONE (DTYPE) (1.0)
extern "C" __global__
void stable_kernel(const DTYPE* x1, const DTYPE* x2, DTYPE* out, const double variance, const int N, int M, int D) {
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
str_dtype = "float" if X1.dtype == torch.float32 else "double"
kernel_code = kernel_code.replace("DTYPE", str_dtype)
kernel = cp.RawKernel(kernel_code, 'stable_kernel')

x1cp = cp.fromDlpack(to_dlpack(X1))
x2cp = cp.fromDlpack(to_dlpack(X2))
outcp = cp.fromDlpack(to_dlpack(out))
print(x1cp.flags, x2cp.flags, outcp.flags)

pt_dim = int(X1.shape[1])
dims = int(X1.shape[0]), int(X2.shape[0])
threads = (64, 64)  # TODO: Maybe hardcoding this is bad
blocks = tuple((dims[i] + threads[i] - 1) // threads[i] for i in range(2))

print(x1cp.shape, x2cp.shape, outcp.shape)
print(dims[0], dims[1], pt_dim)

kernel(threads, blocks, (x1cp, x2cp, outcp, 1.0, dims[0], dims[1], pt_dim))
# out = from_dlpack(outcp.toDlpack())
print(out)