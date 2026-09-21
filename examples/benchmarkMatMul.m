%% benchmarkMatMul.m
metalConfig  = Metal.Config;
device       = MetalDevice(metalConfig.gpudevice);
libraryCode  = string(fileread("MyKernels.mtl"));
library      = MetalLibrary(device, libraryCode);
matmul_cps   = MetalComputePipelineState(device, MetalFunction(library, "matmul"));

N = 1024;
A = rand([N,N], 'single');
B = rand([N,N], 'single');

bufA = MetalBuffer(device, A);
bufB = MetalBuffer(device, B);
bufC = MetalBuffer(device, zeros(N,N,'single'));
bufN = MetalBuffer(device, uint32(N));

tic; C_cpu = A * B; cpu_time = toc;

q = MetalCommandQueue(device); b = MetalCommandBuffer(q); e = MetalCommandEncoder(b);
e.SetComputePipelineState(matmul_cps);
e.SetBuffer(bufC, 1);
e.SetBuffer(bufA, 2);
e.SetBuffer(bufB, 3);
e.SetBuffer(bufN, 4);
e.SetThreadsAndShape(matmul_cps, N*N);
e.EndEncoding; b.Commit;
tic; b.WaitForCompletion; gpu_time = toc;

C_gpu = single(bufC);
fprintf('\nMatrix multiply [%dx%d]\n', N, N)
fprintf('CPU: %.4f sec\nGPU: %.4f sec\nSpeedup: %.1fx\n', cpu_time, gpu_time, cpu_time/gpu_time)
fprintf('Max error: %.4f\n', max(abs(C_gpu(:)-C_cpu(:))))

