%% benchmarkPipeline.m
% Compares CPU vs GPU for a chained pipeline run 1000 times.
% GPU advantage: data never leaves the GPU across iterations.

%% Setup
metalConfig  = Metal.Config;
device       = MetalDevice(metalConfig.gpudevice);
libraryCode  = string(fileread("MyKernels.mtl"));
library      = MetalLibrary(device, libraryCode);

multiply_cps = MetalComputePipelineState(device, MetalFunction(library, "multiply"));
add_cps      = MetalComputePipelineState(device, MetalFunction(library, "add"));
sqrt_cps     = MetalComputePipelineState(device, MetalFunction(library, "sqrtArr"));

%% Data
N     = 5000;
ITERS = 100;   % reduced from 1000 — each iter is 3 dispatches, still meaningful

A = rand([N, N], 'single');
B = rand([N, N], 'single');
E = rand([N, N], 'single');

nElem = N * N;

%% --- CPU loop ---
tic
F_cpu = A;
for i = 1:ITERS
    F_cpu = sqrt((F_cpu .* B) + E);
end
cpu_time = toc;

%% --- GPU: single command buffer, all iterations encoded upfront ---
% Allocate persistent buffers — data stays on GPU the whole time
bufF   = MetalBuffer(device, A);
bufB   = MetalBuffer(device, B);
bufE   = MetalBuffer(device, E);
bufTmp = MetalBuffer(device, zeros(N, N, 'single')); % scratch buffer

tic
q = MetalCommandQueue(device);
b = MetalCommandBuffer(q);

for i = 1:ITERS
    % Step 1: tmp = F .* B
    e = MetalCommandEncoder(b);
    e.SetComputePipelineState(multiply_cps);
    e.SetBuffer(bufTmp, 1);   % output
    e.SetBuffer(bufF,   2);   % input A
    e.SetBuffer(bufB,   3);   % input B
    e.SetThreadsAndShape(multiply_cps, nElem);
    e.EndEncoding;

    % Step 2: F = tmp + E
    e = MetalCommandEncoder(b);
    e.SetComputePipelineState(add_cps);
    e.SetBuffer(bufF,   1);   % output
    e.SetBuffer(bufTmp, 2);   % input A
    e.SetBuffer(bufE,   3);   % input B
    e.SetThreadsAndShape(add_cps, nElem);
    e.EndEncoding;

    % Step 3: F = sqrt(F)  — needs a tmp to avoid in-place read/write
    e = MetalCommandEncoder(b);
    e.SetComputePipelineState(sqrt_cps);
    e.SetBuffer(bufTmp, 1);   % output
    e.SetBuffer(bufF,   2);   % input
    e.SetThreadsAndShape(sqrt_cps, nElem);
    e.EndEncoding;

    % Copy tmp back into bufF for next iteration
    e = MetalCommandEncoder(b);
    e.SetComputePipelineState(add_cps);   % reuse add: F = tmp + 0
    bufZero = MetalBuffer(device, zeros(N, N, 'single'));
    e.SetBuffer(bufF,    1);
    e.SetBuffer(bufTmp,  2);
    e.SetBuffer(bufZero, 3);
    e.SetThreadsAndShape(add_cps, nElem);
    e.EndEncoding;
end

b.Commit;
b.WaitForCompletion;
F_gpu = single(bufF);
gpu_time = toc;

%% Results
fprintf('\nPipeline: F = sqrt((F .* B) + E)  x%d iterations  [%dx%d]\n\n', ITERS, N, N)
fprintf('%-20s %.4f sec\n', 'CPU:', cpu_time)
fprintf('%-20s %.4f sec\n', 'GPU:', gpu_time)
fprintf('\nSpeedup: %.1fx\n', cpu_time / gpu_time)

%% Verify
assert(~any(abs(F_gpu - F_cpu) > 1e-2, 'all'));
disp('Results match CPU output.')

