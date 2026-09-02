function bufferOut = metalSqrt(device, sqrt_cps, bufferA)
    bufferOut = MetalBuffer(device, zeros(size(single(bufferA)), 'single'));

    q = MetalCommandQueue(device);
    b = MetalCommandBuffer(q);
    e = MetalCommandEncoder(b);
    e.SetComputePipelineState(sqrt_cps);
    e.SetBuffer(bufferOut, 1);
    e.SetBuffer(bufferA,   2);
    e.SetThreadsAndShape(sqrt_cps, numel(single(bufferA)));
    e.EndEncoding;
    b.Commit;
    b.WaitForCompletion;
end
