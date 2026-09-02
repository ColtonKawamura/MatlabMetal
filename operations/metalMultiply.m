function bufferOut = metalMultiply(device, multiply_cps, bufferA, bufferB)
    bufferOut = MetalBuffer(device, zeros(size(single(bufferA)), 'single'));

    q = MetalCommandQueue(device);
    b = MetalCommandBuffer(q);
    e = MetalCommandEncoder(b);
    e.SetComputePipelineState(multiply_cps);
    e.SetBuffer(bufferOut, 1);
    e.SetBuffer(bufferA,   2);
    e.SetBuffer(bufferB,   3);
    e.SetThreadsAndShape(multiply_cps, numel(single(bufferA)));
    e.EndEncoding;
    b.Commit;
    b.WaitForCompletion;
end

