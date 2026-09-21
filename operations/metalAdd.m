function bufferOut = metalAdd(device, add_cps, bufferA, bufferB)
    bufferOut = MetalBuffer(device, zeros(size(single(bufferA)), 'single'));

    q = MetalCommandQueue(device);
    b = MetalCommandBuffer(q);
    e = MetalCommandEncoder(b);
    e.SetComputePipelineState(add_cps);
    e.SetBuffer(bufferOut, 1);
    e.SetBuffer(bufferA,   2);
    e.SetBuffer(bufferB,   3);
    e.SetThreadsAndShape(add_cps, numel(single(bufferA)));
    e.EndEncoding;
    b.Commit;
    b.WaitForCompletion;
end
