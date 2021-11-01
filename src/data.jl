function unitNormalise(X)
    minX = minimum(X[:,1,:])
    maxX = maximum(X[:,1,:])

    minY = minimum(X[:,2,:])
    maxY = maximum(X[:,2,:])
    
    minZ = minimum(X[:,3,:])
    maxZ = maximum(X[:,3,:])

    normX = (X[:,1,:] .- minX) ./ (maxX - minX);
    normY = (X[:,2,:] .- minY) ./ (maxY - minY);
    normZ = (X[:,3,:] .- minZ) ./ (maxZ - minZ);

    return cat(normX, normY, normZ, dims=3)
end