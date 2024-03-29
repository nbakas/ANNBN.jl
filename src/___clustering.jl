function ___clustering(neurons,xx_clust,max_iter)

    ##### A: standard k-means
    iseeds=Vector{Int}(undef, neurons)
    # initseeds!(iseeds, KmppAlg(), xx_clust')
    # initseeds!(iseeds, KmCentralityAlg(), xx_clust')
    initseeds!(iseeds, RandSeedAlg(), xx_clust')
    R = kmeans(xx_clust', neurons; maxiter=max_iter, display=:iter,
                init=iseeds) #   none
    a = assignments(R); c = counts(R); minimum(c)
    i1=sortperm(a);inds_all=copy(i1);items_per_neuron=copy(c);n_per_part=[0;cumsum(items_per_neuron)]
    ##### B: Enhanced Algorithm for equally sized clusters and invertible matrices (slower)
    # include("cluster_tests1.jl")
    return inds_all,n_per_part,items_per_neuron
end
