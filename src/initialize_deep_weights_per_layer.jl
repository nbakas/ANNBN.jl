function initialize_deep_weights_per_leyer(neurons,vars,a_all,nodes,a_layer1,n_weigths,x,y,obs,layer1,mat1)

    opti_ww=zeros(n_weigths)

    it1=0
    for i=1:neurons
        # for j=2:vars+1
        for j=2:length(a_all[i])
            it1+=1
            # opti_ww[it1]=a_all[i,j]
            opti_ww[it1]=a_all[i][j]
        end
    end
    println(Dates.format(now(), "HH:MM:SS")," ",it1," weights layer 1")

    layer_prev=copy(layer1)
    a_bias=Array{Float64}(undef,0)
    for i=2:length(nodes)
        layer_new=zeros(obs,neurons)

        # i1=layer_prev.>=0.9
        # layer_prev[i1].=0.9
        # i1=layer_prev.<=0.1
        # layer_prev[i1].=0.1
        # a_all_next=[ones(obs) layer_prev]\isigm1.(y[:,1])
        if i==2
            a_all_next=mat1*[layer_prev ones(obs)]'*isigm1.(y[:,1])
            # a_all_next=[layer_prev ones(obs)]\isigm1.(y[:,1])
        end
        if i==3
            # mat1=inv([layer_prev ones(obs)]'*[layer_prev ones(obs)])
            # a_all_next=mat1*[layer_prev ones(obs)]'*isigm1.(y[:,1])

            a_all_next=[layer_prev ones(obs)]\isigm1.(y[:,1])
        end
        layer_new=repeat(sigm1.([layer_prev ones(obs)]*a_all_next),1,neurons)

        for j=1:neurons # or nodes[i]
            push!(a_bias,a_all_next[end])
            for k=1:neurons
                it1+=1
                opti_ww[it1]=a_all_next[k]
            end
            # println("weight layer $(i) of $(length(nodes)), neuron $(j) of $(neurons)")
        end
        layer_prev=copy(layer_new)
        println(Dates.format(now(), "HH:MM:SS")," ",it1," weights layer ", i)
    end


    a_all_next=[layer_prev ones(obs)]\y
    # a_all_next=mat1*[layer_prev ones(obs)]'*y
    for i=1:neurons
        it1+=1
        opti_ww[it1]=a_all_next[i]
    end
    # for i=1:neurons
    #     it1+=1
    #     opti_ww[it1]=a_layer1[i]
    # end
    println(Dates.format(now(), "HH:MM:SS"),it1," weights layer out")

    # bias
    for i=1:neurons
        it1+=1
        # opti_ww[it1]=a_all[i,1]
        opti_ww[it1]=a_all[i][1]
    end
    println(it1," bias layer 1")

    it2=0
    for i=2:length(nodes)
        for j=1:nodes[i]
            it1+=1
            it2+=1
            opti_ww[it1]=a_bias[it2]
        end
        println(it1," bias layer ",i)
    end


    it1+=1
    opti_ww[it1]=a_all_next[end]
    # it1+=1
    # opti_ww[it1]=a_layer1[end]
    println(Dates.format(now(), "HH:MM:SS")," ",it1," bias layer out")

    return opti_ww,layer_prev


end
