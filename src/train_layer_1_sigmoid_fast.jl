function train_layer_1_sigmoid_fast(neurons,vars,obs,n_per_part,inds_all,x,y)

    a_all=Vector{Vector{Float64}}()
    ooops=0
    for i=1:neurons
        # if i in 1:10:neurons println(Dates.format(now(), "HH:MM:SS")," Calculating weights w for ",i," neuron, of ",neurons," Total.") end
        try
            atmp=[ones(n_per_part[i+1]-n_per_part[i]) x[inds_all[n_per_part[i]+1:n_per_part[i+1]],:]]\isigm1.(y[inds_all[n_per_part[i]+1:n_per_part[i+1]]])
            obs1=length(inds_all[n_per_part[i]+1:n_per_part[i+1]])
            if sum(isnan.(atmp))==0 && sum(isinf.(atmp))==0 && sum(abs.(atmp[2:end]))>0
                push!(a_all,atmp)
            else
                ooops+=1
            end
        catch ex
            ooops+=1
            println("problem neuron=",i," total ooops=",ooops," ex=",ex)
            push!(a_all,zeros(vars+1))
        end
    end
    layer1=sigm1.([ones(obs) x]*reduce(hcat, a_all))
    mat1=inv([layer1 ones(obs)]'*[layer1 ones(obs)])
    # println(Dates.format(now(), "HH:MM:SS")," Solving system (",obs," by ",size(layer1,2)+1,") for weights v in the output Layer")
    a_layer1=mat1*[layer1 ones(obs)]'*y
    # println(Dates.format(now(), "HH:MM:SS")," All weights computed.")
    return a_all,a_layer1,layer1,mat1

end
