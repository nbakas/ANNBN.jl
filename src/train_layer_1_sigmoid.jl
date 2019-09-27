function train_layer_1_sigmoid(neurons,vars,obs,n_per_part,inds_all,x,y)

    # a_all=zeros(neurons,vars+1)
    a_all=Vector{Vector{Float64}}()
    ooops=0
    for i=1:neurons # num_ranges^vars
        # if i in 1:1000:neurons println(Dates.format(now(), "HH:MM:SS")," Calculating weights w for ",i," neuron, of ",neurons," Total.") end
        # if n_per_part[i+1]-n_per_part[i]!=0

            try
                atmp=[ones(n_per_part[i+1]-n_per_part[i]) x[inds_all[n_per_part[i]+1:n_per_part[i+1]],:]]\isigm1.(y[inds_all[n_per_part[i]+1:n_per_part[i+1]]])
                if sum(isnan.(atmp))==0
                    # a_all[i,:].=atmp
                    err=[ones(n_per_part[i+1]-n_per_part[i]) x[inds_all[n_per_part[i]+1:n_per_part[i+1]],:]]*atmp.-isigm1.(y[inds_all[n_per_part[i]+1:n_per_part[i+1]]])
                    println(i," ",mean(abs.(err)))
                else
                    ooops+=1
                    # println("local inverse problem, neuron=",i," total ooops=",ooops)
                end
                # # println(a_all[i,:])

                obs1=length(inds_all[n_per_part[i]+1:n_per_part[i+1]])
                # println(i," ",obs1)
                if sum(isnan.(atmp))==0 && sum(isinf.(atmp))==0 push!(a_all,atmp) else push!(a_all,zeros(vars+1)); ooops+=1 end

            catch ex
                ooops+=1
                println("problem neuron=",i," total ooops=",ooops," ex=",ex)
                # push!(a_all,zeros(length(inds_all[n_per_part[i]+1:n_per_part[i+1]])))
                push!(a_all,zeros(vars+1))
                # continue
            end
        # end
        # println("calculating weights for neuron ",i," of ",neurons)
    end
    # println("ooops=",ooops)
    layer1=zeros(obs,neurons)
    for i=1:neurons
        if i in 1:100:neurons println(Dates.format(now(), "HH:MM:SS")," Applying sigmoid on ",i," neuron, of ",neurons," Total.") end
        # layer1[:,i].=sigm1.([ones(obs) x]*a_all[i,:])
        # println(size(a_all[i]))
        layer1[:,i].=sigm1.([ones(obs) x]*a_all[i])
        # x[i_so,:]*a_all[i,:]
    end
    # println("calculating linear terms...")
    # a_layer1=[sigm1.(layer1) ones(obs)]\y
    # println(Dates.format(now(), "HH:MM:SS")," Solving system (",obs," by ",neurons+1,") for weights v in the output Layer")
    a_layer1=[layer1 ones(obs)]\y
    # println(Dates.format(now(), "HH:MM:SS")," All weights computed.")
    # predl1=[sigm1.(layer1) ones(obs)]*a_layer1


    return a_all,a_layer1,layer1

end
