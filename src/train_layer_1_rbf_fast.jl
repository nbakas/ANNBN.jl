function train_layer_1_rbf_fast(neurons,vars,obs,n_per_part,inds_all,x,y,cc)

    # a_all=zeros(neurons,1000vars)
    a_all=Vector{Vector{Float64}}()
    ooops=0
    for i=1:neurons
        # if i in 1:1:neurons println(Dates.format(now(), "HH:MM:SS")," Calculating weights w for ",i," neuron, of ",neurons," Total.") end
        # if n_per_part[i+1]-n_per_part[i]!=0
            try
                x1=copy(x[inds_all[n_per_part[i]+1:n_per_part[i+1]],:])
                obs1=size(x1,1)
                phi1 = calc_phi(x1,x1,obs1,obs1,cc,vars)
                atmp=phi1\y[inds_all[n_per_part[i]+1:n_per_part[i+1]]]
                # display(plot(y[inds_all[n_per_part[i]+1:n_per_part[i+1]]]))
                # println(mean(y[inds_all[n_per_part[i]+1:n_per_part[i+1]]])," ",n_per_part[i+1]-n_per_part[i])
                # if sum(isnan.(atmp))==0 && sum(isinf.(atmp))==0 a_all[i,1:obs1].=atmp else ooops+=1 end
                # if sum(isnan.(atmp))==0 && sum(isinf.(atmp))==0 push!(a_all,atmp) else push!(a_all,zeros(obs1)); ooops+=1 end
                if sum(isnan.(atmp))==0 && sum(isinf.(atmp))==0 && sum(abs.(atmp[2:end]))>0 push!(a_all,atmp)
                    push!(i_keep,i)
                else
                    ooops+=1
                end
            catch ex
                ooops+=1; println(i," ",ex)
            end
        # end
    end
    println("ooops=",ooops)
    layer1=zeros(obs,neurons)
    for i=1:neurons
        if i in 1:100:neurons println(Dates.format(now(), "HH:MM:SS")," Calculating output of neuron ",i," , of ",neurons," Total.") end
        x1=copy(x[inds_all[n_per_part[i]+1:n_per_part[i+1]],:])
        obs1=size(x1,1)
        # obs1=length(inds_all[n_per_part[i]+1:n_per_part[i+1]])
        # phi1 = calc_phi(x1,x,obs1,obs,cc,vars)
        # # layer1[:,i].=phi1*a_all[i,1:obs1]
        # layer1[:,i].=phi1*a_all[i]


        # phi=zeros(nn,n)
        # obs1=length(inds_all[n_per_part[i]+1:n_per_part[i+1]])
        for ii=1:obs
            phis=0.0
            for j=1:obs1
                dr=0
                for k=1:vars
                    dr = dr + (x[ii,k] - x1[j,k])^2
                    # dr = dr + (x[ii,k] - x[inds_all[n_per_part[i]+j],k])^2
                end
                phis+=exp(-dr/cc)*a_all[i][j]
                # phis+=-(dr^4)/4/cc
            end
            layer1[ii,i]=phis
        end







    end
    println(Dates.format(now(), "HH:MM:SS")," Solving system (",obs," by ",neurons+1,") for weights v in the output Layer")
    a_layer1=[layer1 ones(obs)]\y
    println(Dates.format(now(), "HH:MM:SS")," All weights computed.")

    return a_all,a_layer1,layer1

end
