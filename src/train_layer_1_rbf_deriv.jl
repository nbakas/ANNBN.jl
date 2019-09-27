function train_layer_1_rbf_deriv(neurons,vars,obs,n_per_part,inds_all,x,y,cc)
    a_all=Vector{Vector{Float64}}()
    ooops=0
    errs=Array{Float64}(undef,0)
    obs1=50
    itmps=Vector{Vector{Int64}}()
    for i=1:neurons
        if i in 1:1:neurons println(Dates.format(now(), "HH:MM:SS")," Calculating weights w for ",i," neuron, of ",neurons," Total.") end
        # try

            # x1=copy(x[inds_all[n_per_part[i]+1:n_per_part[i+1]],:])
            # obs1=size(x1,1)
            # phi1 = calc_phi(x1,x1,obs1,obs1,cc,vars)
            # atmp=phi1\y[inds_all[n_per_part[i]+1:n_per_part[i+1]]]
            # # phi1_deriv = calc_phi_deriv(x1,x1,obs1,obs1,cc,vars,rand(1:vars))
            # # println(sum(abs.(atmp[1:end])))
            # # atmp=[phi1;phi1_deriv]\[y[inds_all[n_per_part[i]+1:n_per_part[i+1]]];zeros(obs1)]


            itmp=randperm(obs)[1:obs1]
            x1=copy(x[itmp,:])
            phi1 = calc_phi(x1,x1,obs1,obs1,cc,vars)
            phi1_deriv = calc_phi_deriv(x1,x1,obs1,obs1,cc,vars,rand(1:vars))
            atmp=[phi1;phi1_deriv]\[y[itmp];zeros(obs1)]
            err1=mean(abs.(phi1*atmp-y[itmp]))


            if sum(isnan.(atmp))==0 && sum(isinf.(atmp))==0 && sum(abs.(atmp[1:end]))>0
                push!(a_all,atmp)
                push!(itmps,itmp)
                push!(errs,err1)
            else ooops+=1 end
        # catch ex
        #     ooops+=1;
        #     # println(i," ",ex)
        # end
    end
    println("ooops=",ooops)

    ii1=sortperm(errs)[1:2000]
    a_all=a_all[ii1]
    itmps=itmps[ii1]

    return a_all,itmps
end
