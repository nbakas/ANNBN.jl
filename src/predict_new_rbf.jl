function predict_new_rbf(a_all,a_layer1,x_new,obs,neurons,n_per_part,inds_all,x_old,vars,cc)
    println("hello")
    layer1=zeros(obs,neurons)
    for i=1:neurons
        if i in 1:1:neurons println(Dates.format(now(), "HH:MM:SS")," Predicting responce of ",i," neuron, of ",neurons," Total.") end
        x1=copy(x_old[inds_all[n_per_part[i]+1:n_per_part[i+1]],:])
        obs1=size(x1,1)
        phi1 = calc_phi(x1,x_new,obs1,obs,cc,vars)
        # layer1[:,i].=phi1*a_all[i,1:obs1]
        layer1[:,i].=phi1*a_all[i]

    end

    predl1=[layer1 ones(obs)]*a_layer1

    return predl1,layer1

end
