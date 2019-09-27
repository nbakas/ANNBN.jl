function predict_new(a_all,a_layer1,x_new,obs,neurons)

    # layer1=zeros(obs,neurons)
    # for i=1:neurons
    #     if i in 1:100:neurons println(Dates.format(now(), "HH:MM:SS"),"applying sigmoid on neuron ",i," of ",neurons) end
    #     # layer1[:,i].=sigm1.([ones(obs) x_new]*a_all[i,:])
    #     layer1[:,i].=sigm1.([ones(obs) x_new]*a_all[i])
    #     # x[i_so,:]*a_all[i,:]
    # end
    # # a_layer1=[sigm1.(layer1) ones(obs)]\y
    # # predl1=[sigm1.(layer1) ones(obs)]*a_layer1
    # predl1=[layer1 ones(obs)]*a_layer1


    layer1=sigm1.([ones(obs) x_new]*reduce(hcat, a_all))
    predl1=[layer1 ones(obs)]*a_layer1

    return predl1,layer1

end
