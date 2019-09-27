function predict_new_rbf_deriv(a_all,a_layer1,x_new,obs,neurons,n_per_part,inds_all,x_old,vars,cc,var_seq)

    layer1=zeros(obs,neurons)
    for i=1:neurons
        x1=copy(x_old[inds_all[n_per_part[i]+1:n_per_part[i+1]],:])
        obs1=size(x1,1)
        phi1 = calc_phi_deriv(x1,x_new,obs1,obs,cc,vars,var_seq)
        # if sum(abs.(a_all[i,1:obs1]))>0 
        #     layer1[:,i].=phi1*a_all[i,1:obs1]
        # end
        layer1[:,i].=phi1*a_all[i]
    end

    predl1=[layer1 ones(obs)]*a_layer1

    return predl1

end