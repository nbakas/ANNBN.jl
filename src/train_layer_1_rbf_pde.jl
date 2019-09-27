function train_layer_1_rbf_pde(neurons,vars,obs,n_per_part,inds_all,x,cc,xb,yb,y_source)


    # a_all=zeros(neurons,1000vars)
    a_all=Vector{Vector{Float64}}()
    ooops=0
    for i=1:neurons
        if n_per_part[i+1]-n_per_part[i]!=0
            try 
                x1=copy(x[inds_all[n_per_part[i]+1:n_per_part[i+1]],:])
                obs1=size(x1,1)
                phi_x = calc_phi_deriv_deriv(x1,x1,obs1,obs1,cc,vars,1)
                phi_y = calc_phi_deriv_deriv(x1,x1,obs1,obs1,cc,vars,2)
                phi = calc_phi(x1,xb,obs1,size(xb,1),cc,vars)
                atmp=[(phi_x+phi_y);phi]\[y_source[inds_all[n_per_part[i]+1:n_per_part[i+1]]];yb]             
                # if sum(isnan.(atmp))==0 && sum(isinf.(atmp))==0 a_all[i,1:obs1].=atmp else ooops+=1 end             
                if sum(isnan.(atmp))==0 && sum(isinf.(atmp))==0 push!(a_all,atmp) else ooops+=1 end             
            catch ex
                ooops+=1; println(i," ",ex); break
            end
        end
    end
    println("ooops=",ooops)

    layer1=zeros(obs+size(yb,1),neurons)
    for i=1:neurons
        x1=copy(x[inds_all[n_per_part[i]+1:n_per_part[i+1]],:])
        obs1=size(x1,1)
        phi_x = calc_phi_deriv_deriv(x1,x,obs1,obs,cc,vars,1)
        phi_y = calc_phi_deriv_deriv(x1,x,obs1,obs,cc,vars,2)
        phi = calc_phi(x1,xb,obs1,size(xb,1),cc,vars)
        # layer1[:,i].=[(phi_x+phi_y);phi]*a_all[i,1:obs1]
        layer1[:,i].=[(phi_x+phi_y);phi]*a_all[i]
    end

    xxxx=[layer1 ones(obs+size(yb,1))]
    yyyy=[y_source;yb]
    a_layer1=xxxx\yyyy

    return a_all,a_layer1,layer1

end