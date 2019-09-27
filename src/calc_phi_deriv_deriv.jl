function calc_phi_deriv_deriv(x,xi,n,nn,c,vars,var_seq)
    phi_der=zeros(nn,n)
    for i=1:nn
        for j=1:n
            dr=0
            for k=1:vars
                dr = dr + (xi[i,k] - x[j,k])^2
            end
            r = (dr)

            # 2 r^4
            # phi_der[i,j] = -12*((xi[i,var_seq]-x[j,var_seq])^2)*(r^2) - 2*(r^3)
        

            # 1 Gaussian
            phi_der[i,j]=-2*exp(-r/c)/c + ((-2*xi[i,var_seq] + 2*x[j,var_seq])^2)*exp(-r/c)/(c^2)
        
        end
    end
    return phi_der
end


# phi[i,j]=exp(-dr^2/c^2)*(-2*dr/c)
# phi(i,j) = exp(-r^2)/2 + (r*pi^(1/2)*erf(r))/2;
# phi[i,j] = 2*(1./(1+exp.(-r)))-1;
# phi[i,j] = (-2*r/c[i,j])*exp(-r^2/c[i,j])
# phi[i,j]=tanh.(r/c[i,j])
# phi_der[i,j] = -(r^2)*(2*(xi[i,var_seq]-x[j,var_seq]))
# phi_der[i,j] = -2*exp(-r/c)/c + ((-2*xi[i,var_seq] + 2*x[j,var_seq])^2)*exp(-r/c)/(c^2)
# phi_der[i,j] = exp(-r/c^2)