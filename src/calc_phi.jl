function calc_phi(x::Array{Float64,2},xi::Array{Float64,2},n::Int64,nn::Int64,c::Float64,vars::Int64)
  phi=zeros(nn,n)
  for i=1:nn
      for j=1:n
          # 2 r^4
          dr=0
          for k=1:vars
            dr = dr + (xi[i,k] - x[j,k])^2
          end
          # r = (dr)
          # phi[i,j] = -(dr^4)/4/c


          # 1 Gaussian
          # phi[i,j] = exp(-sum((xi[i,:] - x[j,:]).^2)/c)
          phi[i,j] = exp(-dr/c)
      end
      # if i in 1:100:nn println(Dates.format(now(), "HH:MM:SS "),i," of ",nn) end
  end


  # xi_all=Array{Float64,3}(undef, nn,1,vars)
  # xi_all[:,1,:]=xi
  # # println(size(xi_all))
  # xi_all=repeat(xi_all,1,n,1)
  # # println(size(xi_all))

  # x_all=Array{Float64,3}(undef, n,1,vars)
  # x_all[:,1,:]=x
  # # println(size(x_all))
  # x_all=repeat(x_all,1,nn,1)
  # # println(size(x_all))
  # x_all = permutedims(x_all, [2, 1, 3])
  # # println(size(x_all))


  # # @time m1=sum((xi_all - x_all).^2,dims=3)
  # # # println(size(m1))
  # phi2=exp.(-sum((xi_all - x_all).^2,dims=3)/c)[:,:,1]
  # # println(size(phi2))
  # # println(mean(abs.(phi-phi2)))


  return phi
end



# phi[i,j] = exp(-sqrt(dr)/c)
# phi[i,j] = exp(-r^2)/2 + (r*pi^(1/2)*erf(r))/2;
# phi[i,j] = 2*(1./(1+exp.(-r)))-1;
# phi[i,j] = exp(-r^2/c[i,j])
# phi[i,j] = exp(-dr/c)
# phi[i,j]=tanh.(r^2/c[i,j])
# phi[i,j] = -(r^3)/3
# phi[i,j] = exp(-r/c)
# phi[i,j] =(c^2*exp(-r/c^2))/2 + (c*sqrt(pi)*erf(sqrt(r)/c)*sqrt(r))/2;
# phi[i,j] = -sum((xi[i,:] - x[j,:]).^2)
