using Plots, Statistics, MLDatasets, LinearAlgebra, Clustering, Printf, Dates, Random
if !(pwd() in LOAD_PATH) push!(LOAD_PATH, pwd()) end
using ANNBN


##### PARTIAL DIFFERENTIAL EQUATIONS
# Laplace's Equation http://www.robots.ox.ac.uk/~jmb/lectures/pdelecture5.pdf
a=1;b=1;n=50;dx=a/n;x0=zeros(n); x=Array{Float64}(undef,0); for i=1:n-1 global x0,x,dx; x=[x;x0.+(i)*dx] end; x
ny=50;dy=b/n;y=convert(Array{Float64},range(dy,b-dy,length=ny)); y=repeat(y,n-1)
xx_train=[x y];i_train=size(xx_train,1);vars=2;
# boundaries
nof_b=20;xb1=zeros(nof_b,2);xb1[:,2]=range(0,1,length=nof_b);xb2=zeros(nof_b,2);xb2[:,1]=range(0,1,length=nof_b);
xb3=ones(nof_b,2);xb3[:,2]=range(0,1,length=nof_b);xb4=ones(nof_b,2);xb4[:,1]=range(0,1,length=nof_b)
xb=[xb1;xb2;xb3;xb4];yb=zeros(size(xb,1));yb[end-nof_b+1:end].=sin.(pi.*range(0,1,length=nof_b))
# source 
y_source=zeros(i_train).+0.0(rand(i_train)) # you ay adjust the level of noise here


##### Clustering
neurons=50;inds_all,n_per_part=___clustering(neurons,xx_train,200)

##### Solve PDE
# Use # 1 Gaussian for calc_phi,calc_phi_deriv,calc_phi_deriv_deriv
cc1=1.0
a_all,a_layer1,layer1=ANNBN.train_layer_1_rbf_pde(neurons,vars,i_train,n_per_part,inds_all,xx_train,cc1,xb,yb,y_source);
predl1,layer1_train=ANNBN.predict_new_rbf(a_all,a_layer1,xx_train,i_train,neurons,n_per_part,inds_all,xx_train,vars,cc1)
scatter3d(xx_train[1:1:end,1],xx_train[1:1:end,2],predl1[1:1:end],label="ANNBN solution",
    color=:black,markershape=:xcross,legend=:topleft,markersize=1)
# Analytical solution
an_sol=sinh.(pi*xx_train[:,2]./a).*sin.(pi*xx_train[:,1]./a)./sinh(pi*b/a)
scatter3d!(xx_train[:,1],xx_train[:,2],an_sol,label="Analytical solution",
    color=:black,markershape=:star5,legend=:topleft,markersize=1)
# savefig("Laplace.pdf")
err=an_sol .- predl1;mean(abs.(err))
# scatter3d(xx_train[:,1],xx_train[:,2],err,color=:black,markershape=:xcross,legend=false,markersize=1)
# savefig("Laplace_noise_err.pdf")

# Calculate Derivatives
predl1_deriv_deriv1=ANNBN.predict_new_rbf_deriv_deriv(a_all,a_layer1,xx_train,i_train,neurons,
            n_per_part,inds_all,xx_train,vars,cc1,1)
predl1_deriv_deriv2=ANNBN.predict_new_rbf_deriv_deriv(a_all,a_layer1,xx_train,i_train,neurons,
            n_per_part,inds_all,xx_train,vars,cc1,2)
err_2nd_derivatives=predl1_deriv_deriv1+predl1_deriv_deriv2.-0.0
mean(abs.(err_2nd_derivatives))
scatter3d(xx_train[:,1],xx_train[:,2],predl1_deriv_deriv1,label="Derivative fxx",
    color=:black,markershape=:xcross,legend=:topleft,markersize=1,camera=(70, 40)
    ,xlabel="x",ylabel="y")
scatter3d!(xx_train[:,1],xx_train[:,2],predl1_deriv_deriv2,label="Derivative fyy",
    color=:black,markershape=:star5,legend=:topleft,markersize=1)
scatter3d!(xx_train[:,1],xx_train[:,2],predl1_deriv_deriv1+predl1_deriv_deriv2,label="fxx + fyy",
    color=:black,markershape=:circle,legend=:topleft,markersize=1)
# savefig("Laplace_fxx_fyy.pdf") 