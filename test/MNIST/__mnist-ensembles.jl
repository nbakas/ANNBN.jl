

predl1_train_ens=Array{Float64}(undef,i_train,0)
predl1_test_ens=Array{Float64}(undef,i_test,0)
maetrs=Vector{Float64}(undef,0)
maetes=Vector{Float64}(undef,0)
for i=1:5
    global predl1_train_ens,predl1_test_ens,maetrs,maetes
    inds_all=1:i_train
    neurons=7000
    items_per_neuron=(Int64(floor(i_train/(neurons))))*ones(Int64,neurons);n_per_part=[0;cumsum(items_per_neuron)];n_per_part[end]=i_train
    @time include("~\\mnist_distort.jl")
    @time a_all,a_layer1,layer1=ANNBN.train_layer_1_sigmoid_fast(neurons,vars,i_train,n_per_part,inds_all,xx_train2,yy_train)
    predl1=[layer1 ones(i_train)]*a_layer1
    maetr=mean(abs.(yy_train-predl1))
    maximum(abs.(yy_train-predl1))
    predl1_test,layer1_test=ANNBN.predict_new(a_all,a_layer1,xx_test2,i_test,neurons);
    maete=mean(abs.(yy_test-predl1_test))
    maximum(abs.(yy_test-predl1_test))


    predl1_train_ens=[predl1_train_ens predl1]
    predl1_test_ens=[predl1_test_ens predl1_test]

    i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<1e-10;maetr=100sum(i2)/length(i2)
    i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<1e-10;
    maete=100sum(i2)/length(i2)

    println(i," ",maetr," ",maete)



    predl1=mean(predl1_train_ens,dims=2)
    maetr=mean(abs.(yy_train-predl1))
    maximum(abs.(yy_train-predl1))
    predl1_test=mean(predl1_test_ens,dims=2)
    maete=mean(abs.(yy_test-predl1_test))
    maximum(abs.(yy_test-predl1_test))
    i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<1e-10;maetr=100sum(i2)/length(i2)
    i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<1e-10;
    maete=100sum(i2)/length(i2)
    println(i," ",maetr," ",maete)
    push!(maetrs,maetr)
    push!(maetes,maete)
    # display(plot(maetrs))
    # display(plot!(maetes))


end

predl1=mean(predl1_train_ens,dims=2)
maetr=mean(abs.(yy_train-predl1))
maximum(abs.(yy_train-predl1))
predl1_test=mean(predl1_test_ens,dims=2)
maete=mean(abs.(yy_test-predl1_test))
maximum(abs.(yy_test-predl1_test))
i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<1e-10;maetr=100sum(i2)/length(i2)
i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<1e-10;
maete=100sum(i2)/length(i2)
