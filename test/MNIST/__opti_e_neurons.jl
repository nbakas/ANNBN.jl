

# using Base.Threads
#
# nthreads()

neurons_all=Vector{Int64}(undef,0)
ee_all=Vector{Float64}(undef,0)
maetes=Vector{Float64}(undef,0)
for i=7000
    global neurons_all,ee_all,maetes
    for j=0.01:0.01:0.1
        tol1=copy(j)
        num_to_test=0
        yy_train=copy(yy_train_all)
        i1=abs.(yy_train.-num_to_test).<0.01;i2=abs.(yy_train.-num_to_test).>0.01;yy_train[i1].=1;;yy_train[i2].=0;
        mi=minimum(yy_train);yy_train.-=mi;ma=maximum(yy_train);yy_train./=ma;yy_train.*=1.0-2tol1;yy_train.+=tol1

        yy_test=copy(yy_test_all)
        i1=abs.(yy_test.-num_to_test).<0.01;i2=abs.(yy_test.-num_to_test).>0.01;yy_test[i1].=1;yy_test[i2].=0;
        yy_test.-=mi;yy_test./=ma;yy_test.*=1.0-2tol1;yy_test.+=tol1


        inds_all=1:i_train
        neurons=copy(i);items_per_neuron=(Int64(floor(i_train/(neurons))))*ones(Int64,neurons);n_per_part=[0;cumsum(items_per_neuron)];n_per_part[end]=i_train
        @time a_all,a_layer1,layer1=ANNBN.train_layer_1_sigmoid_fast(neurons,vars,i_train,n_per_part,inds_all,xx_train,yy_train)
        predl1=[layer1 ones(i_train)]*a_layer1
        maetr=mean(abs.(yy_train-predl1))
        maximum(abs.(yy_train-predl1))
        predl1_test,layer1_test=ANNBN.predict_new(a_all,a_layer1,xx_test,i_test,neurons);
        maete=mean(abs.(yy_test-predl1_test))
        maximum(abs.(yy_test-predl1_test))
        i1=predl1.<=0.5;predl1[i1].=tol1;i1=predl1.>0.5;predl1[i1].=1.0-tol1;i2=abs.(predl1.-yy_train).<1e-10;maetr=100sum(i2)/length(i2)
        i1=predl1_test.<=0.5;predl1_test[i1].=tol1;i1=predl1_test.>0.5;predl1_test[i1].=1.0-tol1;i2=abs.(predl1_test.-yy_test).<1e-10;
        maete=100sum(i2)/length(i2)

        println(i," ",tol1," ",maete)
        push!(neurons_all,i)
        push!(ee_all,tol1)
        push!(maetes,maete)

    end
end
