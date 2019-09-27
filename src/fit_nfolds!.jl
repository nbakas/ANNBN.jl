function fit_nfolds!(xx_train,yy_train,nof_folds,vars,i_train,i_fold,cc1,neurons)# ,inds_all,n_per_part,neurons)
    maetrs=Array{Float64}(undef,0)
    a_all_all=Vector{Vector{Vector{Float64}}}()
    a_layer1_all=Vector{Vector{Float64}}()
    n_per_part_all=Vector{Vector{Int64}}()
    inds_all_all=Vector{Vector{Int64}}()
    xx_fold_all=Vector{Array{Float64}}()
    neurons_all=Vector{Int64}()
    pred_folds_all=zeros(i_fold,nof_folds)
    for i=1:nof_folds
        inds_fold=randperm(i_train)[1:i_fold]
        # inds_fold=1:i_fold
        xx_fold=copy(xx_train[inds_fold,:]);push!(xx_fold_all,xx_fold);
        yy_fold=copy(yy_train[inds_fold])
        # neurons=Int64(floor(i_fold/(vars+1)))   # todo: change number of neurons for each fold
        # neurons=10
        # push!(neurons_all,neurons)
        inds_all_vec_unvec,n_per_part_new=___clustering(neurons,xx_fold,200)
        # inds_all=randperm(i_fold);items_per_neuron=(Int64(floor(i_fold/(neurons))))*ones(Int64,neurons)
        # n_per_part=[0;cumsum(items_per_neuron)];n_per_part[end]=i_train
		
		
		inds_fold=randperm(i_train)[1:i_fold];
        # xx_fold=copy(xx_train[inds_fold,:]);push!(xx_fold_all,xx_fold);
        # yy_fold=copy(yy_train[inds_fold])
        # neurons=Int64(floor(i_fold/(vars+1)))   # todo: change number of neurons for each fold
        # neurons=20
        push!(neurons_all,neurons);
        # inds_all,n_per_part=___clustering(neurons,xx_fold,200)


        # n_per_part_new=copy(n_per_part);
        # inds_all_vec=Vector{Vector{Int64}}();
        # k=0
        # for ii=1:length(n_per_part)-1
        #     iii=inds_all[n_per_part[ii]+1:n_per_part[ii+1]];
        #     vec1=Vector{Int64}();
        #     for j=1:length(iii)
        #         if iii[j] in inds_fold 
        #             k+=1
        #             # println(k," ",iii[j])
        #             push!(vec1,k); 
        #         end
                
        #     end
        #     n_per_part_new[ii+1]=n_per_part_new[ii]+length(vec1);
        #     push!(inds_all_vec,vec1);
        # end
        # inds_all_vec_unvec=Vector{Int64}();
        # for ii=1:length(n_per_part_new)-1
        #     inds_all_vec_unvec=[inds_all_vec_unvec;inds_all_vec[ii]];
        # end
        # xx_fold=copy(xx_train[inds_fold,:]);push!(xx_fold_all,xx_fold);
        # yy_fold=copy(yy_train[inds_fold]);

        # println("training ",i)
        # println(n_per_part_new)
        # println(inds_all_vec_unvec)
        # println(size(inds_all_vec_unvec),size(n_per_part_new),size(inds_all_vec))
        
        # return maetrs,a_all_all,a_layer1_all,n_per_part_all,inds_all_all,xx_fold_all,neurons_all




        a_all,a_layer1,layer1=train_layer_1_rbf(neurons,vars,i_fold,n_per_part_new,inds_all_vec_unvec,xx_fold,yy_fold,cc1)
        push!(a_all_all,a_all);push!(a_layer1_all,a_layer1);push!(n_per_part_all,n_per_part_new);push!(inds_all_all,inds_all_vec_unvec);

        # predl1=predict_new_rbf(a_all,a_layer1,xx_train,i_train,neurons,n_per_part,inds_all,xx_fold,vars,cc1)
        # # i1=predl1.<=0.5;predl1[i1].=0.1;i1=predl1.>0.5;predl1[i1].=0.9;i2=(predl1.-yy_train).<0.01;maetr=sum(i2)/length(i2);
        # maetr=mean(abs.(yy_train-predl1))
        # push!(maetrs,maetr); 

        predl1=[layer1 ones(i_fold)]*a_layer1
        pred_folds_all[:,i].=predl1
        maetr=mean(abs.(yy_fold-predl1))
        # i1=predl1.<=0.5;predl1[i1].=0.1;i1=predl1.>0.5;predl1[i1].=0.9;i2=abs.(predl1.-yy_fold).<0.01;maetr=1.0-sum(i2)/length(i2)
        # if maetr<1e-4 maetr=1e-4 end
        push!(maetrs,maetr); 


        




        println(Dates.format(now(), "HH:MM:SS"), " ",i, " maetr=",maetr)
    end
    return maetrs,a_all_all,a_layer1_all,n_per_part_all,inds_all_all,xx_fold_all,neurons_all,pred_folds_all
end