
__precompile__(false)
module ANNBN
    # __precompile__()
    using Plots, Statistics, Dates, LinearAlgebra, Random, SpecialFunctions, Clustering, IterativeSolvers
    using Base.Threads
    using CUDAdrv, CUDAnative, CuArrays
    # __precompile__()

    function sigm1(x) 1.0/(1.0+exp(-x)) end
    function isigm1(x) -log((1.0-x)/x) end
    # function sigm1(x) atan(x) end
    # function isigm1(x) tan(x) end
    # function sigm1(x) 2atan(pi*x/2)/pi end
    # function isigm1(x) 2tan(pi*x/2)/pi end
    # function sigm1(x) x^2 end
    # function isigm1(x) sqrt(x) end
    # function sigm1(x) max(0,x) end
    # function isigm1(x) if x>0.1 return x else return -1.0 end end
    # function sigm1(x) log(1.0+exp(x)) end
    # function isigm1(x) log(exp(x)-1.0) end
    # function sigm1(x) if x<-1.0 -1.0 elseif x<1.0 x else 1.0 end end
    # function isigm1(x) if x<-1.0 -10.0 elseif x<1.0 x else 10.0 end end
    # function sigm1(x) 1.0/(1.0+exp(-1000.0x)) end
    # function isigm1(x) -log((1.0-1000.0x)/1000.0x) end
    # function sigm1(x) erf(x) end
    # function isigm1(x) erfi(x) end
    # function sigm1(x) exp(x) end
    # function isigm1(x) log(x) end
    # function sigm1(x) x^3 end
    # function isigm1(x) x^(1/3) end


    include("___clustering.jl")
    # include("new_clustering.jl")
    # include("___augment_data.jl")

    include("calc_phi.jl")
    include("calc_phi_deriv.jl")
    include("calc_phi_deriv_deriv.jl")
    include("predict_new.jl")
    include("predict_new_rbf.jl")
    include("predict_new_rbf_deriv.jl")
    include("predict_new_rbf_deriv_deriv.jl")
    include("train_layer_1_rbf.jl")
    include("train_layer_1_rbf_pde.jl")
    include("train_layer_1_sigmoid.jl")
    include("train_layer_1_sigmoid_fast.jl")
    include("train_layer_1_sigmoid_fast2.jl")
    include("train_layer_1_rbf_fast.jl")
    include("train_layer_1_rbf_deriv.jl")

    include("fit_nfolds!.jl")
    include("predict_nfolds.jl")
    include("fit_nfolds_sigmoid.jl")
    include("predict_nfolds_sigmoid.jl")

    include("deep_nnm.jl")
    include("initialize_deep_weights_per_layer.jl")
    # include("run_iit.jl")

    # include("train_layer_1_rbf_fast.jl")
    include("train_layer_1_rbf_new.jl")

    export ___clustering,new_clustering,
    calc_phi,calc_phi_deriv,calc_phi_deriv_deriv,
    predict_new,predict_new_rbf,predict_new_rbf_deriv,predict_new_rbf_deriv_deriv,
    train_layer_1_rbf,train_layer_1_rbf_pde,train_layer_1_sigmoid,
    fit_nfolds!,predict_nfolds,fit_nfolds_sigmoid,predict_nfolds_sigmoid,train_layer_1_rbf_fast,
    deep_nnm,initialize_deep_weights_per_layer,run_iit,train_layer_1_rbf_new,train_layer_1_rbf_deriv,
	train_layer_1_sigmoid_fast,train_layer_1_sigmoid_fast2



    # precompile(sigm1, (Float64,))
    # precompile(isigm1, (Float64,))
    # precompile(deep_nnm, (Array{Float64,1},Array{Float64,2},Array{Float64,2},Int64,Int64,Int64))
    # precompile(calc_phi, (Array{Float64,2},Array{Float64,2},Int64,Int64,Float64,Int64))


end
