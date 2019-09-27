function deep_nnm(tmp_weights::Array{Float64,1},tmp_xx::Array{Float64,2},nodes::Array{Int64,2},obs::Int64,vars::Int64,vars_depe::Int64)

    n_bias=sum(nodes)+vars_depe
    w_bias=(tmp_weights[end-n_bias+1:end])
    w1=(tmp_weights[1:vars*nodes[1]])
    w11=reshape(w1,vars,nodes[1])
    a_sig=sigm1.(tmp_xx*w11+ones(obs,nodes[1]).*w_bias[1:nodes[1]]')
    
    ind=length(w1)
    for ii=2:length(nodes)
        w1=(tmp_weights[ind+1:ind+nodes[ii-1]*nodes[ii]])
        ind+=length(w1)
        w11=reshape(w1,nodes[ii-1],nodes[ii])
        a_sig=sigm1.(a_sig*w11+ones(obs,nodes[ii]).*w_bias[sum(nodes[1:ii-1])+1:sum(nodes[1:ii])]')
    end

    w1=(tmp_weights[ind+1:end-n_bias])
    w11=reshape(w1,nodes[end],vars_depe)
    out=(a_sig*w11+ones(obs,vars_depe).*w_bias[end-vars_depe+1:end]')

    return out

end

#
