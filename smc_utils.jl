import Distributions

function logsumexp(w, keep_dims::Bool)
    """
    w is a B x S array
    sum over S dimension (and keeps it as a single-element dimension)
    """
    lastdim = ndims(w)
    maxw = maximum(w, dims=lastdim)
    result = log.(sum(exp.(w.-maxw), dims=lastdim)).+maxw
    if keep_dims
        return result
    else
        return dropdims(result, dims=lastdim)
    end
end

function ESS(logw, comm)
    rank = MPI.Comm_rank(comm)
    if rank != 0
        return 0
    end
    w = exp.(logw)
    return sum(w)^2 / sum(w.^2)
end

function resample_indices(logw)
    logZ = logsumexp(logw, true)
    p = exp.(logw.-logZ)
    cat = Distributions.Categorical(p)
    return rand(cat, size(p))
end
