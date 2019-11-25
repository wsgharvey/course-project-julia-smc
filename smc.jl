using MPI
import Distributions


#=
We have a series of distributions 0...T, represented by
1. sampler for distribution 0 and
2. unnormalised likelihood functions for each distribution thereafter.
=#

function logsumexp(w)
    """
    w is a B x S array
    does sum over S dimension
    """
    maxw = maximum(w, dims=2)
    return log.(sum(exp.(w.-maxw), dims=2)).+maxw
end

function collect_weights(logw, comm, δlogw)
    rank = MPI.Comm_rank(comm)
    n_processes = MPI.Comm_size(comm)
    if rank != 0
        MPI.Send(δlogw, 0, rank+32, comm)
        return 0
    else
        particles_per_process = size(δlogw, 1)
        logw[1:particles_per_process] = logw[1:particles_per_process]+δlogw
        received = Array{Float64}(undef, particles_per_process)
        for worker in 1:(n_processes-1)       # awful
            rreq = MPI.Irecv!(received, worker, worker+32, comm)
            MPI.Waitall!([rreq])
            logw[1+(worker*particles_per_process):(worker+1)*particles_per_process] =
                logw[1+(worker*particles_per_process):(worker+1)*particles_per_process]+received
        end
        return logw
    end
end

function update_values()
    return 0
end

# define series of distributions -----------------------------------------
prior = Distributions.Normal(1, sqrt(5))
function logpdf(alpha::Float64, x)
    #=
    alpha in [0, 1]. alpha = 0 is prior, alpha = 1 is final dist.
    =#
    priorpdf = Distributions.logpdf.(prior, x)
    function finalpdf(x)
        return logsumexp(hcat(Distributions.logpdf.(Distributions.Normal(-10., 0.5), x),
                              Distributions.logpdf.(Distributions.Normal(10., 0.5), x)))
    end
    return alpha*finalpdf(x) + (1-alpha)*priorpdf
end

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n_processes = MPI.Comm_size(comm)
particles_per_process = 2
n_particles = particles_per_process * n_processes

δlogw = Array{Float64}(undef, 1)
δlogw[1] = 1
logw = zeros(n_particles)

samples = rand(prior, particles_per_process)

δα = 0.1
for α in δα:δα:1
    global logw
    δlogw = logpdf(α, samples) - logpdf(α-δα, samples)
    logw = collect_weights(logw, comm, δlogw)
    if rank == 0
        println(logw)
    end
end

MPI.Barrier(comm)
