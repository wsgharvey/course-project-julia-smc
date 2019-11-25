using MPI
import Distributions

include("mpi_utils.jl")
include("smc_utils.jl")

#=
We have a series of distributions 0...T, represented by
1. sampler for distribution 0 and
2. unnormalised likelihood functions for each distribution thereafter.
=#

function collect_weights(logw, comm, δlogw)
    return logw + collect(δlogw, comm)
end

function resample(logw, samples, comm)
    rank = MPI.Comm_rank(comm)
    all_samples = collect(samples, comm)

    if rank != 0
        rreq = MPI.Irecv!(samples, 0, rank+32, comm)
        MPI.Waitall!([rreq])
        return 0
    end

    new_indices = resample_indices(logw)
    latentdim = size(samples, 2)
    for worker in 1:(n_processes-1)
        worker_samples = Array{Float64}(undef, particles_per_process, latentdim)
        for sample in 1:particles_per_process
            full_index = sample + worker*particles_per_process
            worker_samples[sample] = all_samples[new_indices[full_index]]
        end
        MPI.Send(worker_samples, worker, worker+32, comm)
    end
    return 0*logw
end

# define series of distributions -----------------------------------------
prior = Distributions.Normal(1, sqrt(5))
function sample_prior(batchsize)
    return rand(prior, batchsize, 1)
end
function logpdf(alpha::Float64, x)
    #=
    alpha in [0, 1]. alpha = 0 is prior, alpha = 1 is final dist.
    =#
    priorpdf = Distributions.logpdf.(prior, x[:, 1])
    function finalpdf(x)
        return logsumexp(hcat(Distributions.logpdf.(Distributions.Normal(-10., 0.5), x[:, 1]),
                              Distributions.logpdf.(Distributions.Normal(10., 0.5), x[:, 1])),
                         false)
    end
    return alpha*finalpdf(x) + (1-alpha)*priorpdf
end

# do SMC -----------------------------------------------------------------
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n_processes = MPI.Comm_size(comm)
particles_per_process = 4
n_particles = particles_per_process * n_processes

δlogw = Array{Float64}(undef, 1)
δlogw[1] = 1
if rank == 0
    logw = zeros(n_particles)
else
    logw = 0
end

samples = sample_prior(particles_per_process)

δα = 0.1
for α in δα:δα:1
    global logw
    δlogw = logpdf(α, samples) - logpdf(α-δα, samples)
    logw = collect_weights(logw, comm, δlogw)
    if rank == 0
        print("$samples $logw \n")
    end

    logw = resample(logw, samples, comm)
end

MPI.Barrier(comm)
