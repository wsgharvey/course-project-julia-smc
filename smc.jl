using MPI
import Distributions

include("mpi_utils.jl")
include("smc_utils.jl")
include("model.jl")

#=
We have a series of distributions 0...T, represented by
1. sampler for distribution 0 and
2. unnormalised likelihood functions for each distribution thereafter.
=#

# high-level functions ---------------------------------------------------

function collect_weights(logw, comm, δlogw)
    return logw + collect(δlogw, comm)
end

function resample(logw, samples, comm)
    all_samples = collect(samples, comm)
    if rank == 0
        new_indices = resample_indices(logw)
    else
        new_indices = 0
    end
    samples = share_out(all_samples, new_indices, size(samples), comm)
    return (samples, 0*logw)
end

function maybe_resample(logw, samples, comm)
    rank = MPI.Comm_rank(comm)
    if rank == 0
        ess = ESS(logw)
        dont_resample = ess > size(logw, 1)/2
    else
        dont_resample = true
    end
    dont_resample = send_bool(dont_resample, comm)
    if dont_resample
        return (samples, logw)
    end
    return resample(logw, samples, comm)
end

function rejuvenate(samples, logpdf_func)
    logpdf1 = logpdf_func(samples)
    perturb_dist = Distributions.Normal(0, 0.1)
    proposed_samples = samples + rand(perturb_dist, size(samples))
    logpdf2 = logpdf_func(proposed_samples)
    α = exp.(logpdf2-logpdf1)
    accept = rand(Distributions.Uniform(), size(α)) .< α
    samples = proposed_samples.*accept + samples.*(1 .- accept)
    return samples
end

# define series of distributions -----------------------------------------
function logpdf(alpha::Float64, x)
    # select observations
    obs = 1
    obs_mask = zeros(784)
    obs_mask[1+28*10:28*11] = ones(28)
    obs = zeros(784)
    obs[1+28*10:28*11] = cat(zeros(8), ones(4), zeros(4), ones(4), zeros(8), dims=1)
    # calculate stuff
    return logpdf_prior(x) .+ alpha*logpdf_obs(obs_mask, obs, x)
end


# do SMC -----------------------------------------------------------------
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n_processes = MPI.Comm_size(comm)
particles_per_process = 3
n_particles = particles_per_process * n_processes

δlogw = Array{Float64}(undef, 1)
δlogw[1] = 1
if rank == 0
    logw = zeros(n_particles)
else
    logw = 0
end

samples = sample_prior(particles_per_process)

δα = 0.04
for α in δα:δα:1
    global logw, samples
    δlogw = logpdf(α, samples) - logpdf(α-δα, samples)
    logw = collect_weights(logw, comm, δlogw)
    samples, logw = maybe_resample(logw, samples, comm)

    logpdf_func(x) = logpdf(α, x)
    samples = rejuvenate(samples, logpdf_func)

end

samples, _ = resample(logw, samples, comm)

samples = collect(samples, comm)
if rank == 0
    println(samples, '\n')
end

MPI.Barrier(comm)
