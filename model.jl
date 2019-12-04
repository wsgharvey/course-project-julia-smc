using Flux
using BSON: @load
using Distributions

include("nn.jl")

# Components of recognition model / "encoder" MLP.
A, μ, logσ = Dense(28^2, Dh, tanh), Dense(Dh, Dz), Dense(Dh, Dz)
g(X) = (h = A(X); (μ(h), logσ(h)))
z(μ, logσ) = μ + exp(logσ) * randn(Float32)

# Generative model / "decoder" MLP.
f = Chain(Dense(Dz, Dh, tanh), Dense(Dh, 28^2, σ))
@load "ckpt/generator.bson" f
f_gpu = f |> gpu

function sample_prior(batchdim)
    global Dz
    return z.(zeros(batchdim, Dz), zeros(batchdim, Dz))
end

function logpdf_prior(z)
    return sum(Distributions.logpdf.(Distributions.Normal(), z), dims=2)[:, 1]
end

function logpdf_obs(obs_mask, obs, zz)
    """
    z should have a batch dimension
    """
    global f_gpu
    # p(z)
    obs_mask = obs_mask |> gpu
    obs = obs |> gpu
    zz = transpose(zz) |> gpu
    println(typeof(zz))
    x̂ = f_gpu(zz).data
    ŷ = reshape(obs_mask, size(obs_mask, 1), 1) .* x̂ |> cpu
    obs = obs_mask .* obs |> cpu
    # p(y | z)
    log_p_y_z = [sum(Distributions.logpdf.(Distributions.Bernoulli.(ŷ[:, i]), obs))
                 for i = 1:size(ŷ, 2)]
    return log_p_y_z
end

function cpu_logpdf_obs(obs_mask, obs, zz)
    """
    z should have a batch dimension
    """
    global f
    # p(z)
    obs_mask = obs_mask
    obs = obs
    zz = transpose(zz)
    println(typeof(zz))
    x̂ = f(zz).data
    ŷ = reshape(obs_mask, size(obs_mask, 1), 1) .* x̂
    obs = obs_mask .* obs
    # p(y | z)
    log_p_y_z = [sum(Distributions.logpdf.(Distributions.Bernoulli.(ŷ[:, i]), obs))
                 for i = 1:size(ŷ, 2)]
    return log_p_y_z
end

# Image saving

using Images

function save_images(zs, fname, obs, obs_mask)
    cd(@__DIR__)
    to_img(x) = Gray.(reshape(x, 28, 28))
    image = hcat(to_img.([f(zs[i, :]).*(1. .- obs_mask).+obs.*obs_mask
                          for i = 1:size(zs, 1)])...)
    save(fname, image)
end
