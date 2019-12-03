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
    function single_logpdf_joint(obs_mask, obs, zzz)
        global f
        # p(z)
        x̂ = f(zzz).data
        ŷ = obs_mask .* x̂
        obs = obs_mask .* obs
        # p(y | z)
        log_p_y_z = sum(Distributions.logpdf.(Distributions.Bernoulli.(ŷ), obs))
        return log_p_y_z
    end
    return [single_logpdf_joint(obs_mask, obs, zz[i, :]) for i in 1:size(zz, 1)]
end

# Image saving

using Images

function save_images(zs, fname, obs, obs_mask)
    cd(@__DIR__)
    to_img(x) = Gray.(reshape(x, 28, 28))
    image = hcat(to_img.([rand.(Bernoulli.(f(zs[i, :]).*(1. .- obs_mask).+obs.*obs_mask))
                          for i = 1:size(zs, 1)])...)
    save(fname, image)
end
