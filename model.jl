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

function sample_z_prior()
    global Dz
    return z.(zeros(Dz), zeros(Dz))
end

function logpdf_joint(obs_mask, obs, z)
    global f
    # p(z)
    log_p_z = sum(Distributions.logpdf.(Distributions.Normal(), z))
    x̂ = f(z)
    ŷ = obs_mask .* x̂
    obs = obs_mask .* obs
    # p(y | z)
    log_p_y_z = sum(Distributions.logpdf.(Distributions.Bernoulli.(ŷ), obs))
    return log_p_z .+ log_p_y_z
end

# ################################# Sample Output ##############################

# using Images

# img(x) = Gray.(reshape(x, 28, 28))

# cd(@__DIR__)
# sample = hcat(img.([modelsample() for i = 1:10])...)
# save("sample.png", sample)
