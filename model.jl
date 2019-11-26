using Flux
using BSON: @load

include("nn.jl")

# Latent dimensionality, # hidden units.
Dz, Dh = 5, 500

# Components of recognition model / "encoder" MLP.
A, μ, logσ = Dense(28^2, Dh, tanh), Dense(Dh, Dz), Dense(Dh, Dz)
g(X) = (h = A(X); (μ(h), logσ(h)))
z(μ, logσ) = μ + exp(logσ) * randn(Float32)

# Generative model / "decoder" MLP.
f = Chain(Dense(Dz, Dh, tanh), Dense(Dh, 28^2, σ))
@load "ckpt/generator.bson" f

################################# Sample Output ##############################

using Images

img(x) = Gray.(reshape(x, 28, 28))

cd(@__DIR__)
sample = hcat(img.([modelsample() for i = 1:10])...)
save("sample.png", sample)
