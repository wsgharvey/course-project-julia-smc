using Flux, Flux.Data.MNIST, Statistics
using Flux: throttle, params
using Juno: @progress
using BSON: @save

include("nn.jl")

# Load data, binarise it, and partition into mini-batches of M.
X = float.(hcat(vec.(MNIST.images())...)) .> 0.5
N, M = size(X, 2), 100
data = [X[:,i] for i in Iterators.partition(1:N,M)]

evalcb = throttle(() -> @show(-L̄(X[:, rand(1:N, M)])), 30)
opt = ADAM()
ps = params(A, μ, logσ, f)

@progress for i = 1:20
  global f
  @info "Epoch $i"
  Flux.train!(loss, ps, zip(data), opt, cb=evalcb)
  @save "ckpt/generator.bson" f
end
