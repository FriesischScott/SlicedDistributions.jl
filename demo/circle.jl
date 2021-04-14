using Distributions
using IntervalArithmetic
using Plots
using SlicedNormals

n = 500

# Step 1
θ = rand(Normal(π / 2, 1.3), n)
r = 3 .+ rand(Uniform(0, 0.2), n) .* θ .- π / 2

δ1 = r .* cos.(θ)
δ2 = r .* sin.(θ)

idx = δ1 .< 0
δ2[idx] = δ2[idx] * -1

δ = [δ1 δ2]

# Step 2
θ = rand(Normal(π / 2, 1.3), n)
r = 3 .+ rand(Uniform(0, 0.2), n) .* θ .- π / 2

δ1 = r .* cos.(θ)
δ2 = r .* sin.(θ)

idx = δ1 .< 0
δ2[idx] = δ2[idx] * -1

δ = vcat(δ, [δ1 δ2] .* -1)

# Fit Sliced Normal Distribution
d = 5
μ, P = SlicedNormals.fit(δ, d)
Δ = IntervalBox(-2.5..2.5, -2.5..2.5)

sn = SlicedNormal(d, μ, P, Δ)

samples = rand(sn, 1000)

# Plot generated data and new samples
p = scatter(δ[:, 1], δ[:, 2], aspect_ratio=:equal, lims=[-2.5, 2.5], xlab="δ1", ylab="δ2", legend=:none)
scatter!(p, samples[:, 1], samples[:, 2])

# Plot sliced normal density
resolution = 100
x = range(-2.5, 2.5, length=resolution)
y = range(-2.5, 2.5, length=resolution)
z = zeros(resolution, resolution)

z = [SlicedNormals.pdf(sn, [x[i], y[j]]) for i = 1:resolution, j = 1:resolution]

contourf(x, y, z, aspect_ratio=:equal, lims=[-2.5, 2.5], c=:tempo)
