using IntervalArithmetic
using Plots
using SlicedNormals

n = 500

δ1 = rand(n)
δ2 = [0.5 <= rand() ? δ1[i] : 1 - δ1[i] for i = 1:n]
δ = [δ1 δ2]

d = 3
μ, P = SlicedNormals.fit(δ, d)
Δ = IntervalBox(0..1, 0..1)

sn = SlicedNormal(d, μ, P, Δ)

resolution = 100
x = range(0, 1, length=resolution)
y = range(0, 1, length=resolution)
z = zeros(resolution, resolution)

z = [SlicedNormals.pdf(sn, [x[i], y[j]]) for i = 1:resolution, j = 1:resolution]

samples = rand(sn, 1000)

p = plot()
contourf!(p, x, y, z)
scatter!(p, δ1, δ2, legend=:none)
scatter!(p, samples[:, 1], samples[:, 2], legend=:none)
