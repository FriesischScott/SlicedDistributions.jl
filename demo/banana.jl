using DelimitedFiles
using Plots
using SlicedDistributions

δ = readdlm("demo/data/banana_data.csv", ',')

# define custom support

lb = [-3.5, 0]
ub = [3.5, 60]

d = 4
b = 10000

sn, lh = SlicedNormal(δ, d, b, lb, ub)

println("Likelihood: $lh")

# Plot density
xs = range(lb[1], ub[1]; length=1000)
ys = range(lb[2], ub[2]; length=1000)

p = contour(
    xs,
    ys,
    (x, y) -> pdf(sn, [x, y]);
    xlims=[lb[1], ub[1]],
    ylims=[lb[2], ub[2]],
)

scatter!(p,
    δ[1, :],
    δ[2, :];
    xlims=[lb[1], ub[1]],
    ylims=[lb[2], ub[2]],
    xlab="δ1",
    ylab="δ2",
    label="data",
    markersize=2,
    markeralpha=0.3,
)

samples = rand(sn, 2000)

scatter!(p,
    samples[1, :],
    samples[2, :];
    xlims=[lb[1], ub[1]],
    ylims=[lb[2], ub[2]],
    xlab="δ1",
    ylab="δ2",
    label="samples",
    markersize=2,
    markeralpha=0.3,
)



δ = readdlm("demo/data/swirl_data.csv", ',')

# define custom support

lb = [-18, -18]
ub = [18, 18]

d = 8
b = 10000

sn, lh = SlicedNormal(δ, d, b, lb, ub)

println("Likelihood: $lh")

# Plot density
xs = range(lb[1], ub[1]; length=1000)
ys = range(lb[2], ub[2]; length=1000)

p2 = contour(
    xs,
    ys,
    (x, y) -> pdf(sn, [x, y]);
    xlims=[lb[1], ub[1]],
    ylims=[lb[2], ub[2]],
)


scatter!(p2,
    δ[1, :],
    δ[2, :];
    xlims=[lb[1], ub[1]],
    ylims=[lb[2], ub[2]],
    xlab="δ1",
    ylab="δ2",
    label="data",
    markersize=2,
    markeralpha=0.3,
)

samples = rand(sn, 2000)

scatter!(p2,
    samples[1, :],
    samples[2, :];
    xlims=[lb[1], ub[1]],
    ylims=[lb[2], ub[2]],
    xlab="δ1",
    ylab="δ2",
    label="samples",
    markersize=2,
    markeralpha=0.3,
)
