using DelimitedFiles
using Plots
using SlicedDistributions

δ = readdlm("demo/data/vanderpol.csv", ',')

# define custom support

lb = [-2.5, -3]
ub = [2.5, 3.5]

d = 3
b = 100_000

sn, lh = SlicedExponential(δ, d, b, lb, ub)

println("Likelihood: $lh")

p = scatter(
    δ[1, :],
    δ[2, :];
    aspect_ratio=:equal,
    xlims=[lb[1], ub[1]],
    ylims=[lb[2], ub[2]],
    xlab="δ1",
    ylab="δ2",
    label="data",
)

# samples = rand(sn, 2000)

# scatter(
#     samples[1, :],
#     samples[2, :];
#     aspect_ratio=:equal,
#     xlims=[lb[1], ub[1]],
#     ylims=[lb[2], ub[2]],
#     xlab="δ1",
#     ylab="δ2",
#     label="samples",
# )

# Plot density
xs = range(lb[1], ub[1]; length=1000)
ys = range(lb[2], ub[2]; length=1000)

contourf(
    xs,
    ys,
    (x, y) -> pdf(sn, [x, y]);
    color=:turbo,
    aspect_ratio=:equal,
    xlims=[lb[1], ub[1]],
    ylims=[lb[2], ub[2]],
)
