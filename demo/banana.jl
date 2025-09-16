using DelimitedFiles
using Plots
using SlicedDistributions
using ECOS
using QuasiMonteCarlo
using Monomials
using LogExpFunctions
using JuMP
δ = readdlm("demo/data/banana.csv", ',')

# Fit Sliced Normal Distribution
d = 3
b = 10_000

lb = [-4, -2]
ub = [4, 70]

s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

t = monomials(["δ$i" for i in 1:size(δ, 1)], d, GradedLexicographicOrder())

zδ = permutedims(t(δ))
zΔ = permutedims(t(s))

n = size(δ, 2)
nz = size(zδ, 2)

# function f(λ)
#     return n * LogExpFunctions.logsumexp(zΔ * -λ) + sum(zδ * λ)
# end

model = Model(ECOS.Optimizer)
set_attribute(model, "maxit", 10^6)
@variable(model, λ[1:nz])
@variable(model, t)
@variable(model, s[1:b] >= 0)

# Exponential cone constraints for logsumexp
for i in 1:b
    # zΔ[i, :] is the i-th row, λ is a vector
    @constraint(model, [dot(zΔ[i, :], -λ) - t, 1.0, s[i]] in MOI.ExponentialCone())
end
@constraint(model, sum(s) == 1)

# Objective
@objective(model, Min, n * t + sum(zδ * λ))

optimize!(model)

cΔ = exp(log(prod(ub - lb)) - log(b) + LogExpFunctions.logsumexp(zΔ * -value.(λ)))

t = monomials(["δ$i" for i in 1:size(δ, 1)], d, GradedLexicographicOrder())

sn = SlicedExponential(d, t, value.(λ), lb, ub, cΔ)

lh = objective_value(model)

p = scatter(
    δ[1, :], δ[2, :]; xlims=[-4, 4], ylims=[-2, 70], xlab="δ1", ylab="δ2", label="data"
)
# scatter!(p, samples[1, :], samples[2, :]; label="samples")

display(p)

# Plot density
xs = range(-4, 4; length=1000)
ys = range(-2, 70; length=1000)

contour!(xs, ys, (x, y) -> pdf(sn, [x, y]))
