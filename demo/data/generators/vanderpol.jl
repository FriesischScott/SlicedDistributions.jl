using DelimitedFiles
using DifferentialEquations
using Random

Random.seed!(8562)

n = 1000

function vanderpol(du, u, p, t)
    du[1] = u[2]
    du[2] = p[1] * (1 - u[1] .^ 2) .* u[2] - u[1]

    return du
end

tspan = (0, 7.5)

δ = map(1:floor(n / 75)) do _
    μ = 1.1 + randn() / 10
    u₀ = [0.15, 0.15] + max.(min.(randn(2) / 10, 0.1), -0.1)

    prob = ODEProblem(vanderpol, u₀, tspan, [μ])

    sol = solve(prob; alg_hints=[:stiff], reltol=1e-8, abstol=1e-8, saveat=0.1)

    return permutedims(mapreduce(x -> permutedims(x), vcat, sol.u))
end

δ = hcat(δ...)

writedlm("demo/data/vanderpol.csv", δ, ',')
