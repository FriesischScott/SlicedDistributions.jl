module SlicedNormals

using CovarianceEstimation
using Distributions
using DynamicPolynomials
using IntervalArithmetic
using JuMP
using LinearAlgebra
using MinimumVolumeEllipsoids
using NLopt
using TransitionalMCMC
using QuasiMonteCarlo

import Base: rand

export SlicedNormal, rand, pdf

struct SlicedNormal
    d::Integer
    λ::AbstractVector
    μ::AbstractVector
    M::AbstractMatrix
    Δ::IntervalBox
    c::Float64
end

function SlicedNormal(δ::AbstractMatrix, d::Integer, b::Integer=10000)
    ϵ = minimum_volume_ellipsoid(δ')
    s = rand(ϵ, b)

    zδ = mapreduce(r -> transpose(Z(r, 2d)), vcat, eachrow(δ))
    zΔ = mapreduce(r -> transpose(Z(r, 2d)), vcat, eachcol(s))

    μ, P = mean_and_covariance(zδ)

    M = cholesky(P).U

    zsosδ = transpose(mapreduce(z -> Zsos(z, μ, M), hcat, eachrow(zδ)))
    zsosΔ = transpose(mapreduce(z -> Zsos(z, μ, M), hcat, eachrow(zΔ)))

    n = size(δ, 1)
    nz = size(zδ, 2)

    D = λ -> sum([ϕE(x, λ) for x in eachrow(zsosδ)]) / 2
    cΔ = λ -> volume(ϵ) / b * sum([exp.(ϕE(δ, λ) / -2) for δ in eachrow(zsosΔ)])

    function f(λ...)
        return n * log(cΔ(λ)) + D(λ)
    end

    nz = size(zδ, 2)

    model = Model(NLopt.Optimizer)
    set_optimizer_attribute(model, "algorithm", NLopt.LD_SLSQP)

    @variable(model, λ[1:nz] >= 0.0)

    register(model, :f, nz, f; autodiff=true)
    @NLobjective(model, Min, f(λ...))

    JuMP.optimize!(model)

    lb = vec(minimum(δ; dims=1))
    ub = vec(maximum(δ; dims=1))

    Δ = IntervalBox(interval.(lb, ub)...)

    cΔ = volume(ϵ) / b * sum([exp(-ϕE(δ, value.(λ)) / 2) for δ in eachrow(zsosΔ)])
    return SlicedNormal(d, value.(λ), μ, M, Δ, cΔ), objective_value(model)
end

function pdf(sn::SlicedNormal, δ::AbstractVector)
    if δ ∈ sn.Δ
        z = Zsos(Z(δ, sn.d * 2), sn)
        return exp(-ϕE(z, sn.λ) / 2) / sn.c
    else
        return 0
    end
end

function Distributions.pdf(sn::SlicedNormal, δ::AbstractMatrix)
    n, m = size(δ)
    if n == 1 || m == 1
        return pdf(sn, vec(δ))
    end
    if n < m
        return [pdf(sn, c) for c in eachcol(δ)]
    end
    return return [pdf(sn, c) for c in eachrow(δ)]
end

function Z(δ::AbstractVector, d::Integer)
    x = @polyvar x[1:length(δ)]
    z = mapreduce(p -> monomials(x..., p), vcat, 1:d)

    # double reverse to achieve graded lexographic order
    return map(p -> p(δ), z)
end

function Zsos(z::AbstractVector, μ::AbstractVector, M::AbstractMatrix)
    return (M * (z - μ)) .^ 2
end

Zsos(z::AbstractVector, sn::SlicedNormal) = Zsos(z, sn.μ, sn.M)

function c(
    μ::AbstractVector, P::AbstractMatrix, d::Integer, x::AbstractMatrix, b::Integer=10000
)
    ϵ = minimum_volume_ellipsoid(x')

    V = volume(ϵ)
    u = rand(ϵ, b)

    return V / b * sum([exp(-_ϕ(δ, μ, P, d)) for δ in eachcol(u)])
end

function bounds(Δ::IntervalBox)
    lb = getfield.(Δ, :lo)
    ub = getfield.(Δ, :hi)

    return lb, ub
end

function mean_and_covariance(z::AbstractMatrix)
    μ = vec(mean(z; dims=1))
    P = inv(cov(LinearShrinkage(ConstantCorrelation()), z))

    return μ, P
end

function rand(sn::SlicedNormal, n::Integer)
    lb, ub = bounds(sn.Δ)

    prior = Uniform.(lb, ub)

    logprior(x) = sum(logpdf.(prior, x))
    sampler(n) = mapreduce(u -> rand(u, n), hcat, prior)
    loglikelihood(x) = log(SlicedNormals.pdf(sn, x))

    samples, _ = tmcmc(loglikelihood, logprior, sampler, n)

    return samples
end

function ϕE(z, λ)
    return dot(λ, z)
end

end # module
