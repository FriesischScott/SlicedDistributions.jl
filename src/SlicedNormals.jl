module SlicedNormals

using CovarianceEstimation
using Distributions
using DynamicPolynomials
using IntervalArithmetic
using LinearAlgebra
using MinimumVolumeEllipsoids
using Optim
using TransitionalMCMC

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

    zδ = mapreduce(r -> transpose(Z(r, d)), vcat, eachrow(δ))
    zΔ = mapreduce(r -> transpose(Z(r, d)), vcat, eachcol(s))

    μ, P = mean_and_covariance(zδ)

    M = cholesky(P).U

    zsosδ = transpose(mapreduce(z -> Zsos(z, μ, M), hcat, eachrow(zδ)))
    zsosΔ = transpose(mapreduce(z -> Zsos(z, μ, M), hcat, eachrow(zΔ)))

    m = size(δ, 1)

    function f(λ)
        D = sum([ϕE(x, λ) for x in eachrow(zsosδ)]) / 2
        cΔ = volume(ϵ) / b * sum([exp(-ϕE(δ, λ) / 2) for δ in eachrow(zsosΔ)])

        lh = -m * log(cΔ) - D
        return -lh
    end

    nz = size(zδ, 2)

    opt = optimize(
        f, zeros(nz), Inf .* ones(nz), ones(nz), Fminbox(LBFGS()); autodiff=:forward
    )

    λ = Optim.minimizer(opt)
    lh = -1 * Optim.minimum(opt)

    lb = vec(minimum(δ; dims=1)) .* 1.1
    ub = vec(maximum(δ; dims=1)) .* 1.1

    Δ = IntervalBox(interval.(lb, ub)...)

    cΔ = volume(ϵ) / b * sum([exp(-ϕE(δ, λ) / 2) for δ in eachrow(zsosΔ)])
    return SlicedNormal(d, λ, μ, M, Δ, cΔ), lh
end

function pdf(sn::SlicedNormal, δ::AbstractVector)
    if δ ∈ sn.Δ
        z = Zsos(Z(δ, sn.d), sn)
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
