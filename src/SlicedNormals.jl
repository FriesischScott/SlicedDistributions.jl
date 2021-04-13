module SlicedNormals

using CovarianceEstimation
using Distributions
using DynamicPolynomials
using HCubature
using IntervalArithmetic
using LinearAlgebra
using Memoize
using PDMats
using TransitionalMCMC

import Base:rand

export SlicedNormal, Z, rand, pdf, fit

struct SlicedNormal
    d::Integer
    μ::AbstractVector
    P::AbstractPDMat
    Δ::IntervalBox
end

function pdf(sn::SlicedNormal, δ::AbstractVector)
    if δ ∈ sn.Δ
        mvn = MvNormal(sn.μ, inv(sn.P))
        z = Z(δ, sn.d)

        return (γ(sn) / c(sn)) * Distributions.pdf(mvn, z)
    else
        return 0
    end
end

function Z(δ::AbstractVector, d::Integer)
    x = @polyvar x[1:length(δ)]
    z = monomials(x..., 1:d)

    # double reverse to achieve graded lexographic order
    map(p -> p(reverse(δ)), z) |> reverse
end

@memoize function c(sn::SlicedNormal)
    mvn = MvNormal(sn.μ, inv(sn.P))

    lb, ub = support(sn)

    normalization, _ = hcubature(x -> γ(sn) * Distributions.pdf(mvn, Z(x, sn.d)), lb, ub)

    return normalization
end

@memoize function γ(sn::SlicedNormal)
    (2 * π)^(length(sn.μ) / 2) * sqrt(det(inv(sn.P)))
end


@memoize function support(sn::SlicedNormal)
    lb = [map(x -> x.lo, sn.Δ.v.data)...]
    ub = [map(x -> x.hi, sn.Δ.v.data)...]

    return lb, ub
end

function fit(x::AbstractMatrix, d::Integer)
    z  = mapreduce(r -> Z(r, d) |> transpose, vcat, eachrow(x))

    μ = mean(z, dims=1) |> vec
    P = cov(LinearShrinkage(ConstantCorrelation()), z) |> inv |> PDMat

    return μ, P
end

function rand(sn::SlicedNormal, n::Integer)
    lb, ub = support(sn)

    prior = Uniform.(lb, ub)

    logprior(x) = logpdf.(prior, x) |> sum
    sampler(n) = mapreduce(u -> rand(u, n), hcat, prior)
    loglikelihood(x) = SlicedNormals.pdf(sn, x) |> log

    samples, _ = tmcmc(loglikelihood, logprior, sampler, n)

    return samples
end

end # module
