module SlicedDistributions

using CovarianceEstimation
using Distributions
using IntervalArithmetic
using LinearAlgebra
using Monomials
using TransitionalMCMC
using QuasiMonteCarlo
using Optim

import Base: rand

export SlicedNormal, SlicedExponential, rand, pdf

abstract type SlicedDistribution end

function Distributions.pdf(sn::SlicedDistribution, δ::AbstractMatrix)
    n, m = size(δ)
    if n == 1 || m == 1
        return pdf(sn, vec(δ))
    end
    if n < m
        return [pdf(sn, c) for c in eachcol(δ)]
    end
    return return [pdf(sn, c) for c in eachrow(δ)]
end

function rand(sd::SlicedDistribution, n::Integer)
    lb, ub = getfield.(sd.Δ, :lo), getfield.(sd.Δ, :hi)

    prior = Uniform.(lb, ub)

    logprior(x) = sum(logpdf.(prior, x))
    sampler(n) = mapreduce(u -> rand(u, n), hcat, prior)
    loglikelihood(x) = log(SlicedDistributions.pdf(sd, x))

    samples, _ = tmcmc(loglikelihood, logprior, sampler, n)

    return samples
end

include("exponentials/poly.jl")
include("normals/sum-of-squares.jl")

end
