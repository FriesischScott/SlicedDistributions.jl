struct SlicedNormal <: SlicedDistribution
    d::Integer
    t::Vector{Monomial}
    λ::AbstractVector
    μ::AbstractVector
    M::AbstractMatrix
    lb::AbstractVector{<:Real}
    ub::AbstractVector{<:Real}
    c::Float64
end

function SlicedNormal(
    δ::AbstractMatrix,
    d::Integer,
    b::Integer=10000,
    lb::AbstractVector{<:Real}=vec(minimum(δ; dims=2)),
    ub::AbstractVector{<:Real}=vec(maximum(δ; dims=2)),
)
    s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

    t = monomials(["δ$i" for i in 1:size(δ, 1)], 2d, GradedLexicographicOrder())

    zδ = t(δ)
    zΔ = t(s)

    μ, P = mean_and_covariance(zδ)

    M = cholesky(P).U

    zsosδ = permutedims(Zsos(zδ, μ, M))
    zsosΔ = permutedims(Zsos(zΔ, μ, M))

    n = size(δ, 2)
    nz = size(zδ, 1)

    f = get_likelihood(zsosδ, zsosΔ, n, prod(ub - lb), b)

    ∇f! = get_gradient(zsosδ, zsosΔ, n)

    ∇²f! = get_hessian(zsosΔ, n)

    result = optimize(f, ∇f!, ∇²f!, zeros(nz), fill(Inf, nz), ones(nz), IPNewton())

    cΔ = prod(ub - lb) / b * sum(exp.(zsosΔ * result.minimizer / -2))

    sn = SlicedNormal(d, t, result.minimizer, μ, M, lb, ub, cΔ)
    return sn, -result.minimum
end

function Zsos(z::AbstractVector, μ::AbstractVector, M::AbstractMatrix)
    return (M * (z - μ)) .^ 2
end

function Zsos(z::AbstractMatrix, μ::AbstractVector, M::AbstractMatrix)
    return (M * (z .- μ)) .^ 2
end

Zsos(z::AbstractVector, sn::SlicedNormal) = Zsos(z, sn.μ, sn.M)

function mean_and_covariance(z::AbstractMatrix)
    μ = vec(mean(z; dims=2))
    P = inv(cov(LinearShrinkage(ConstantCorrelation()), z; dims=2))

    return μ, P
end

function _logpdf(sn::SlicedNormal, δ::AbstractArray)
    if all(sn.lb .<= δ .<= sn.ub)
        z = Zsos(sn.t(δ), sn)
        return log(exp(-dot(z, sn.λ) / 2) / sn.c)
    else
        return log(0)
    end
end

function Base.length(sn::SlicedNormal)
    return length(sn.t[1].x)
end

function Base.eltype(sn::SlicedNormal)
    return eltype(sn.λ)
end
