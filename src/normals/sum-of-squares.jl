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

function SlicedNormal(δ::AbstractMatrix, d::Integer, b::Integer=10000)
    lb = vec(minimum(δ; dims=1))
    ub = vec(maximum(δ; dims=1))

    s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

    t = monomials(["δ$i" for i in 1:size(δ, 2)], 2d, GradedLexicographicOrder())

    zδ = transpose(t(transpose(δ)))
    zΔ = transpose(t(s))

    μ, P = mean_and_covariance(zδ)

    M = cholesky(P).U

    zsosδ = permutedims(Zsos(zδ', μ, M))
    zsosΔ = permutedims(Zsos(zΔ', μ, M))

    n = size(δ, 1)
    nz = size(zδ, 2)

    f = get_likelihood(zsosδ, zsosΔ, n, prod(ub - lb), b)

    ∇f! = get_gradient(zsosδ, zsosΔ, n)

    ∇²f! = get_hessian(zsosΔ, n)

    result = optimize(f, ∇f!, ∇²f!, zeros(nz), fill(Inf, nz), ones(nz), IPNewton())

    cΔ = prod(ub - lb) / b * sum(exp.(zsosΔ * result.minimizer / -2))

    sn = SlicedNormal(d, t, result.minimizer, μ, M, lb, ub, cΔ)
    return sn, -result.minimum
end

function pdf(sn::SlicedNormal, δ::AbstractVector)
    if all(sn.lb .<= δ .<= sn.ub)
        z = Zsos(sn.t(δ), sn)
        return exp(-dot(z, sn.λ) / 2) / sn.c
    else
        return 0
    end
end

function Zsos(z::AbstractVector, μ::AbstractVector, M::AbstractMatrix)
    return (M * (z - μ)) .^ 2
end

function Zsos(z::AbstractMatrix, μ::AbstractVector, M::AbstractMatrix)
    return (M * (z .- μ)) .^ 2
end

Zsos(z::AbstractVector, sn::SlicedNormal) = Zsos(z, sn.μ, sn.M)

function mean_and_covariance(z::AbstractMatrix)
    μ = vec(mean(z; dims=1))
    P = inv(cov(LinearShrinkage(ConstantCorrelation()), z))

    return μ, P
end
