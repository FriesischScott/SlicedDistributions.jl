struct SlicedExponential <: SlicedDistribution
    d::Integer
    t::Vector{Monomial}
    λ::AbstractVector
    lb::AbstractVector{<:Real}
    ub::AbstractVector{<:Real}
    c::Float64
end

function SlicedExponential(δ::AbstractMatrix, d::Integer, b::Integer=10000)
    lb = vec(minimum(δ; dims=1))
    ub = vec(maximum(δ; dims=1))

    s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

    t = monomials(["δ$i" for i in 1:size(δ, 2)], 2d, GradedLexicographicOrder())

    zδ = permutedims(t(transpose(δ)))
    zΔ = permutedims(t(s))

    n = size(δ, 1)
    nz = size(zδ, 2)

    f = get_likelihood(zδ, zΔ, n, prod(ub - lb), b)

    ∇f! = get_gradient(zδ, zΔ, n)

    ∇²f! = get_hessian(zΔ, n)

    result = optimize(f, ∇f!, ∇²f!, ones(nz), Newton())

    cΔ = prod(ub - lb) / b * sum(exp.(zΔ * result.minimizer / -2))

    sn = SlicedExponential(d, t, result.minimizer, lb, ub, cΔ)
    return sn, -result.minimum
end

function pdf(sn::SlicedExponential, δ::AbstractVector)
    if all(sn.lb .<= δ .<= sn.ub)
        return exp(-dot(sn.t(δ), sn.λ) / 2) / sn.c
    else
        return 0
    end
end
