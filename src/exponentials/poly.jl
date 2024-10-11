struct SlicedExponential <: SlicedDistribution
    d::Integer
    t::Vector{Monomial}
    λ::AbstractVector
    Δ::IntervalBox
    c::Float64
end

function SlicedExponential(δ::AbstractMatrix, d::Integer, b::Integer=10000)
    lb = vec(minimum(δ; dims=1))
    ub = vec(maximum(δ; dims=1))

    Δ = IntervalBox(interval.(lb, ub)...)

    s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

    t = monomials(["δ$i" for i in 1:size(δ, 2)], 2d, GradedLexicographicOrder())

    zδ = permutedims(t(transpose(δ)))
    zΔ = permutedims(t(s))

    n = size(δ, 1)
    nz = size(zδ, 2)

    f = get_likelihood(zδ, zΔ, n, prod(ub - lb), b)

    function ∇f!(g, λ)
        exp_Δ = exp.(zΔ * λ / -2)
        sum_exp_Δ = sum(exp_Δ)
        for i in eachindex(g)
            g[i] = @views n * sum(exp_Δ .* -0.5zΔ[:, i]) / sum_exp_Δ + sum(zδ[:, i]) / 2
        end
        return nothing
    end

    function ∇²f!(H, λ)
        exp_Δ = exp.(zΔ * λ / -2)
        sum_exp_Δ = sum(exp_Δ)
        sum_exp_Δ² = sum_exp_Δ^2

        for (i, Δ_i) in enumerate(eachcol(zΔ))
            exp_Δ_i = exp_Δ .* -0.5Δ_i
            sum_exp_Δ_i = sum(exp_Δ_i)

            for (j, Δ_j) in enumerate(eachcol(zΔ))
                H[i, j] =
                    n * (
                        sum(exp_Δ_i .* -0.5Δ_j) * sum_exp_Δ -
                        sum_exp_Δ_i * sum(exp_Δ .* -0.5Δ_j)
                    ) / sum_exp_Δ²
            end
        end
        return nothing
    end

    result = optimize(f, ∇f!, ∇²f!, ones(nz), Newton())

    cΔ = prod(ub - lb) / b * sum(exp.(zΔ * result.minimizer / -2))

    sn = SlicedExponential(d, t, result.minimizer, Δ, cΔ)
    return sn, -result.minimum
end

function pdf(sn::SlicedExponential, δ::AbstractVector)
    if δ ∈ sn.Δ
        return exp(-dot(sn.t(δ), sn.λ) / 2) / sn.c
    else
        return 0
    end
end
