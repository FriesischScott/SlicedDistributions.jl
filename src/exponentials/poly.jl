struct SlicedExponential <: SlicedDistribution
    d::Integer
    λ::AbstractVector
    Δ::IntervalBox
    c::Float64
end

function SlicedExponential(δ::AbstractMatrix, d::Integer, b::Integer=10000)
    lb = vec(minimum(δ; dims=1))
    ub = vec(maximum(δ; dims=1))

    Δ = IntervalBox(interval.(lb, ub)...)

    s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

    zδ = mapreduce(r -> transpose(Z(r, 2d)), vcat, eachrow(δ))
    zΔ = mapreduce(r -> transpose(Z(r, 2d)), vcat, eachcol(s))

    n = size(δ, 1)
    nz = size(zδ, 2)

    function f(λ)
        return n * log(prod(ub - lb) / b * sum(exp.(zΔ * λ / -2))) + sum(zδ * λ) / 2
    end

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

    sn = SlicedExponential(d, result.minimizer, Δ, cΔ)
    return sn, -result.minimum
end

function pdf(sn::SlicedExponential, δ::AbstractVector)
    if δ ∈ sn.Δ
        z = Z(δ, 2sn.d)
        return exp(-dot(z, sn.λ) / 2) / sn.c
    else
        return 0
    end
end
