struct SlicedNormal <: SlicedDistribution
    d::Integer
    λ::AbstractVector
    μ::AbstractVector
    M::AbstractMatrix
    Δ::IntervalBox
    c::Float64
end

function SlicedNormal(δ::AbstractMatrix, d::Integer, b::Integer=10000)
    lb = vec(minimum(δ; dims=1))
    ub = vec(maximum(δ; dims=1))

    Δ = IntervalBox(interval.(lb, ub)...)

    s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

    basis = monomials(["δ$i" for i in 1:size(δ, 2)], 2d, GradedLexicographicOrder())

    zδ = transpose(basis(transpose(δ)))
    zΔ = transpose(basis(s))

    μ, P = mean_and_covariance(zδ)

    M = cholesky(P).U

    zsosδ = transpose(mapreduce(z -> Zsos(z, μ, M), hcat, eachrow(zδ)))
    zsosΔ = transpose(mapreduce(z -> Zsos(z, μ, M), hcat, eachrow(zΔ)))

    n = size(δ, 1)
    nz = size(zδ, 2)

    function f(λ)
        return n * log(prod(ub - lb) / b * sum(exp.(zsosΔ * λ / -2))) + sum(zsosδ * λ) / 2
    end

    function ∇f!(g, λ)
        exp_Δ = exp.(zsosΔ * λ / -2)
        sum_exp_Δ = sum(exp_Δ)
        for i in eachindex(g)
            g[i] = @views n * sum(exp_Δ .* -0.5zsosΔ[:, i]) / sum_exp_Δ +
                sum(zsosδ[:, i]) / 2
        end
        return nothing
    end

    function ∇²f!(H, λ)
        exp_Δ = exp.(zsosΔ * λ / -2)
        sum_exp_Δ = sum(exp_Δ)
        sum_exp_Δ² = sum_exp_Δ^2

        for (i, Δ_i) in enumerate(eachcol(zsosΔ))
            exp_Δ_i = exp_Δ .* -0.5Δ_i
            sum_exp_Δ_i = sum(exp_Δ_i)

            for (j, Δ_j) in enumerate(eachcol(zsosΔ))
                H[i, j] =
                    n * (
                        sum(exp_Δ_i .* -0.5Δ_j) * sum_exp_Δ -
                        sum_exp_Δ_i * sum(exp_Δ .* -0.5Δ_j)
                    ) / sum_exp_Δ²
            end
        end
        return nothing
    end

    result = optimize(f, ∇f!, ∇²f!, zeros(nz), fill(Inf, nz), ones(nz), IPNewton())

    cΔ = prod(ub - lb) / b * sum(exp.(zsosΔ * result.minimizer / -2))

    sn = SlicedNormal(d, result.minimizer, μ, M, Δ, cΔ)
    return sn, -result.minimum
end

function pdf(sn::SlicedNormal, δ::AbstractVector)
    if δ ∈ sn.Δ
        z = Zsos(Z(δ, 2sn.d), sn)
        return exp(-dot(z, sn.λ) / 2) / sn.c
    else
        return 0
    end
end

function Zsos(z::AbstractVector, μ::AbstractVector, M::AbstractMatrix)
    return (M * (z - μ)) .^ 2
end

Zsos(z::AbstractVector, sn::SlicedNormal) = Zsos(z, sn.μ, sn.M)

function mean_and_covariance(z::AbstractMatrix)
    μ = vec(mean(z; dims=1))
    P = inv(cov(LinearShrinkage(ConstantCorrelation()), z))

    return μ, P
end
