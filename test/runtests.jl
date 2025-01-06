using DelimitedFiles
using HCubature
using Logging
using SlicedDistributions
using Test

Logging.disable_logging(Logging.Info)

circle = readdlm("../demo/data/circle.csv", ',')

d = 3
b = 10000

@testset "SlicedNormal" begin
    sn, _ = SlicedNormal(circle, d, b)

    @test all(insupport.(sn, eachrow(circle)))

    @test hcubature(x -> pdf(sn, x), sn.lb, sn.ub)[1] ≈ 1.0 atol = 1e-3

    samples = rand(sn, 1000)

    @test all(insupport.(sn, eachcol(samples)))
end

@testset "SlicedExponential" begin
    se, _ = SlicedExponential(circle, d, b)

    @test all(insupport.(se, eachrow(circle)))

    @test hcubature(x -> pdf(se, x), se.lb, se.ub)[1] ≈ 1.0 atol = 1e-3

    samples = rand(se, 1000)

    @test all(insupport.(se, eachcol(samples)))
end
