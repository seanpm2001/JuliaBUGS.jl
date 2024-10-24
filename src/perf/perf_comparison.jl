using JuliaBUGS
using JuliaBUGS.BUGSPrimitives
using JuliaBUGS.BangBang
using JuliaBUGS.Bijectors
using ADTypes
using AbstractMCMC
using MCMCChains
using LogDensityProblems
using LogDensityProblemsAD
using ReverseDiff
using Mooncake
using Enzyme
using Chairmarks
using Test

# compile the model
(; model_def, data, inits) = JuliaBUGS.BUGSExamples.rats
model = compile(model_def, data, inits)

# rats has 65 parameters
θ = rand(65)


# `BUGSModel` is already compatible with `LogDensityProblems`
struct RatsLogDensityUnrolled 
    evaluation_env
end
struct RatsLogDensityWithForLoops
    evaluation_env
end
struct RatsLogDensityMatchStanForMooncake
    evaluation_env
end

include("rats_logdensity_functions.jl")
LogDensityProblems.logdensity(rld::RatsLogDensityUnrolled, θ) = rats_logdensity_unrolled(rld.evaluation_env, θ)
LogDensityProblems.dimension(::RatsLogDensityUnrolled) = 65

LogDensityProblems.logdensity(rld::RatsLogDensityWithForLoops, θ) = rats_logdensity_with_for_loops(rld.evaluation_env, θ)
LogDensityProblems.dimension(::RatsLogDensityWithForLoops) = 65

LogDensityProblems.logdensity(rld::RatsLogDensityMatchStanForMooncake, θ) = rats_logdensity_match_stan_for_mooncake(rld.evaluation_env, θ)
LogDensityProblems.dimension(::RatsLogDensityMatchStanForMooncake) = 65

rld_unrolled = RatsLogDensityUnrolled(deepcopy(model.evaluation_env))
rld_with_for_loops = RatsLogDensityWithForLoops(deepcopy(model.evaluation_env))
rld_match_stan_for_mooncake = RatsLogDensityMatchStanForMooncake(deepcopy(model.evaluation_env))

LogDensityProblems.logdensity(model, θ)
LogDensityProblems.logdensity(rld_unrolled, θ)
LogDensityProblems.logdensity(rld_with_for_loops, θ)
LogDensityProblems.logdensity(rld_match_stan_for_mooncake, θ)

# Primals
@be LogDensityProblems.logdensity($model, $θ)
@be LogDensityProblems.logdensity($rld_unrolled, $θ)
@be LogDensityProblems.logdensity($rld_with_for_loops, $θ)

# ReverseDiff
reversediff_ad_logdensity_bugsmodel = LogDensityProblemsAD.ADgradient(:ReverseDiff, model; compile=Val(true))
reversediff_ad_logdensity_unrolled = LogDensityProblemsAD.ADgradient(:ReverseDiff, rld_unrolled; compile=Val(true))
reversediff_ad_logdensity_with_for_loops = LogDensityProblemsAD.ADgradient(:ReverseDiff, rld_with_for_loops; compile=Val(true))

@be LogDensityProblems.logdensity_and_gradient($reversediff_ad_logdensity_bugsmodel, $θ)
@be LogDensityProblems.logdensity_and_gradient($reversediff_ad_logdensity_unrolled, $θ)
@be LogDensityProblems.logdensity_and_gradient($reversediff_ad_logdensity_with_for_loops, $θ)

# Enzyme
enzyme_ad_logdensity_bugsmodel = LogDensityProblemsAD.ADgradient(AutoEnzyme(), model)
enzyme_ad_logdensity_unrolled = LogDensityProblemsAD.ADgradient(AutoEnzyme(), rld_unrolled)
enzyme_ad_logdensity_with_for_loops = LogDensityProblemsAD.ADgradient(AutoEnzyme(), rld_with_for_loops)
enzyme_ad_logdensity_match_stan_for_mooncake = LogDensityProblemsAD.ADgradient(AutoEnzyme(), rld_match_stan_for_mooncake)

@be LogDensityProblems.logdensity_and_gradient($enzyme_ad_logdensity_bugsmodel, $θ)
@be LogDensityProblems.logdensity_and_gradient($enzyme_ad_logdensity_unrolled, $θ)
@be LogDensityProblems.logdensity_and_gradient($enzyme_ad_logdensity_with_for_loops, $θ)
@be LogDensityProblems.logdensity_and_gradient($enzyme_ad_logdensity_match_stan_for_mooncake, $θ)
# did not run

# Benchmark: 2729 samples with 10 evaluations
#  min    2.838 μs (7.40 allocs: 3.794 KiB)
#  median 2.983 μs (7.40 allocs: 3.794 KiB)
#  mean   3.474 μs (7.40 allocs: 3.794 KiB, 0.11% gc time)
#  max    679.312 μs (7.40 allocs: 3.794 KiB, 98.80% gc time)

# Benchmark: 2940 samples with 8 evaluations
#  min    3.349 μs (7.50 allocs: 928 bytes)
#  median 3.469 μs (7.50 allocs: 928 bytes)
#  mean   3.909 μs (7.50 allocs: 928 bytes, 0.03% gc time)
#  max    1.101 ms (7.50 allocs: 928 bytes, 99.24% gc time)

# Mooncake
mooncake_ad_logdensity_bugsmodel = LogDensityProblemsAD.ADgradient(AutoMooncake(; config=Mooncake.Config()), model)
mooncake_ad_logdensity_unrolled = LogDensityProblemsAD.ADgradient(AutoMooncake(; config=Mooncake.Config()), rld_unrolled)
mooncake_ad_logdensity_with_for_loops = LogDensityProblemsAD.ADgradient(AutoMooncake(; config=Mooncake.Config()), rld_with_for_loops)
mooncake_ad_logdensity_match_stan_for_mooncake = LogDensityProblemsAD.ADgradient(AutoMooncake(; config=Mooncake.Config()), rld_match_stan_for_mooncake)

@be LogDensityProblems.logdensity_and_gradient($mooncake_ad_logdensity_bugsmodel, $θ)
@be LogDensityProblems.logdensity_and_gradient($mooncake_ad_logdensity_unrolled, $θ) # stack overflow
@be LogDensityProblems.logdensity_and_gradient($mooncake_ad_logdensity_with_for_loops, $θ)
@be LogDensityProblems.logdensity_and_gradient($mooncake_ad_logdensity_match_stan_for_mooncake, $θ)

# # with release - v0.4.18

# Benchmark: 4 samples with 1 evaluation
#         29.215 ms (265791 allocs: 12.083 MiB)
#         30.004 ms (265791 allocs: 12.083 MiB)
#         30.697 ms (265791 allocs: 12.083 MiB)
#         52.408 ms (265791 allocs: 12.083 MiB, 43.87% gc time)

# stack overflow

# Benchmark: 3352 samples with 3 evaluations
#  min    8.708 μs (52.33 allocs: 2.409 KiB)
#  median 9.139 μs (52.33 allocs: 2.409 KiB)
#  mean   9.291 μs (52.33 allocs: 2.409 KiB)
#  max    22.250 μs (52.33 allocs: 2.409 KiB)

# with faster-stack
# Benchmark: 4956 samples with 2 evaluations
#  min    8.521 μs (53 allocs: 2.430 KiB)
#  median 9.021 μs (53 allocs: 2.430 KiB)
#  mean   9.433 μs (53 allocs: 2.430 KiB)
#  max    35.688 μs (53 allocs: 2.430 KiB)

# with some vectorization
# Benchmark: 3264 samples with 3 evaluations
#  min    7.805 μs (66.33 allocs: 4.284 KiB)
#  median 8.167 μs (66.33 allocs: 4.284 KiB)
#  mean   9.781 μs (66.33 allocs: 4.284 KiB, 0.03% gc time)
#  max    4.313 ms (66.33 allocs: 4.284 KiB, 99.18% gc time)


using StanLogDensityProblems, BridgeStan

stan_model = BridgeStan.StanModel(
    "/Users/xiandasun/JuliaBUGS.jl.worktrees/master/benchmark/stan/bugs_examples/vol1/rats/rats.stan",
    "/Users/xiandasun/JuliaBUGS.jl.worktrees/master/benchmark/stan/bugs_examples/vol1/rats/rats.data.json",
)
stan_problem = StanLogDensityProblems.StanProblem(stan_model)

@be LogDensityProblems.logdensity(stan_problem, θ)

# Benchmark: 3551 samples with 5 evaluations
#  min    4.933 μs (2.80 allocs: 54.400 bytes)
#  median 5.025 μs (2.80 allocs: 54.400 bytes)
#  mean   5.156 μs (2.80 allocs: 54.400 bytes)
#  max    8.517 μs (2.80 allocs: 54.400 bytes)

