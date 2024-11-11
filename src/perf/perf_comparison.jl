using JuliaBUGS
using JuliaBUGS.BUGSPrimitives
using BangBang
using Bijectors
using ADTypes
using AbstractMCMC
using MCMCChains
using LogDensityProblems
using BenchmarkTools
using Test
##

using LogDensityProblemsAD
using ReverseDiff
using Mooncake
using Enzyme
##

# compile the model
(; model_def, data, inits) = JuliaBUGS.BUGSExamples.rats
model = compile(model_def, data, inits)

# rats has 65 parameters
params = rand(65)
params = JuliaBUGS.getparams(model)

##
# Define log density types for different implementations
abstract type RatsLogDensity end

struct RatsLogDensityUnrolled <: RatsLogDensity
    evaluation_env
end

struct RatsLogDensityWithForLoops <: RatsLogDensity
    evaluation_env
end

struct RatsLogDensityMatchStanForMooncake <: RatsLogDensity
    evaluation_env
end

# Include implementations
Revise.includet("rats_logdensity_functions.jl")

# Implement LogDensityProblems interface
for T in (
    :RatsLogDensityUnrolled,
    :RatsLogDensityWithForLoops,
    :RatsLogDensityMatchStanForMooncake,
)
    @eval begin
        LogDensityProblems.dimension(::$T) = 65
    end
end

function LogDensityProblems.logdensity(rld::RatsLogDensityUnrolled, params)
    return rats_logdensity_unrolled(rld.evaluation_env, params)
end

function LogDensityProblems.logdensity(rld::RatsLogDensityWithForLoops, params)
    return rats_logdensity_with_for_loops(rld.evaluation_env, params)
end

function LogDensityProblems.logdensity(rld::RatsLogDensityMatchStanForMooncake, params)
    return rats_logdensity_match_stan_for_mooncake(rld.evaluation_env, params)
end

# Create instances
rld_unrolled = RatsLogDensityUnrolled(deepcopy(model.evaluation_env))
rld_with_for_loops = RatsLogDensityWithForLoops(deepcopy(model.evaluation_env))
rld_match_stan_for_mooncake = RatsLogDensityMatchStanForMooncake(
    deepcopy(model.evaluation_env)
)

# Verify implementations give same results
LogDensityProblems.logdensity(model, params)
LogDensityProblems.logdensity(rld_unrolled, params)
LogDensityProblems.logdensity(rld_with_for_loops, params)
LogDensityProblems.logdensity(rld_match_stan_for_mooncake, params)

##

# Benchmark primals
@benchmark LogDensityProblems.logdensity($model, $params)
@benchmark LogDensityProblems.logdensity($rld_unrolled, $params)
@benchmark LogDensityProblems.logdensity($rld_with_for_loops, $params)

# Setup ReverseDiff AD
const reversediff_config = (; compile=Val(true))
const reversediff_ad_logdensity_bugsmodel = LogDensityProblemsAD.ADgradient(
    :ReverseDiff, model; reversediff_config...
)
const reversediff_ad_logdensity_unrolled = LogDensityProblemsAD.ADgradient(
    :ReverseDiff, rld_unrolled; reversediff_config...
)
const reversediff_ad_logdensity_with_for_loops = LogDensityProblemsAD.ADgradient(
    :ReverseDiff, rld_with_for_loops; reversediff_config...
)

@be LogDensityProblems.logdensity_and_gradient($reversediff_ad_logdensity_bugsmodel, $θ)
@be LogDensityProblems.logdensity_and_gradient($reversediff_ad_logdensity_unrolled, $θ)
@be LogDensityProblems.logdensity_and_gradient(
    $reversediff_ad_logdensity_with_for_loops, $θ
)

# Enzyme
enzyme_ad_logdensity_bugsmodel = LogDensityProblemsAD.ADgradient(AutoEnzyme(), model)
enzyme_ad_logdensity_unrolled = LogDensityProblemsAD.ADgradient(AutoEnzyme(), rld_unrolled)
enzyme_ad_logdensity_with_for_loops = LogDensityProblemsAD.ADgradient(
    AutoEnzyme(), rld_with_for_loops
)
enzyme_ad_logdensity_match_stan_for_mooncake = LogDensityProblemsAD.ADgradient(
    AutoEnzyme(), rld_match_stan_for_mooncake
)

@be LogDensityProblems.logdensity_and_gradient($enzyme_ad_logdensity_bugsmodel, $θ)
@be LogDensityProblems.logdensity_and_gradient($enzyme_ad_logdensity_unrolled, $θ)
@be LogDensityProblems.logdensity_and_gradient($enzyme_ad_logdensity_with_for_loops, $θ)
@be LogDensityProblems.logdensity_and_gradient(
    $enzyme_ad_logdensity_match_stan_for_mooncake, $θ
)
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
mooncake_ad_logdensity_bugsmodel = LogDensityProblemsAD.ADgradient(
    AutoMooncake(; config=Mooncake.Config()), model
)
mooncake_ad_logdensity_unrolled = LogDensityProblemsAD.ADgradient(
    AutoMooncake(; config=Mooncake.Config()), rld_unrolled
)
mooncake_ad_logdensity_with_for_loops = LogDensityProblemsAD.ADgradient(
    AutoMooncake(; config=Mooncake.Config()), rld_with_for_loops
)
mooncake_ad_logdensity_match_stan_for_mooncake = LogDensityProblemsAD.ADgradient(
    AutoMooncake(; config=Mooncake.Config()), rld_match_stan_for_mooncake
)

@be LogDensityProblems.logdensity_and_gradient($mooncake_ad_logdensity_bugsmodel, $θ)
@be LogDensityProblems.logdensity_and_gradient($mooncake_ad_logdensity_unrolled, $θ) # stack overflow
@be LogDensityProblems.logdensity_and_gradient($mooncake_ad_logdensity_with_for_loops, $θ)
@be LogDensityProblems.logdensity_and_gradient(
    $mooncake_ad_logdensity_match_stan_for_mooncake, $θ
)

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
