open(
    "/Users/xiandasun/JuliaBUGS.jl.worktrees/master/src/perf/reversediff_tape_bugsmodel.txt",
    "w",
) do io
    println(io, reversediff_ad_logdensity_bugsmodel.compiledtape.tape.tape)
end

open(
    "/Users/xiandasun/JuliaBUGS.jl.worktrees/master/src/perf/reversediff_tape_with_for_loops.txt",
    "w",
) do io
    println(io, reversediff_ad_logdensity_with_for_loops.compiledtape.tape.tape)
end
