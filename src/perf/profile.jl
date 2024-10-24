using Profile, PProf

Profile.clear()

@profile LogDensityProblems.logdensity(model, θ)

pprof()

PProf.Allocs.allocs()

# Benchmark: 20 samples with 1 evaluation
#  min    4.872 ms (18946 allocs: 725.328 KiB)
#  median 4.913 ms (18946 allocs: 725.328 KiB)
#  mean   4.956 ms (18946 allocs: 725.328 KiB)
#  max    5.196 ms (18946 allocs: 725.328 KiB)

@profview LogDensityProblems.logdensity(model, θ)

# Benchmark: 21 samples with 1 evaluation
#  min    4.513 ms (17548 allocs: 657.688 KiB)
#  median 4.633 ms (17548 allocs: 657.688 KiB)
#  mean   4.624 ms (17548 allocs: 657.688 KiB)
#  max    4.685 ms (17548 allocs: 657.688 KiB)
 