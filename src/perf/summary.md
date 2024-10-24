# About the performance of JuliaBUGS

Results and my conclusions:

1. The current speed of JuliaBUGS (10 times slower than Stan) appears to be a limitation of ReverseDiff.
   1. This was verified by writing two versions of the same log density function:
      - One fully unrolled to mimic what JuliaBUGS currently does
      - The other using for loops
   2. The goal was to see if ReverseDiff performs better on a better written program. However, the results showed that all three versions of the log density function perform almost the same (JuliaBUGS is slightly worse at 210 μs vs 190 μs).
   3. The primal takes about 4 ms (compared to the compiled tape at 200 μs).
      1. The ReverseDiff number is basically the type-stable performance.
      2. Profiling shows that the primal is slow due to type stability issues.

2. In comparison, Enzyme takes about 3-4 μs (slightly faster than Stan), and Mooncake takes 7-8 μs (slightly worse than Stan).
3. Conclusions and future directions for JuliaBUGS:
   1. The current performance bottleneck appears to be related to ReverseDiff's limitations.
   2. A promising path forward is to generate type-stable Julia functions, then use Mooncake or Enzyme.
   3. Hypothesis on ReverseDiff's performance (proposed by Will):
      a. The tape likely stores a lot of fast operations.
      b. The overhead dominates the actual computation.
      c. As a result, the runtime scales almost linearly with the number of instructions on the tape.

optimize the primal function: understanding the type stable number
