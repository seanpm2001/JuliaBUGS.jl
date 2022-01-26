# more examples can be found here: https://www.mrc-bsu.cam.ac.uk/software/bugs/

regression = @bugsast begin
    for i in 1:N
        Y[i] ~ dnorm(μ[i], τ)
        μ[i] = α + β * (x[i] - x̄)
    end
    τ ~ dgamma(0.001, 0.001)
    σ = 1 / sqrt(τ)
    logτ = log(τ)
    α = dnorm(0.0, 1e-6)
    β = dnorm(0.0, 1e-6)
end

regression_data = (
    x = [1,2,3,4,5],
    Y = [1,3,3,3,5],
    x̄ = 3,
    N = 5
)


rats = @bugsast begin
    for i in 1:N
        for j in 1:T
            Y[i, j] ~ dnorm(μ[i, j], τ_c)
            μ[i, j] = α[i] + β[i] * (x[j] - x̄)
        end
        α[i] ~ dnorm(α_c, α_τ)
        β[i] ~ dnorm(β_c, β_τ)
    end

    τ_c ~ dgamma(0.001, 0.001)
    σ = 1 / sqrt(τ_c)
    α_c ~ dnorm(0.0, 1e-6)
    α_τ ~ dgamma(0.001, 0.001)
    β_c ~ dnorm(0.0, 1e-6)
    β_τ ~ dgamma(0.001, 0.001)
    α₀ = α_c - x̄ * β_c
end

hearts = @bugsast begin
    for i in 1:N
        y[i] ~ dbin(q[i], t[i])
        q[i] = P[state1[i]]
        state1[i] = state[i] + 1
        state[i] ~ dbern(θ)
        t[i] = x[i] + y[i]
    end

    P[1] = p
    P[2] = 0
    logit(p) = α
    α ~ dnorm(0, 1e-4)
    β = exp(α)
    logit(θ) = δ
    delta ~ dnorm(0, 1e-4)
end


regions1 = @bugsast begin
    x[1] = 10
    x[2] ~ dnorm(0, 1)
    for i in 1:x[1]
        y[i] = i
    end
end

regions2 = @bugsast begin
    x[2] ~ dnorm(0, 1)
    for i in 1:x[1]
        y[i] = i
    end
end

regions3 = @bugsast begin
    x1 = 10
    x2 ~ dnorm(0, 1)
    for i in 1:x1
        y[i] = i
    end
end