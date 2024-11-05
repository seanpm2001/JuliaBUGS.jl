using JuliaBUGS
using JuliaBUGS: condition, decondition

@testset "condition and decondition" begin
    model_def = @bugs begin
        mu ~ Normal(0, 1)
        sigma ~ Gamma(2, 2)
        for i in 1:3
            y[i] ~ Normal(mu, sigma)
        end
        theta = mu + 2
        x ~ Normal(theta, 1)
        z ~ Normal(0, sigma)
    end
    model = compile(model_def, NamedTuple(), NamedTuple())

    cond_model = condition(model, (; mu=1.0, sigma=2.0))
    @test cond_model.evaluation_env[:mu] == 1.0
    @test cond_model.evaluation_env[:sigma] == 2.0

    @test cond_model.g[@varname(mu)].is_observed == true
    @test cond_model.g[@varname(sigma)].is_observed == true

    cond_model = condition(cond_model, (; z=1.0))
    @test cond_model.g[@varname(z)].is_observed == true

    cond_model = condition(cond_model, (; y=[1.0, 2.0, 3.0]))
    @test cond_model.g[@varname(y[1])].is_observed == true
    @test cond_model.g[@varname(y[2])].is_observed == true
    @test cond_model.g[@varname(y[3])].is_observed == true
    @test cond_model.g[@varname(mu)].is_observed == true
    @test cond_model.g[@varname(sigma)].is_observed == true

    decond_model = decondition(cond_model, [@varname(y[1:2]), @varname(z)])
    @test decond_model.g[@varname(y[1])].is_observed == false
    @test decond_model.g[@varname(y[2])].is_observed == false
    @test decond_model.g[@varname(y[3])].is_observed == true
    @test decond_model.g[@varname(z)].is_observed == false
end
