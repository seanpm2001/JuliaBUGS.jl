struct Gibbs{N,S} <: AbstractMCMC.AbstractSampler
    sampler_map::OrderedDict{N,S}
    adtype::ADType
end

function Gibbs(sampler_map::ODT) where {ODT<:OrderedDict}
    return Gibbs(sampler_map, ADTypes.AutoReverseDiff(; compile=false))
end

struct MHFromPrior <: AbstractMCMC.AbstractSampler end

abstract type AbstractGibbsState end

"""
    _create_conditioned_submodel_for_gibbs(model::BUGSModel, variables_to_update::Vector{<:VarName})

Create a sub-model containing only the variables specified in `variables_to_update` and their Markov blanket. The 
variables in the Markov blanket will be treated as observed (conditioned on) in the sub-model, while the variables 
in `variables_to_update` will remain unobserved parameters. This allows sampling from the conditional distribution 
of the variables to update given their Markov blanket.
"""
function _create_conditioned_submodel_for_gibbs(model::BUGSModel, variables_to_update::VarName)
    return _create_conditioned_submodel_for_gibbs(model, [variables_to_update])
end
function _create_conditioned_submodel_for_gibbs(
    model::BUGSModel, variables_to_update::Vector{<:VarName}
)
    mb = markov_blanket(model.g, variables_to_update)
    mb_with_vars = union(mb, variables_to_update)

    sub_model = factor(model, mb_with_vars)
    conditioned_sub_model = AbstractPPL.condition(sub_model, mb)

    return conditioned_sub_model
end

struct GibbsState{T,M,C} <: AbstractGibbsState
    values::T
    submodels::M
    states::C
end

function verify_sampler_map(model::BUGSModel, sampler_map::OrderedDict)
    all_variables_in_keys = Set(vcat(keys(sampler_map)...))
    model_parameters = Set(model.parameters)

    # Check for extra variables in sampler_map that are not in model parameters
    extra_variables = setdiff(all_variables_in_keys, model_parameters)
    if !isempty(extra_variables)
        throw(
            ArgumentError(
                "Sampler map contains variables not in the model: $extra_variables"
            ),
        )
    end

    # Check for model parameters not covered by sampler_map
    left_over_variables = setdiff(model_parameters, all_variables_in_keys)
    if !isempty(left_over_variables)
        throw(
            ArgumentError(
                "Some model parameters are not covered by the sampler map: $left_over_variables",
            ),
        )
    end

    return true
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    logdensitymodel::AbstractMCMC.LogDensityModel{<:BUGSModel},
    sampler::Gibbs{N,S};
    model=logdensitymodel.logdensity,
    kwargs...,
) where {N,S}
    verify_sampler_map(model, sampler.sampler_map)

    submodels = Vector{BUGSModel}()
    for (i, variables_to_update) in enumerate(keys(sampler.sampler_map))
        sampler_i = sampler.sampler_map[variables_to_update]
        submodel = _create_conditioned_submodel_for_gibbs(
            model, variables_to_update
        )
        push!(submodels, conditioned_submodel)
    end
    param_values = JuliaBUGS.getparams(model)
    return param_values, GibbsState(param_values, submodels, cached_eval_caches)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    l_model::AbstractMCMC.LogDensityModel{<:BUGSModel},
    sampler::Gibbs,
    state::AbstractGibbsState;
    model=l_model.logdensity,
    kwargs...,
)
    param_values = state.values
    for vs in keys(state.conditioning_schedule)
        model = initialize!(model, param_values)
        cond_model = _create_conditioned_model_for_gibbs(
            model, vs; evaluation_env=model.evaluation_env, eval_cache=state.cached_eval_caches[vs]
        )
        param_values = gibbs_internal(rng, cond_model, state.conditioning_schedule[vs])
    end
    return param_values,
    GibbsState(param_values, state.conditioning_schedule, state.cached_eval_caches)
end

function gibbs_internal end

function gibbs_internal(rng::Random.AbstractRNG, cond_model::BUGSModel, ::MHFromPrior)
    transformed_original = JuliaBUGS.getparams(cond_model)
    values, logp = evaluate!!(cond_model, transformed_original)
    values_proposed, logp_proposed = evaluate!!(rng, cond_model)

    if logp_proposed - logp > log(rand(rng))
        values = values_proposed
    end

    return JuliaBUGS.getparams(
        BangBang.setproperty!!(cond_model.base_model, :evaluation_env, values)
    )
end

function AbstractMCMC.bundle_samples(
    ts,
    logdensitymodel::AbstractMCMC.LogDensityModel{<:JuliaBUGS.BUGSModel},
    sampler::Gibbs,
    state,
    ::Type{T};
    discard_initial=0,
    kwargs...,
) where {T}
    return JuliaBUGS.gen_chains(
        logdensitymodel, ts, [], []; discard_initial=discard_initial, kwargs...
    )
end
