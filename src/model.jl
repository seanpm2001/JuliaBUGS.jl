"""
    EvalCache{TNF,TNA,TV}

Pre-compute the values of the nodes in the model to avoid lookups from MetaGraph.
"""
struct EvalCache{TNF,TNA,TV}
    sorted_nodes::Vector{<:VarName}
    is_stochastic_vals::Vector{Bool}
    is_observed_vals::Vector{Bool}
    node_function_vals::TNF
    node_args_vals::TNA
    loop_vars_vals::TV
end

function EvalCache(sorted_nodes::Vector{<:VarName}, g::BUGSGraph)
    is_stochastic_vals = Array{Bool}(undef, length(sorted_nodes))
    is_observed_vals = Array{Bool}(undef, length(sorted_nodes))
    node_function_vals = []
    node_args_vals = []
    loop_vars_vals = []
    for (i, vn) in enumerate(sorted_nodes)
        (; is_stochastic, is_observed, node_function, node_args, loop_vars) = g[vn]
        is_stochastic_vals[i] = is_stochastic
        is_observed_vals[i] = is_observed
        push!(node_function_vals, node_function)
        push!(node_args_vals, Val(node_args))
        push!(loop_vars_vals, loop_vars)
    end
    return EvalCache(
        sorted_nodes,
        is_stochastic_vals,
        is_observed_vals,
        node_function_vals,
        node_args_vals,
        loop_vars_vals,
    )
end

# `BUGSModel` does not subtype `AbstractPPL.AbstractProbabilisticProgram` since we use
# the `LogDensityProblems` pipeline. If it did subtype `AbstractPPL.AbstractProbabilisticProgram` 
# (which subtypes `AbstractMCMC.AbstractModel`), then it would not be wrapped in a 
# `LogDensityModel` as intended.

"""
    BUGSModel

The `BUGSModel` object is used for inference and represents the output of compilation. It implements the
[`LogDensityProblems.jl`](https://github.com/tpapp/LogDensityProblems.jl) interface.
"""
struct BUGSModel{T,TNF,TNA,TV}
    " Indicates whether the model parameters are in the transformed space. "
    transformed::Bool

    "The length of the parameters vector in the original (constrained) space."
    untransformed_param_length::Int
    "The length of the parameters vector in the transformed (unconstrained) space."
    transformed_param_length::Int
    "A dictionary mapping the names of the variables to their lengths in the original (constrained) space."
    untransformed_var_lengths::Dict{<:VarName,Int}
    "A dictionary mapping the names of the variables to their lengths in the transformed (unconstrained) space."
    transformed_var_lengths::Dict{<:VarName,Int}

    "A `NamedTuple` containing the values of the variables in the model, all the values are in the constrained space."
    evaluation_env::T

    "A vector containing the names of the model parameters (unobserved stochastic variables)."
    parameters::Vector{<:VarName}

    "An `EvalCache` object containing pre-computed values of the nodes in the model. For each topological order, this needs to be recomputed."
    eval_cache::EvalCache{TNF,TNA,TV}

    "An instance of `BUGSGraph`, representing the dependency graph of the model."
    g::BUGSGraph
end

function Base.show(io::IO, model::BUGSModel)
    if model.transformed
        println(
            io,
            "BUGSModel (transformed, with dimension $(model.transformed_param_length)):",
            "\n",
        )
    else
        println(
            io,
            "BUGSModel (untransformed, with dimension $(model.untransformed_param_length)):",
            "\n",
        )
    end
    println(io, "  Model parameters:")
    println(io, "    ", join(model.parameters, ", "), "\n")
    println(io, "  Variable values:")
    return println(io, "$(model.evaluation_env)")
end

"""
    parameters(m::BUGSModel)

Return a vector of `VarName` containing the names of the model parameters (unobserved stochastic variables).
"""
parameters(m::BUGSModel) = m.parameters

"""
    variables(m::BUGSModel)

Return a vector of `VarName` containing the names of all the variables in the model.
"""
variables(m::BUGSModel) = collect(labels(m.g))

@generated function prepare_arg_values(
    ::Val{args}, evaluation_env::NamedTuple, loop_vars::NamedTuple{lvars}
) where {args,lvars}
    fields = []
    for arg in args
        if arg in lvars
            push!(fields, :(loop_vars[$(QuoteNode(arg))]))
        else
            push!(fields, :(evaluation_env[$(QuoteNode(arg))]))
        end
    end
    return :(NamedTuple{$(args)}(($(fields...),)))
end

function BUGSModel(
    g::BUGSGraph,
    evaluation_env::NamedTuple,
    initial_params::NamedTuple=NamedTuple();
    is_transformed::Bool=true,
)
    sorted_nodes = VarName[label_for(g, node) for node in topological_sort(g)]
    parameters = VarName[]
    untransformed_param_length, transformed_param_length = 0, 0
    untransformed_var_lengths, transformed_var_lengths = Dict{VarName,Int}(),
    Dict{VarName,Int}()

    for vn in sorted_nodes
        (; is_stochastic, is_observed, node_function, node_args, loop_vars) = g[vn]
        args = prepare_arg_values(Val(node_args), evaluation_env, loop_vars)
        if !is_stochastic
            value = Base.invokelatest(node_function; args...)
            evaluation_env = BangBang.setindex!!(evaluation_env, value, vn)
        elseif !is_observed
            push!(parameters, vn)
            dist = Base.invokelatest(node_function; args...)

            untransformed_var_lengths[vn] = length(dist)
            # not all distributions are defined for `Bijectors.transformed`
            transformed_var_lengths[vn] = if Bijectors.bijector(dist) == identity
                untransformed_var_lengths[vn]
            else
                length(Bijectors.transformed(dist))
            end
            untransformed_param_length += untransformed_var_lengths[vn]
            transformed_param_length += transformed_var_lengths[vn]

            initialization = try
                AbstractPPL.get(initial_params, vn)
            catch _
                missing
            end
            if !ismissing(initialization)
                evaluation_env = BangBang.setindex!!(evaluation_env, initialization, vn)
            else
                init_value = try
                    rand(dist)
                catch e
                    error(
                        "Failed to sample from the prior distribution of $vn, consider providing initialization values for $vn or it's parents: $(collect(MetaGraphsNext.inneighbor_labels(g, vn))...).",
                    )
                end
                evaluation_env = BangBang.setindex!!(evaluation_env, init_value, vn)
            end
        end
    end
    return BUGSModel(
        is_transformed,
        untransformed_param_length,
        transformed_param_length,
        untransformed_var_lengths,
        transformed_var_lengths,
        evaluation_env,
        parameters,
        EvalCache(sorted_nodes, g),
        g,
        nothing,
    )
end

function BUGSModel(
    model::BUGSModel,
    g::BUGSGraph,
    parameters::Vector{<:VarName},
    sorted_nodes::Vector{<:VarName},
    evaluation_env::NamedTuple=model.evaluation_env,
)
    return BUGSModel(
        model.transformed,
        sum(model.untransformed_var_lengths[v] for v in parameters),
        sum(model.transformed_var_lengths[v] for v in parameters),
        model.untransformed_var_lengths,
        model.transformed_var_lengths,
        evaluation_env,
        parameters,
        EvalCache(sorted_nodes, g),
        g,
    )
end

"""
    initialize!(model::BUGSModel, initial_params::NamedTuple)

Initialize the model with a NamedTuple of initial values, the values are expected to be in the original space.
"""
function initialize!(model::BUGSModel, initial_params::NamedTuple)
    check_input(initial_params)
    for (i, vn) in enumerate(model.eval_cache.sorted_nodes)
        is_stochastic = model.eval_cache.is_stochastic_vals[i]
        is_observed = model.eval_cache.is_observed_vals[i]
        node_function = model.eval_cache.node_function_vals[i]
        node_args = model.eval_cache.node_args_vals[i]
        loop_vars = model.eval_cache.loop_vars_vals[i]
        args = prepare_arg_values(node_args, model.evaluation_env, loop_vars)
        if !is_stochastic
            value = Base.invokelatest(node_function; args...)
            BangBang.@set!! model.evaluation_env = setindex!!(
                model.evaluation_env, value, vn
            )
        elseif !is_observed
            initialization = try
                AbstractPPL.get(initial_params, vn)
            catch _
                missing
            end
            if !ismissing(initialization)
                BangBang.@set!! model.evaluation_env = setindex!!(
                    model.evaluation_env, initialization, vn
                )
            else
                BangBang.@set!! model.evaluation_env = setindex!!(
                    model.evaluation_env,
                    rand(Base.invokelatest(node_function; args...)),
                    vn,
                )
            end
        end
    end
    return model
end

"""
    initialize!(model::BUGSModel, initial_params::AbstractVector)

Initialize the model with a vector of initial values, the values can be in transformed space if `model.transformed` is set to true.
"""
function initialize!(model::BUGSModel, initial_params::AbstractVector)
    evaluation_env, _ = AbstractPPL.evaluate!!(model, initial_params)
    return BangBang.setproperty!!(model, :evaluation_env, evaluation_env)
end

"""
    getparams(model::BUGSModel)

Extract the parameter values from the model as a flattened vector, in an order consistent with
the what `LogDensityProblems.logdensity` expects.
"""
function getparams(model::BUGSModel)
    param_length = if model.transformed
        model.transformed_param_length
    else
        model.untransformed_param_length
    end

    param_vals = Vector{Float64}(undef, param_length)
    pos = 1
    for v in model.parameters
        if !model.transformed
            val = AbstractPPL.get(model.evaluation_env, v)
            len = model.untransformed_var_lengths[v]
            if val isa AbstractArray
                param_vals[pos:(pos + len - 1)] .= vec(val)
            else
                param_vals[pos] = val
            end
        else
            (; node_function, node_args, loop_vars) = model.g[v]
            args = prepare_arg_values(Val(node_args), model.evaluation_env, loop_vars)
            dist = node_function(; args...)
            transformed_value = Bijectors.transform(
                Bijectors.bijector(dist), AbstractPPL.get(model.evaluation_env, v)
            )
            len = model.transformed_var_lengths[v]
            if transformed_value isa AbstractArray
                param_vals[pos:(pos + len - 1)] .= vec(transformed_value)
            else
                param_vals[pos] = transformed_value
            end
        end
        pos += len
    end
    return param_vals
end

"""
    getparams(T::Type{<:AbstractDict}, model::BUGSModel)

Extract the parameter values from the model into a dictionary of type T.
If model.transformed is true, returns parameters in transformed space.
"""
function getparams(T::Type{<:AbstractDict}, model::BUGSModel)
    d = T()
    for v in model.parameters
        value = AbstractPPL.get(model.evaluation_env, v)
        if !model.transformed
            d[v] = value
        else
            (; node_function, node_args, loop_vars) = model.g[v]
            args = prepare_arg_values(Val(node_args), model.evaluation_env, loop_vars)
            dist = node_function(; args...)
            d[v] = Bijectors.transform(Bijectors.bijector(dist), value)
        end
    end
    return d
end

"""
    settrans(model::BUGSModel, bool::Bool=!(model.transformed))

The `BUGSModel` contains information for evaluation in both transformed and untransformed spaces. The `transformed` field
indicates the current "mode" of the model.

This function enables switching the "mode" of the model.
"""
function settrans(model::BUGSModel, bool::Bool=!(model.transformed))
    return BangBang.setproperty!!(model, :transformed, bool)
end

"""
    AbstractPPL.condition(model::BUGSModel, variables_to_condition_on_and_values)

Condition the model on the given variables and values.
"""
function AbstractPPL.condition(
    model::BUGSModel, variables_to_condition_on_and_values::NamedTuple
)
    return AbstractPPL.condition(
        model,
        Dict(VarName{k}() => v for (k, v) in pairs(variables_to_condition_on_and_values)),
    )
end
function AbstractPPL.condition(
    model::BUGSModel, variables_to_condition_on_and_values::Dict{<:VarName,<:Any}
)
    # Update the evaluation environment with the conditioned values
    evaluation_env = model.evaluation_env
    for (variable, value) in pairs(variables_to_condition_on_and_values)
        evaluation_env = BangBang.setindex!!(evaluation_env, value, variable)
    end
    model = BangBang.setproperty!!(model, :evaluation_env, evaluation_env)

    return AbstractPPL.condition(model, keys(variables_to_condition_on_and_values))
end
function AbstractPPL.condition(
    model::BUGSModel, variables_to_condition_on::Vector{<:VarName}
)
    # Track which variables we're conditioning on
    variables_to_condition_on = VarName[]

    # Process each variable we want to condition on
    for vn in variables_to_condition_on
        if vn ∈ labels(model.g)
            # Variable exists directly in graph - condition on it
            model = _set_is_observed(model, vn, true)
            push!(variables_to_condition_on, vn)
        else
            # Check if this refers to a group of variables
            subsumed_vars = _get_subsumed_vars(model, vn)
            if !isempty(subsumed_vars)
                # Condition on each subsumed variable
                append!(variables_to_condition_on, subsumed_vars)
                for v in subsumed_vars
                    model = _set_is_observed(model, v, true)
                end
            else
                throw(ArgumentError("Variable $vn does not exist in the model"))
            end
        end
    end

    # Update parameters by removing conditioned variables and recalculate lengths
    parameters = setdiff(model.parameters, variables_to_condition_on)
    untransformed_param_length = sum(model.untransformed_var_lengths[v] for v in parameters)
    transformed_param_length = sum(model.transformed_var_lengths[v] for v in parameters)

    return BUGSModel(
        model.transformed,
        untransformed_param_length,
        transformed_param_length,
        model.untransformed_var_lengths,
        model.transformed_var_lengths,
        evaluation_env,
        parameters,
        model.eval_cache,
        model.g,
    )
end

function AbstractPPL.decondition(
    model::BUGSModel, variables_to_decondition::Vector{<:VarName}
)
    variables_to_decondition_on = VarName[]

    for vn in variables_to_decondition
        if vn ∈ labels(model.g)
            model = _set_is_observed(model, vn, false)
            push!(variables_to_decondition_on, vn)
        else
            subsumed_vars = _get_subsumed_vars(model, vn)
            if !isempty(subsumed_vars)
                for v in subsumed_vars
                    model = _set_is_observed(model, v, false)
                    push!(variables_to_decondition_on, v)
                end
            else
                throw(ArgumentError("Variable $vn does not exist in the model"))
            end
        end
    end

    parameters = union(model.parameters, variables_to_decondition_on)
    untransformed_param_length = sum(model.untransformed_var_lengths[v] for v in parameters)
    transformed_param_length = sum(model.transformed_var_lengths[v] for v in parameters)

    return BUGSModel(
        model.transformed,
        untransformed_param_length,
        transformed_param_length,
        model.untransformed_var_lengths,
        model.transformed_var_lengths,
        model.evaluation_env,
        parameters,
        model.eval_cache,
        model.g,
    )
end

function _set_is_observed(model::BUGSModel, variable::VarName, is_observed::Bool)
    if !model.g[variable].is_stochastic
        throw(
            ArgumentError(
                "$variable is not a stochastic variable; changing observation status is not supported",
            ),
        )
    elseif model.g[variable].is_observed == is_observed
        if is_observed
            @warn "$variable is already an observed variable; no changes made"
        else
            @warn "$variable is already treated as a model parameter; no changes made"
        end
    else
        new_g = copy(model.g)
        new_g[variable] = BangBang.setproperty!!(
            model.g[variable], :is_observed, is_observed
        )
        model = BangBang.setproperty!!(model, :g, new_g)
    end
    return model
end

function _get_subsumed_vars(model::BUGSModel, vn::VarName)
    subsumed_vars = VarName[]
    if vn ∉ labels(model.g)
        for v in labels(model.g)
            if subsumes(vn, v)
                push!(subsumed_vars, v)
            end
        end
    end
    return subsumed_vars
end

function factor(model::BUGSModel, variables_to_include::Vector{<:VarName})
    sorted_nodes = filter(vn -> vn in variables_to_include, model.eval_cache.sorted_nodes)
    eval_cache = EvalCache(sorted_nodes, model.g)
    parameters = intersect(model.parameters, variables_to_include)
    return BUGSModel(
        model.transformed,
        sum(model.untransformed_var_lengths[v] for v in parameters),
        sum(model.transformed_var_lengths[v] for v in parameters),
        model.untransformed_var_lengths,
        model.transformed_var_lengths,
        model.evaluation_env,
        parameters,
        eval_cache,
        model.g,
    )
end

function AbstractPPL.evaluate!!(rng::Random.AbstractRNG, model::BUGSModel)
    (; evaluation_env, g) = model
    vi = deepcopy(evaluation_env)
    logp = 0.0
    for (i, vn) in enumerate(model.eval_cache.sorted_nodes)
        is_stochastic = model.eval_cache.is_stochastic_vals[i]
        node_function = model.eval_cache.node_function_vals[i]
        node_args = model.eval_cache.node_args_vals[i]
        loop_vars = model.eval_cache.loop_vars_vals[i]
        args = prepare_arg_values(node_args, evaluation_env, loop_vars)
        if !is_stochastic
            value = node_function(; args...)
            evaluation_env = setindex!!(evaluation_env, value, vn)
        else
            dist = node_function(; args...)
            value = rand(rng, dist) # just sample from the prior
            logp += logpdf(dist, value)
            evaluation_env = setindex!!(evaluation_env, value, vn)
        end
    end
    return evaluation_env, logp
end

function AbstractPPL.evaluate!!(model::BUGSModel)
    logp = 0.0
    evaluation_env = deepcopy(model.evaluation_env)
    for (i, vn) in enumerate(model.eval_cache.sorted_nodes)
        is_stochastic = model.eval_cache.is_stochastic_vals[i]
        node_function = model.eval_cache.node_function_vals[i]
        node_args = model.eval_cache.node_args_vals[i]
        loop_vars = model.eval_cache.loop_vars_vals[i]
        args = prepare_arg_values(node_args, evaluation_env, loop_vars)
        if !is_stochastic
            value = node_function(; args...)
            evaluation_env = setindex!!(evaluation_env, value, vn)
        else
            dist = node_function(; args...)
            value = AbstractPPL.get(evaluation_env, vn)
            if model.transformed
                # although the values stored in `evaluation_env` are in their original space, 
                # here we behave as accepting a vector of parameters in the transformed space
                value_transformed = Bijectors.transform(Bijectors.bijector(dist), value)
                logp +=
                    Distributions.logpdf(dist, value) + Bijectors.logabsdetjac(
                        Bijectors.inverse(Bijectors.bijector(dist)), value_transformed
                    )
            else
                logp += Distributions.logpdf(dist, value)
            end
        end
    end
    return evaluation_env, logp
end

function AbstractPPL.evaluate!!(model::BUGSModel, flattened_values::AbstractVector)
    var_lengths = if model.transformed
        model.transformed_var_lengths
    else
        model.untransformed_var_lengths
    end

    evaluation_env = deepcopy(model.evaluation_env)
    current_idx = 1
    logp = 0.0
    for (i, vn) in enumerate(model.eval_cache.sorted_nodes)
        is_stochastic = model.eval_cache.is_stochastic_vals[i]
        is_observed = model.eval_cache.is_observed_vals[i]
        node_function = model.eval_cache.node_function_vals[i]
        node_args = model.eval_cache.node_args_vals[i]
        loop_vars = model.eval_cache.loop_vars_vals[i]
        args = prepare_arg_values(node_args, evaluation_env, loop_vars)
        if !is_stochastic
            value = node_function(; args...)
            evaluation_env = BangBang.setindex!!(evaluation_env, value, vn)
        else
            dist = node_function(; args...)
            if !is_observed
                l = var_lengths[vn]
                if model.transformed
                    b = Bijectors.bijector(dist)
                    b_inv = Bijectors.inverse(b)
                    reconstructed_value = reconstruct(
                        b_inv,
                        dist,
                        view(flattened_values, current_idx:(current_idx + l - 1)),
                    )
                    value, logjac = Bijectors.with_logabsdet_jacobian(
                        b_inv, reconstructed_value
                    )
                else
                    value = reconstruct(
                        dist, view(flattened_values, current_idx:(current_idx + l - 1))
                    )
                    logjac = 0.0
                end
                current_idx += l
                logp += logpdf(dist, value) + logjac
                evaluation_env = BangBang.setindex!!(evaluation_env, value, vn)
            else
                logp += logpdf(dist, AbstractPPL.get(evaluation_env, vn))
            end
        end
    end
    return evaluation_env, logp
end
