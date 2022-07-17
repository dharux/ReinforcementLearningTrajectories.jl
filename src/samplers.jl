using Random

abstract type AbstractSampler end

#####
# DummySampler
#####

export DummySampler

struct DummySampler end

sample(s::DummySampler, t::AbstractTraces) = t

#####
# BatchSampler
#####

export BatchSampler
struct BatchSampler{names} <: AbstractSampler
    batch_size::Int
    rng::Random.AbstractRNG
end

"""
    BatchSampler{names}(;batch_size, rng=Random.GLOBAL_RNG)
    BatchSampler{names}(batch_size ;rng=Random.GLOBAL_RNG)

Uniformly sample a batch of examples for each trace specified in `names`. 
By default, all the traces will be sampled.

See also [`sample`](@ref).
"""
BatchSampler(batch_size; kw...) = BatchSampler(; batch_size=batch_size, kw...)
BatchSampler(; kw...) = BatchSampler{nothing}(; kw...)
BatchSampler{names}(batch_size; kw...) where {names} = BatchSampler{names}(; batch_size=batch_size, kw...)
BatchSampler{names}(; batch_size, rng=Random.GLOBAL_RNG) where {names} = BatchSampler{names}(batch_size, rng)

sample(s::BatchSampler{nothing}, t::AbstractTraces) = sample(s, t, keys(t))
sample(s::BatchSampler{names}, t::AbstractTraces) where {names} = sample(s, t, names)

function sample(s::BatchSampler, t::AbstractTraces, names)
    inds = rand(s.rng, 1:length(t), s.batch_size)
    NamedTuple{names}(map(x -> t[x][inds], names))
end

#####

sample(s::BatchSampler{nothing}, t::CircularPrioritizedTraces) = sample(s, t, keys(t.traces))

function sample(s::BatchSampler, t::CircularPrioritizedTraces, names)
    inds, priorities = rand(s.rng, t.priorities, s.batch_size)
    NamedTuple{(:key, :priority, names...)}((t.keys[inds], priorities, map(x -> t.traces[x][inds], names)...))
end

#####
# MetaSampler
#####

export MetaSampler

"""
    MetaSampler(::NamedTuple)

Wraps a NamedTuple containing multiple samplers. When sampled, returns a named tuple with a 
batch from each sampler.
Used internally for algorithms that sample multiple times per epoch.
Note that a single "sampling" with a MetaSampler only increases the Trajectory controler 
count by 1, not by the number of internal samplers. This should be taken into account when
initializing an agent.


# Example
```
MetaSampler(policy = BatchSampler(10), critic = BatchSampler(100))
```
"""
struct MetaSampler{names,T} <: AbstractSampler
    samplers::NamedTuple{names,T}
end

MetaSampler(; kw...) = MetaSampler(NamedTuple(kw))

sample(s::MetaSampler, t) = map(x -> sample(x, t), s.samplers)

#####
# MultiBatchSampler
#####

export MultiBatchSampler

"""
    MultiBatchSampler(sampler, n)

Wraps a sampler. When sampled, will sample n batches using sampler. Useful in combination 
with MetaSampler to allow different sampling rates between samplers.
Note that a single "sampling" with a MultiBatchSampler only increases the Trajectory 
controler count by 1, not by `n`. This should be taken into account when
initializing an agent.

# Example
```
MetaSampler(policy = MultiBatchSampler(BatchSampler(10), 3), 
            critic = MultiBatchSampler(BatchSampler(100), 5))
```
"""
struct MultiBatchSampler{S<:AbstractSampler} <: AbstractSampler
    sampler::S
    n::Int
end

sample(m::MultiBatchSampler, t) = [sample(m.sampler, t) for _ in 1:m.n]

#####
# NStepBatchSampler
#####

export NStepBatchSampler

mutable struct NStepBatchSampler{traces}
    n::Int # !!! n starts from 1
    γ::Float32
    batch_size::Int
    stack_size::Union{Nothing,Int}
    rng::Any
end

NStepBatchSampler(; kw...) = NStepBatchSampler{SS′ART}(; kw...)
NStepBatchSampler{names}(; n, γ, batch_size=32, stack_size=nothing, rng=Random.GLOBAL_RNG) where {names} = NStepBatchSampler{names}(n, γ, batch_size, stack_size, rng)

function sample(s::NStepBatchSampler{names}, ts) where {names}
    valid_range = isnothing(s.stack_size) ? (1:(length(ts)-s.n+1)) : (s.stack_size:(length(ts)-s.n+1))# think about the exteme case where s.stack_size == 1 and s.n == 1
    inds = rand(s.rng, valid_range, s.batch_size)
    sample(s, ts, Val(names), inds)
end

function sample(nbs::NStepBatchSampler, ts, ::Val{SS′ART}, inds)
    if isnothing(nbs.stack_size)
        s = ts[:state][inds]
        s′ = ts[:next_state][inds.+(nbs.n-1)]
    else
        s = ts[:state][[x + i for i in -nbs.stack_size+1:0, x in inds]]
        s′ = ts[:next_state][[x + nbs.n - 1 + i for i in -nbs.stack_size+1:0, x in inds]]
    end

    a = ts[:action][inds]
    t_horizon = ts[:terminal][[x + j for j in 0:nbs.n-1, x in inds]]
    r_horizon = ts[:reward][[x + j for j in 0:nbs.n-1, x in inds]]

    @assert ndims(t_horizon) == 2
    t = any(t_horizon, dims=1) |> vec

    @assert ndims(r_horizon) == 2
    r = map(eachcol(r_horizon), eachcol(t_horizon)) do r⃗, t⃗
        foldr(((rr, tt), init) -> rr + nbs.γ * init * (1 - tt), zip(r⃗, t⃗); init=0.0f0)
    end

    NamedTuple{SS′ART}((s, s′, a, r, t))
end

function sample(s::NStepBatchSampler, ts, ::Val{SS′L′ART}, inds)
    s, s′, a, r, t = sample(s, ts, Val(SSART), inds)
    l = consecutive_view(ts[:next_legal_actions_mask], inds)
    NamedTuple{SSLART}((s, s′, l, a, r, t))
end

function sample(s::NStepBatchSampler{names}, t::CircularPrioritizedTraces) where {names}
    inds, priorities = rand(s.rng, t.priorities, s.batch_size)
    merge(
        (key=t.keys[inds], priority=priorities),
        sample(s, t.traces, Val(names), inds)
    )
end