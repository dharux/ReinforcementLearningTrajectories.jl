using Random
export EpisodesSampler, Episode, BatchSampler, NStepBatchSampler, MetaSampler, MultiBatchSampler, DummySampler

struct SampleGenerator{S,T}
    sampler::S
    traces::T
end

Base.iterate(s::SampleGenerator) = StatsBase.sample(s.sampler, s.traces), nothing
Base.iterate(s::SampleGenerator, ::Nothing) = nothing

#####
# DummySampler
#####

export DummySampler

"""
Just return the underlying traces.
"""
struct DummySampler end

StatsBase.sample(::DummySampler, t) = t

#####
# BatchSampler
#####

export BatchSampler

struct BatchSampler{names}
    batch_size::Int
    rng::Random.AbstractRNG
end

"""
    BatchSampler{names}(;batch_size, rng=Random.GLOBAL_RNG)
    BatchSampler{names}(batch_size ;rng=Random.GLOBAL_RNG)

Uniformly sample **ONE** batch of `batch_size` examples for each trace specified
in `names`. If `names` is not set, all the traces will be sampled.
"""
BatchSampler(batch_size; kw...) = BatchSampler(; batch_size=batch_size, kw...)
BatchSampler(; kw...) = BatchSampler{nothing}(; kw...)
BatchSampler{names}(batch_size; kw...) where {names} = BatchSampler{names}(; batch_size=batch_size, kw...)
BatchSampler{names}(; batch_size, rng=Random.GLOBAL_RNG) where {names} = BatchSampler{names}(batch_size, rng)

StatsBase.sample(s::BatchSampler{nothing}, t::AbstractTraces) = StatsBase.sample(s, t, keys(t))
StatsBase.sample(s::BatchSampler{names}, t::AbstractTraces) where {names} = StatsBase.sample(s, t, names)

function StatsBase.sample(s::BatchSampler, t::AbstractTraces, names, weights = StatsBase.UnitWeights{Int}(length(t)))
    inds = StatsBase.sample(s.rng, 1:length(t), weights, s.batch_size)
    NamedTuple{names}(map(x -> collect(t[Val(x)][inds]), names))
end

function StatsBase.sample(s::BatchSampler, t::EpisodesBuffer, names)
    StatsBase.sample(s, t.traces, names, StatsBase.FrequencyWeights(t.sampleable_inds[1:end-1]))
end

# !!! avoid iterating an empty trajectory
function Base.iterate(s::SampleGenerator{<:BatchSampler})
    if length(s.traces) > 0
        StatsBase.sample(s.sampler, s.traces), nothing
    else
        nothing
    end
end

#####

StatsBase.sample(s::BatchSampler{nothing}, t::CircularPrioritizedTraces) = StatsBase.sample(s, t, keys(t.traces))

function StatsBase.sample(s::BatchSampler, e::EpisodesBuffer{<:Any, <:Any, <:CircularPrioritizedTraces}, names)
    t = e.traces
    st = deepcopy(t.priorities)
    st .*= e.sampleable_inds[1:end-1] #temporary sumtree that puts 0 priority to non sampleable indices.
    inds, priorities = rand(s.rng, st, s.batch_size)
    NamedTuple{(:key, :priority, names...)}((t.keys[inds], priorities, map(x -> collect(t.traces[Val(x)][inds]), names)...))
end

function StatsBase.sample(s::BatchSampler, t::CircularPrioritizedTraces, names)
    inds, priorities = rand(s.rng, t.priorities, s.batch_size)
    NamedTuple{(:key, :priority, names...)}((t.keys[inds], priorities, map(x -> collect(t.traces[Val(x)][inds]), names)...))
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
struct MetaSampler{names,T}
    samplers::NamedTuple{names,T}
end

MetaSampler(; kw...) = MetaSampler(NamedTuple(kw))

StatsBase.sample(s::MetaSampler, t) = map(x -> StatsBase.sample(x, t), s.samplers)

function Base.iterate(s::SampleGenerator{<:MetaSampler})
    if length(s.traces) > 0
        StatsBase.sample(s.sampler, s.traces), nothing
    else
        nothing
    end
end

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
struct MultiBatchSampler{S}
    sampler::S
    n::Int
end

StatsBase.sample(m::MultiBatchSampler, t) = [StatsBase.sample(m.sampler, t) for _ in 1:m.n]

function Base.iterate(s::SampleGenerator{<:MultiBatchSampler})
    if length(s.traces) > 0
        StatsBase.sample(s.sampler, s.traces), nothing
    else
        nothing
    end
end

#####
# NStepBatchSampler
#####

export NStepBatchSampler

"""
    NStepBatchSampler{names}(; n, γ, batch_size=32, stack_size=nothing, rng=Random.GLOBAL_RNG)

Used to sample a discounted sum of consecutive rewards in the framework of n-step TD learning.
The "next" element of Multiplexed traces (such as the next_state or the next_action) will be 
that in up to `n > 1` steps later in the buffer. The reward will be
the discounted sum of the `n` rewards, with `γ` as the discount factor.

NStepBatchSampler may also be used with n ≥ 1 to sample a "stack" of states if `stack_size` is set 
to an integer > 1. This samples the (stack_size - 1) previous states. This is useful in the case
of partial observability, for example when the state is approximated by `stack_size` consecutive 
frames.
"""
mutable struct NStepBatchSampler{names, S <: Union{Nothing,Int}}
    n::Int # !!! n starts from 1
    γ::Float32
    batch_size::Int
    stack_size::S
    rng::Any
end

NStepBatchSampler(; kw...) = NStepBatchSampler{SS′ART}(; kw...)
function NStepBatchSampler{names}(; n, γ, batch_size=32, stack_size=nothing, rng=Random.GLOBAL_RNG) where {names} 
    @assert n >= 1 "n must be ≥ 1."
    NStepBatchSampler{names}(n, γ, batch_size, stack_size == 1 ? nothing : stack_size, rng)
end

function valid_range_nbatchsampler(s::NStepBatchSampler, eb::EpisodesBuffer)
    range = copy(eb.sampleable_inds)
    stack_size = isnothing(s.stack_size) ? 1 : s.stack_size 
    for idx in eachindex(range)
        valid = eb.step_numbers[idx] >= stack_size && eb.step_numbers[idx] <= eb.episodes_lengths[idx] + 1 - eb.n && eb.sampleable_inds[idx]
        range[idx] = valid
    end
    return range
end

function StatsBase.sample(s::NStepBatchSampler{names}, ts) where {names}
    StatsBase.sample(s, ts, Val(names))
end

function StatsBase.sample(s::NStepBatchSampler, t::EpisodesBuffer, names)
    valid_range = valid_range_nbatchsampler(s, t)
    StatsBase.sample(s, t.traces, names, StatsBase.FrequencyWeights(valid_range))
end

function StatsBase.sample(s::NStepBatchSampler, t::AbstractTraces, names, weights = StatsBase.UnitWeights{Int}(length(t)))
    inds = StatsBase.sample(s.rng, 1:length(t), weights, s.batch_size)
    NamedTuple{names}map(name -> collect(fetch(s, ts[name], Val(name), inds)), names)
end

#state and next_state have specialized fetch methods due to stack_size
fetch(::NStepBatchSampler{names, Nothing}, trace, ::Val{:state}, inds) where {names} = trace[inds]
fetch(s::NStepBatchSampler{names, Int}, trace, ::Val{:state}, inds) where {names} = trace[[x + s.n - 1 + i for i in -s.stack_size+1:0, x in inds]]
fetch(s::NStepBatchSampler{names, Nothing}, trace, ::Val{:next_state}, inds) where {names} = trace[inds.+(s.n-1)]
fetch(s::NStepBatchSampler{names, Int}, trace, ::Val{:next_state}, inds) where {names} = trace[[x + s.n - 1 + i for i in -s.stack_size+1:0, x in inds]]
#reward due to discounting
function fetch(s::NStepBatchSampler{names}, trace, ::Val{:reward}, inds) where {names} 
    rewards = trace[[x + j for j in 0:nbs.n-1, x in inds]]
    return reduce((x,y)->x + s.γ*y, rewards, init = zero(eltype(rewards)), dims = 1)
end
#terminal is that of the nth step
fetch(s::NStepBatchSampler{names}, trace, ::Val{:terminal}, inds) where {names} = trace[inds.+s.n]
#right multiplex traces must be n-step sampled
fetch(::NStepBatchSampler{names}, trace::RelativeTrace{1,0} , ::Val{<:Symbol}, inds) where {names} = trace[inds.+(s.n-1)]
#normal trace types are fetched at inds
fetch(::NStepBatchSampler{names}, trace, ::Val{<:Symbol}, inds) where {names} = trace[inds] #other types of trace are sampled normaly

function StatsBase.sample(s::NStepBatchSampler{names}, e::EpisodesBuffer{<:Any, <:Any, <:CircularPrioritizedTraces}) where {names}
    t = e.traces
    st = deepcopy(t.priorities)
    st .*= valid_range_nbatchsampler(s,e) #temporary sumtree that puts 0 priority to non sampleable indices.
    inds, priorities = rand(s.rng, st, s.batch_size)
    merge(
        (key=t.keys[inds], priority=priorities),
        fetch(s, t.traces, Val(names), inds)
    )
end

"""
    EpisodesSampler()

A sampler that samples all Episodes present in the Trajectory and divides them into 
Episode containers. Truncated Episodes (e.g. due to the buffer capacity) are sampled as well.
There will be at most one truncated episode and it will always be the first one. 
"""
struct EpisodesSampler{names}
end

EpisodesSampler() = EpisodesSampler{nothing}()
#EpisodesSampler{names}() = new{names}()


struct Episode{names, N <: NamedTuple{names}}
    nt::N
end

@forward Episode.nt Base.keys, Base.haskey, Base.getindex

StatsBase.sample(s::EpisodesSampler{nothing}, t::EpisodesBuffer) = StatsBase.sample(s,t,keys(t))
StatsBase.sample(s::EpisodesSampler{names}, t::EpisodesBuffer) where names = StatsBase.sample(s,t,names)

function make_episode(t::EpisodesBuffer, range, names)
    nt = NamedTuple{names}(map(x -> collect(t[Val(x)][range]), names))
    Episode(nt)
end

function StatsBase.sample(::EpisodesSampler, t::EpisodesBuffer, names)
    ranges = UnitRange{Int}[]
    idx = 1
    while idx < length(t)
        if t.sampleable_inds[idx] == 1
            last_state_idx = idx + t.episodes_lengths[idx] - t.step_numbers[idx]
            push!(ranges,idx:last_state_idx)
            idx = last_state_idx + 1
        else
            idx += 1
        end
    end
    
    return [make_episode(t, r, names) for r in ranges]
end
