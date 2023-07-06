export EpisodesBuffer, PartialNamedTuple
import DataStructures.CircularBuffer

"""
    EpisodesBuffer(traces::AbstractTraces)

Wraps an `AbstractTraces` object, usually the container of a `Trajectory`. 
`EpisodesBuffer` tracks the indexes of the `traces` object that belong to the same episodes.
To that end, it stores 
1. an vector `sampleable_inds` of Booleans that determine whether an index in Traces is legally sampleable
(i.e., it is not the index of a last state of an episode);
2. a vector `episodes_lengths` that contains the total duration of the episode that each step belong to;
3. an vector `step_numbers` that contains the index within the episode of the corresponding step.

This information is used to correctly sample the traces. For example, if we have an episode that lasted 10 steps, the buffer stores 11 states. sampleable_inds[i] will
be true for the index of the first ten steps, and 0 at the index of state number 11. episodes_lengths will be 11
consecutive 10s (the episode saw 11 states but 10 steps occured). step_numbers will be 1 to 11.

If `traces` is a capacitated buffer, such as a CircularArraySARTTraces, then these three vectors will also be circular.

EpisodesBuffer assumes that individual transitions are `push!`ed. Appending is not yet supported.
"""

mutable struct EpisodesBuffer{names, E, T<:AbstractTraces{names, E},B,S} <: AbstractTraces{names,E}
    traces::T
    sampleable_inds::S
    step_numbers::B
    episodes_lengths::B
end

"""
    PartialNamedTuple(::NamedTuple)

Wraps a NamedTuple to signal an EpisodesBuffer that it is pushed into that it should 
ignore the fact that this is a partial insertion. Used at the end of an episode to
complete multiplex traces before moving to the next episode.
"""
struct PartialNamedTuple{T}
    namedtuple::T
end

function EpisodesBuffer(traces::AbstractTraces)
    cap = any(t->t isa MultiplexTraces, traces.traces) ? capacity(traces) + 1 : capacity(traces)
    @assert isempty(traces) "EpisodesBuffer must be initialized with empty traces."
    if !isinf(cap)
        legalinds =  CircularBuffer{Bool}(cap)
        step_numbers = CircularBuffer{Int}(cap)
        eplengths = deepcopy(step_numbers)
        EpisodesBuffer(traces, legalinds, step_numbers, eplengths)
    else
        legalinds =  BitVector()
        step_numbers = Vector{Int}()
        eplengths = deepcopy(step_numbers)
        EpisodesBuffer(traces, legalinds, step_numbers, eplengths)
    end
end

Base.getindex(es::EpisodesBuffer, idx...) = getindex(es.traces, idx...)
Base.setindex!(es::EpisodesBuffer, idx...) = setindex!(es.traces, idx...)
Base.size(es::EpisodesBuffer) = size(es.traces)
Base.length(es::EpisodesBuffer) = length(es.traces)
Base.keys(es::EpisodesBuffer) = keys(es.traces)
Base.keys(es::EpisodesBuffer{<:Any,<:Any,<:CircularPrioritizedTraces}) = keys(es.traces.traces)
function Base.show(io::IO, m::MIME"text/plain", eb::EpisodesBuffer{names}) where {names}
    s = nameof(typeof(eb))
    t = eb.traces
    println(io, "$s containing")
    show(io, m::MIME"text/plain", t)
end

ispartial_insert(traces::Traces, xs) = length(xs) < length(traces.traces) #this is the number of traces it contains not the number of steps.
ispartial_insert(es::EpisodesBuffer, xs) = ispartial_insert(es.traces, xs)
ispartial_insert(traces::CircularPrioritizedTraces, xs) = ispartial_insert(traces.traces, xs)

function fill_multiplex(es::EpisodesBuffer)
    for trace in es.traces.traces
        if !(trace isa MultiplexTraces)
            push!(trace, last(trace)) #push a duplicate of last element as a dummy element, should never be sampled.
        end
    end
end
function fill_multiplex(es::EpisodesBuffer{<:Any,<:Any,<:CircularPrioritizedTraces})
    for trace in es.traces.traces.traces
        if !(trace isa MultiplexTraces)
            push!(trace, last(trace)) #push a duplicate of last element as a dummy element, should never be sampled.
        end
    end
end

function Base.push!(eb::EpisodesBuffer, xs::NamedTuple)
    push!(eb.traces, xs)
    partial = ispartial_insert(eb, xs)
    if length(eb.traces) == 0 && partial #first push should be partial
        push!(eb.step_numbers, 1)
        push!(eb.episodes_lengths, 0)
        push!(eb.sampleable_inds, 0)
    elseif !partial #typical inserting
        if length(eb.traces) < length(eb) && length(eb) > 2 #case when PartialNamedTuple is used. Steps are indexable one step later
            eb.sampleable_inds[end-1] = 1 
        else #case when we don't, length of traces and eb will match.
            eb.sampleable_inds[end] = 1 #previous step is now indexable
        end
        push!(eb.sampleable_inds, 0) #this one is no longer
        ep_length = last(eb.step_numbers)
        push!(eb.episodes_lengths, ep_length)
        startidx = max(1,length(eb.step_numbers) - last(eb.step_numbers))
        eb.episodes_lengths[startidx:end] .= ep_length
        push!(eb.step_numbers, ep_length + 1)
    elseif partial
        fill_multiplex(eb)
        eb.sampleable_inds[end] = 0 #previous step is not indexable because it contains the last state
        push!(eb.sampleable_inds, 0) #this one isn't either
        push!(eb.step_numbers, 1)
        push!(eb.episodes_lengths, 0)
    end
    return nothing
end

function Base.push!(eb::EpisodesBuffer, xs::PartialNamedTuple) #wrap a NamedTuple to push without incrementing the step number. 
    push!(eb.traces, xs.namedtuple)
    eb.sampleable_inds[end-1] = 1 #completes the episode trajectory.
end

#= currently unsupported due to lack of support of appending a named tuple to traces with multiplextraces.
for f in (:append!,) #append! assumes that complete episodes coming from distributed agents will be appended.
    @eval function Base.$f(es::EpisodesBuffer, xs::EpisodesBuffer)
        cap = capacity(es.traces)
        for ep in xs.episodes
            n = length(ep)
            $f(es.traces, xs.traces[ep.startidx:ep.startidx+n-1])
            es.length += n
            trim(es, n)
            old_pointer = es.pointer
            push!(es.episodes, Episode(old_pointer, (es.pointer % cap) + n - 1 , n, true))
            es.pointer = ((es.pointer % cap) + n - 1 ) % cap + 1
        end
    end
end=#

for f in (:pop!, :popfirst!)
    @eval function Base.$f(es::EpisodesBuffer)
        $f(es.episodes_lengths)
        $f(es.sampleable_inds)
        $f(es.step_numbers)
        $f(es.traces)
    end
end

function Base.empty!(es::EpisodesBuffer)
    empty!(es.traces)
    empty!(es.episodes_lengths)
    empty!(es.sampleable_inds)
    empty!(es.step_numbers)
end
