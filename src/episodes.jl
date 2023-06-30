export EpisodesBuffer
import DataStructures.Deque

mutable struct Episode
    startidx::Int
    endidx::Int
    length::Int
    terminated::Bool
end

Base.length(e::Episode) = e.length

"""
    EpisodesBuffer(traces::AbstractTraces)

Wraps an `AbstractTraces` object, usually the container of a `Trajectory`. 
`EpisodesBuffer` tracks the indexes of the `traces` object that belong to the same episodes.
To that end, it stores a Deque of `Episode` objects that keep in memory the traces index of the first
transition of the episode that is still stored, that is when transitions are removed from a circular 
trajectory, the corresponding episodes are also truncated. All of this is automated and needs no intervention
by the user beyond wrapping the traces with an EpisodesBuffer. 

EpisodesBuffer assumes that individual transitions are `push!`ed, and complete episodes are `append!`ed (contained in another EpisodesBuffer). 
"""

mutable struct EpisodesBuffer{T} #T <: AbstractTraces ?
    episodes::Deque{Episode}
    traces::T
    length::Int
    pointer::Int
end

EpisodesBuffer(traces::AbstractTraces) = EpisodesBuffer(Deque{Episode}(), traces, 0, 0)

Base.getindex(es::EpisodesBuffer, idx) = getindex(es.traces, idx)
Base.size(es::EpisodesBuffer) = size(es.traces)
Base.length(es::EpisodesBuffer) = es.length

function trim(es::EpisodesBuffer)
    if length(es) > capacity(es.traces)
        ep1 = first(es.episodes)
        if ep1.startidx == 1
            ep1.length -= 1
        end
        ep1.startidx = 2
        es.length -= 1
        for ep in es.episodes
            ep.startidx -= 1
            ep.endidx = ep.startidx + ep.length - 1
        end
        length(ep1) == 0 && popfirst!(es.episodes)
    end
    return nothing
end

for f in (:push!,)
    @eval function Base.$f(es::EpisodesBuffer, xs::NamedTuple)
        $f(es.traces, xs)
        partial = length(xs) < length(es.traces.traces)
        es.length += 1
        if !isempty(es.episodes) && partial
            for trace in es.traces.traces
                if !(trace isa MultiplexTraces)
                    push!(trace, last(trace)) #push a duplicate of last element as a dummy element, should never be sampled.
                end
            end
            last(es.episodes).terminated = true
            push!(es.episodes, Episode(es.length+1, es.length+1, 0, false))
        elseif isempty(es.episodes)
            push!(es.episodes, Episode(es.length, es.length, 0, false))
            es.length -= 1
        elseif !partial
            ep = last(es.episodes)
            ep.length += 1
            ep.endidx = ep.startidx + ep.length - 1
        end
        trim(es)
        #es.pointer = (es.pointer % capacity(es.traces)) + 1 
    end
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
        $f(es.traces)
        trim(es)
    end
end

function Base.empty!(es::EpisodesBuffer)
    empty!(es.episodes)
    empty!(es.traces)
    es.length = 0
    es.pointer = 0
end
