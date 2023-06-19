import DataStructures.Deque

mutable struct Episode
    startidx::Int
    endidx::Int
    terminated::Bool
end

Base.length(e::Episode) = (e.endidx - e.startidx) + 1

mutable struct Episodes{T} #T <: AbstractTraces ?
    episodes::Deque{Episode}
    traces::T
    length::Int
    pointer::Int
end

Episodes(traces::AbstractTraces) = Episodes(Deque{Episode}(), traces, 0, 0)

Base.getindex(es::Episodes, idx) = getindex(es.traces, idx)
Base.size(es::Episodes) = size(es.traces)
Base.length(es::Episodes) = es.length

function trim(es::Episodes, n = 1)
    difflength = length(es) - length(traces)
    while difflength > 0
        ep = first(es.episodes)
        rem_steps = minimum((n, difflength, length(ep)))
        ep.startidx += rem_steps
        difflength -= rem_steps
        es.length -= rem_steps
        length(ep) == 0 && popfirst!(es.episodes)
        es.pointer = first(es.episodes).startidx
    end
end

for f in (:push!,)
    @eval function Base.$f(es::Episodes, xs::NamedTuple)
        $f(es.traces, xs)
        es.length += 1
        es.pointer += 1
        trim(es)
        if last(es).terminated
            push!(es.episodes, Episode(es.pointer-1, es.pointer-1, false))
        else
            last(es.episodes).endidx += 1
        end
        #create an episode ? What is the startidx ?
    end
end

for f in (:append!,) #append! assumes that complete episodes coming from distributed agents will be appended.
    @eval function Base.$f(es::Episodes, xs::Episodes)
        for ep in xs.episodes
            n = length(ep)
            $f(es.traces, xs.traces[ep.startidx:ep.endidx])
            s = es.pointer
            es.length += n
            es.pointer += n
            trim(es, n)
            push!(es.episodes, Episode(s, es.pointer-1, true))
        end
    end
end

for f in (:pop!, :popfirst!)
    @eval function Base.$f(es::Episodes)
        $f(es.traces)
        trim(es)
    end
end

function Base.empty!(es::Episodes)
    empty!(es.episodes)
    empty!(es.traces)
    es.length = 0
    es.pointer = 0
end