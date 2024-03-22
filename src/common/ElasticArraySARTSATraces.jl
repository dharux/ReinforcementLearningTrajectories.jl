export ElasticArraySARTSATraces

const ElasticArraySARTSATraces = Traces{
    SS′AA′RT,
    <:Tuple{
        <:MultiplexTraces{SS′,<:Trace{<:ElasticArray}},
        <:MultiplexTraces{AA′,<:Trace{<:ElasticArray}},
        <:Trace{<:ElasticArray},
        <:Trace{<:ElasticArray},
    }
}

function ElasticArraySARTSATraces(;
    state=Int => (),
    action=Int => (),
    reward=Float32 => (),
    terminal=Bool => ()
)
    state_eltype, state_size = state
    action_eltype, action_size = action
    reward_eltype, reward_size = reward
    terminal_eltype, terminal_size = terminal

    MultiplexTraces{SS′}(ElasticArray{state_eltype}(undef, state_size..., 0)) +
    MultiplexTraces{AA′}(ElasticArray{action_eltype}(undef, action_size..., 0)) +
    Traces(
        reward=ElasticArray{reward_eltype}(undef, reward_size..., 0),
        terminal=ElasticArray{terminal_eltype}(undef, terminal_size..., 0),
    )
end

