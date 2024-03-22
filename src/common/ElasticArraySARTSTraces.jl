export ElasticArraySARTSTraces

const ElasticArraySARTSTraces = Traces{
    SS′ART,
    <:Tuple{
        <:MultiplexTraces{SS′,<:Trace{<:ElasticArray}},
        <:Trace{<:ElasticArray},
        <:Trace{<:ElasticArray},
        <:Trace{<:ElasticArray},
    }
}

function ElasticArraySARTSTraces(;
    state=Int => (),
    action=Int => (),
    reward=Float32 => (),
    terminal=Bool => ())
    
    state_eltype, state_size = state
    action_eltype, action_size = action
    reward_eltype, reward_size = reward
    terminal_eltype, terminal_size = terminal

    MultiplexTraces{SS′}(ElasticArray{state_eltype}(state_size..., Inf)) +
    Traces(
        action = ElasticArray{action_eltype}(action_size..., Inf),
        reward=ElasticArray{reward_eltype}(reward_size..., Inf),
        terminal=ElasticArray{terminal_eltype}(terminal_size..., Inf),
    )
end
