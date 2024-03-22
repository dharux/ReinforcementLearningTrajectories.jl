export ElasticArraySARTSTraces

const ElasticArraySARTSTraces = Traces{
    SS′ART,
    <:Tuple{
        <:MultiplexTraces{SS′,<:Trace{<:ElasticArrayBuffer}},
        <:Trace{<:ElasticArrayBuffer},
        <:Trace{<:ElasticArrayBuffer},
        <:Trace{<:ElasticArrayBuffer},
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

    MultiplexTraces{SS′}(ElasticArrayBuffer{state_eltype}(state_size..., capacity+1)) +
    Traces(
        action = ElasticArrayBuffer{action_eltype}(action_size..., capacity),
        reward=ElasticArrayBuffer{reward_eltype}(reward_size..., capacity),
        terminal=ElasticArrayBuffer{terminal_eltype}(terminal_size..., capacity),
    )
end
