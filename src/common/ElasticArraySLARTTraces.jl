export ElasticArraySLARTTraces

const ElasticArraySLARTTraces = Traces{
    SS′LL′AA′RT,
    <:Tuple{
        <:MultiplexTraces{SS′,<:Trace{<:ElasticArrayBuffer}},
        <:MultiplexTraces{LL′,<:Trace{<:ElasticArrayBuffer}},
        <:MultiplexTraces{AA′,<:Trace{<:ElasticArrayBuffer}},
        <:Trace{<:ElasticArrayBuffer},
        <:Trace{<:ElasticArrayBuffer},
    }
}

function ElasticArraySLARTTraces(;
    capacity::Int,
    state=Int => (),
    legal_actions_mask=Bool => (),
    action=Int => (),
    reward=Float32 => (),
    terminal=Bool => ()
)
    state_eltype, state_size = state
    action_eltype, action_size = action
    legal_actions_mask_eltype, legal_actions_mask_size = legal_actions_mask
    reward_eltype, reward_size = reward
    terminal_eltype, terminal_size = terminal

    MultiplexTraces{SS′}(ElasticArrayBuffer{state_eltype}(state_size..., capacity + 1)) +
    MultiplexTraces{LL′}(ElasticArrayBuffer{legal_actions_mask_eltype}(legal_actions_mask_size..., capacity + 1)) +
    MultiplexTraces{AA′}(ElasticArrayBuffer{action_eltype}(action_size..., capacity + 1)) +
    Traces(
        reward=ElasticArrayBuffer{reward_eltype}(reward_size..., capacity),
        terminal=ElasticArrayBuffer{terminal_eltype}(terminal_size..., capacity),
    )
end
