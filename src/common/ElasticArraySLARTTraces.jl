export ElasticArraySLARTTraces

const ElasticArraySLARTTraces = Traces{
    SS′LL′AA′RT,
    <:Tuple{
        <:MultiplexTraces{SS′,<:Trace{<:ElasticArray}},
        <:MultiplexTraces{LL′,<:Trace{<:ElasticArray}},
        <:MultiplexTraces{AA′,<:Trace{<:ElasticArray}},
        <:Trace{<:ElasticArray},
        <:Trace{<:ElasticArray},
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

    MultiplexTraces{SS′}(ElasticArray{state_eltype}(state_size..., capacity + 1)) +
    MultiplexTraces{LL′}(ElasticArray{legal_actions_mask_eltype}(legal_actions_mask_size..., capacity + 1)) +
    MultiplexTraces{AA′}(ElasticArray{action_eltype}(action_size..., capacity + 1)) +
    Traces(
        reward=ElasticArray{reward_eltype}(reward_size..., capacity),
        terminal=ElasticArray{terminal_eltype}(terminal_size..., capacity),
    )
end
