using ReinforcementLearningTrajectories
using CircularArrayBuffers
using Test
@testset "EpisodesBuffer" begin
    @testset "with circular traces" begin 
        eb = EpisodesBuffer(
            CircularArraySARSATTraces(;
            capacity=10)
        )
        #push a first episode l=5 
        push!(eb, (state = 1, action = 1))
        @test eb.sampleable_inds[end] == 0
        @test eb.episodes_lengths[end] == 0
        @test eb.step_numbers[end] == 1
        for i = 1:5
            push!(eb, (state = i+1, action =i+1, reward = i, terminal = false))
            @test eb.sampleable_inds[end] == 0
            @test eb.sampleable_inds[end-1] == 1
            @test eb.step_numbers[end] == i + 1
            @test eb.episodes_lengths[end-i:end] == fill(i, i+1)
        end
        @test eb.sampleable_inds == [1,1,1,1,1,0]
        @test length(eb.traces) == 5
        #start new episode of 6 periods.
        push!(eb, (state = 7, action = 7))
        @test eb.sampleable_inds[end] == 0
        @test eb.sampleable_inds[end-1] == 0
        @test eb.episodes_lengths[end] == 0
        @test eb.step_numbers[end] == 1
        @test eb.sampleable_inds == [1,1,1,1,1,0,0]
        @test eb[6][:reward] == 5 #6 is not a valid index, the reward there is dummy duplicate of previous (5)
        ep2_len = 0
        for (j,i) = enumerate(8:11)
            ep2_len += 1
            push!(eb, (state = i, action =i, reward = i-1, terminal = false))
            @test eb.sampleable_inds[end] == 0
            @test eb.sampleable_inds[end-1] == 1
            @test eb.step_numbers[end] == j + 1
            @test eb.episodes_lengths[end-j:end] == fill(ep2_len, ep2_len + 1)
        end
        @test eb.sampleable_inds == [1,1,1,1,1,0,1,1,1,1,0]
        @test length(eb.traces) == 10
        #three last steps replace oldest steps in the buffer.
        for (i, s) = enumerate(12:13)
            ep2_len += 1
            push!(eb, (state = s, action =s, reward = s-1, terminal = false))
            @test eb.sampleable_inds[end] == 0
            @test eb.sampleable_inds[end-1] == 1
            @test eb.step_numbers[end] == i + 1 + 4
            @test eb.episodes_lengths[end-ep2_len:end] == fill(ep2_len, ep2_len + 1)
        end
        #episode 1
        for (i,s) in enumerate(3:13)
            if i in (4, 11)
                @test eb.sampleable_inds[i] == 0
                continue
            else
                @test eb.sampleable_inds[i] == 1
            end
            b = eb[i]
            @test b[:state] == b[:action] == b[:reward] == s
            @test b[:next_state] == b[:next_action] == s + 1
        end
        #episode 2
        #start a third episode
        push!(eb, (state = 14, action = 14))
        @test eb.sampleable_inds[end] == 0
        @test eb.sampleable_inds[end-1] == 0
        @test eb.episodes_lengths[end] == 0
        @test eb.step_numbers[end] == 1
        #push until it reaches it own start
        for (i,s) in enumerate(15:26)
            push!(eb, (state = s, action =s, reward = s-1, terminal = false))
        end
        @test eb.sampleable_inds == [fill(true, 10); [false]]
        @test eb.episodes_lengths == fill(length(15:26), 11)
        @test eb.step_numbers == [3:13;]
        step = popfirst!(eb)
        @test length(eb) == length(eb.sampleable_inds) - 1 == length(eb.step_numbers) - 1 == length(eb.episodes_lengths) - 1 == 9
        @test first(eb.step_numbers) == 4
        step = pop!(eb)
        @test length(eb) == length(eb.sampleable_inds) - 1 == length(eb.step_numbers) - 1 == length(eb.episodes_lengths) - 1 == 8
        @test last(eb.step_numbers) == 12
        @test size(eb) == size(eb.traces) == (8,)
        empty!(eb)
        @test size(eb) == (0,) == size(eb.traces) == size(eb.sampleable_inds) == size(eb.episodes_lengths) == size(eb.step_numbers)
        show(eb); 
    end
    @testset "with PartialNamedTuple" begin 
        eb = EpisodesBuffer(
            CircularArraySARSATTraces(;
            capacity=10)
        )
        #push a first episode l=5 
        push!(eb, (state = 1,))
        @test eb.sampleable_inds[end] == 0
        @test eb.episodes_lengths[end] == 0
        @test eb.step_numbers[end] == 1
        for i = 1:5
            push!(eb, (state = i+1, action =i, reward = i, terminal = false))
            @test eb.sampleable_inds[end] == 0
            @test eb.sampleable_inds[end-1] == 1
            @test eb.step_numbers[end] == i + 1
            @test eb.episodes_lengths[end-i:end] == fill(i, i+1)
        end
        push!(eb, PartialNamedTuple((action = 6,)))
        @test eb.sampleable_inds == [1,1,1,1,1,0]
        @test length(eb.traces) == 5
        #start new episode of 6 periods.
        push!(eb, (state = 7,))
        @test eb.sampleable_inds[end] == 0
        @test eb.sampleable_inds[end-1] == 0
        @test eb.episodes_lengths[end] == 0
        @test eb.step_numbers[end] == 1
        @test eb.sampleable_inds == [1,1,1,1,1,0,0]
        @test eb[6][:reward] == 5 #6 is not a valid index, the reward there is dummy duplicate of previous (5)
        ep2_len = 0
        for (j,i) = enumerate(8:11)
            ep2_len += 1
            push!(eb, (state = i, action =i-1, reward = i-1, terminal = false))
            @test eb.sampleable_inds[end] == 0
            @test eb.sampleable_inds[end-1] == 1
            @test eb.step_numbers[end] == j + 1
            @test eb.episodes_lengths[end-j:end] == fill(ep2_len, ep2_len + 1)
        end
        @test eb.sampleable_inds == [1,1,1,1,1,0,1,1,1,1,0]
        @test length(eb.traces) == 9 #an action is missing at this stage
        #three last steps replace oldest steps in the buffer.
        for (i, s) = enumerate(12:13)
            ep2_len += 1
            push!(eb, (state = s, action =s-1, reward = s-1, terminal = false))
            @test eb.sampleable_inds[end] == 0
            @test eb.sampleable_inds[end-1] == 1
            @test eb.step_numbers[end] == i + 1 + 4
            @test eb.episodes_lengths[end-ep2_len:end] == fill(ep2_len, ep2_len + 1)
        end
        push!(eb, PartialNamedTuple((action = 13,)))
        @test length(eb.traces) == 10
        #episode 1
        for (i,s) in enumerate(3:13)
            if i in (4, 11)
                @test eb.sampleable_inds[i] == 0
                continue
            else
                @test eb.sampleable_inds[i] == 1
            end
            b = eb[i]
            @test b[:state] == b[:action] == b[:reward] == s
            @test b[:next_state] == b[:next_action] == s + 1
        end
        #episode 2
        #start a third episode
        push!(eb, (state = 14,))
        @test eb.sampleable_inds[end] == 0
        @test eb.sampleable_inds[end-1] == 0
        @test eb.episodes_lengths[end] == 0
        @test eb.step_numbers[end] == 1
        #push until it reaches it own start
        for (i,s) in enumerate(15:26)
            push!(eb, (state = s, action =s-1, reward = s-1, terminal = false))
        end
        push!(eb, PartialNamedTuple((action = 26,)))
        @test eb.sampleable_inds == [fill(true, 10); [false]]
        @test eb.episodes_lengths == fill(length(15:26), 11)
        @test eb.step_numbers == [3:13;]
        step = popfirst!(eb)
        @test length(eb) == length(eb.sampleable_inds) - 1 == length(eb.step_numbers) - 1 == length(eb.episodes_lengths) - 1 == 9
        @test first(eb.step_numbers) == 4
        step = pop!(eb)
        @test length(eb) == length(eb.sampleable_inds) - 1 == length(eb.step_numbers) - 1 == length(eb.episodes_lengths) - 1 == 8
        @test last(eb.step_numbers) == 12
        @test size(eb) == size(eb.traces) == (8,)
        empty!(eb)
        @test size(eb) == (0,) == size(eb.traces) == size(eb.sampleable_inds) == size(eb.episodes_lengths) == size(eb.step_numbers)
        show(eb); 
    end
    @testset "with vector traces" begin
        eb = EpisodesBuffer(
            Traces(;
                state=Int[],
                reward=Int[])
        )
        push!(eb, (state = 1,)) #partial inserting
        for i = 1:15
            push!(eb, (state = i+1, reward =i))
        end
        @test length(eb.traces) == 15
        @test eb.sampleable_inds == [fill(true, 15); [false]]
        @test all(==(15), eb.episodes_lengths)
        @test eb.step_numbers == [1:16;]
        push!(eb, (state = 1,)) #partial inserting
        for i = 1:15
            push!(eb, (state = i+1, reward =i))
        end
        @test eb.sampleable_inds == [fill(true, 15); [false];fill(true, 15); [false]]
        @test all(==(15), eb.episodes_lengths)
        @test eb.step_numbers == [1:16;1:16]
        @test length(eb) == 31
    end
end
