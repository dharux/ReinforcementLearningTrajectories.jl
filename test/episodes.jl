using ReinforcementLearningTrajectories
using CircularArrayBuffers
using Test
@testset "EpisodesBuffer" begin
    @testset "push!" begin 
        @testset "with CiruclarTraces" begin
            eb = EpisodesBuffer(
                CircularArraySARTTraces(;
                capacity=10)
            )
            #push a first episode l=5 
            push!(eb, (state = 1, action = 1))
            ep1 = only(eb.episodes)
            for i = 1:5
                push!(eb, (state = i+1, action =i+1, reward = i, terminal = false))
                @test ep1.startidx == 1
                @test ep1.endidx == i
                @test length(ep1) == i
                @test length(eb) == i
                @test length(eb.traces) == i
            end
            
            @test length(eb.traces) == 5
            ep1 = only(eb.episodes)
            #start new episode of 6 periods.
            push!(eb, (state = 7, action = 7))
            @test ep1.terminated == true #mark it as terminated.
            ep2 = last(eb.episodes)
            @test eb[6][:reward] == 0 #6 is not a valid index, the reward there should be a 0
            for i = 8:11
                push!(eb, (state = i, action =i, reward = i-1, terminal = false))
                @test ep2.startidx == 7
                @test ep2.endidx == i - 1
                @test length(ep2) == i - 7
                @test ep1.startidx == 1
                @test ep1.endidx == 5
                @test length(ep1) == 5
                @test length(eb.traces) == i - 1 
            end
            #three last steps replace oldest steps in the buffer.
            for (i, s) = enumerate(12:13)
                println(i)
                push!(eb, (state = s, action =s, reward = s-1, terminal = false))
                @test ep2.startidx == 7 - i
                @test ep2.endidx == 10
                @test ep1.startidx == 1
                @test ep1.endidx == 5 - i
                @test length(ep1) == 5 - i == ep1.endidx - ep1.startidx + 1
                @test length(eb.traces) == 10
                @test length(ep2) == 4 + i 
            end
            @testset "indexing" begin
                #episode 1
                for (i,s) in zip(ep1.startidx:ep1.endidx,3:5)
                    b = eb[i]
                    @test b[:state] == b[:action] == b[:reward] == s
                    @test b[:next_state] == b[:next_action] == s + 1
                end
                #episode 2
                for (i,s) in zip(ep2.startidx:ep2.endidx,7:12)
                    b = eb[i]
                    @test b[:state] == b[:action] == b[:reward] == s
                    @test b[:next_state] == b[:next_action] == s + 1
                end
            end
        end
            #start a third episode
            push!(eb, (state = 14, action = 14))
            @test ep2.terminated == true
            ep3 = last(eb.episodes)
            #push until it reaches it own start
            for (i,s) in enumerate(15:26)
                push!(eb, (state = s, action =s, reward = s-1, terminal = false)); eb.episodes
                @test length(eb.episodes) == (i == 1 ? 3 : i < 9 ? 2 : 1)
                @test length(ep3) == min(10,i)
                @test ep3.endidx == 10
                @test ep3.startidx == max(1,10 - i+1)
            end
            @testset "indexing 2" begin
                for (i,s) in enumerate(collect(15:26)[end-10:end-1])
                    b = eb[i]
                    @test b[:state] == b[:action] == b[:reward] == s
                    @test b[:next_state] == b[:next_action] == s + 1
                end
            end
        @testset "with vector traces" begin
            eb = EpisodesBuffer(
                Traces(;
                    a=Int[],
                    b=Int[])
            )
            for i = 1:15
                push!(eb, (a = i, b =i))
            end
            @test length(eb) == 15
            @test length(eb.traces) == 15
            @test length(eb.episodes) == 1
            for i = 1:15
                @test eb.traces[i][:a] == eb.traces[i][:b] == i
            end
        end
    end
end
