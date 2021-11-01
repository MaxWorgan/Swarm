using Test 
include(srcdir("data.jl"))

unitNormalise([1 2 3; 3 2 1; 9 9 9])

# @test unitNormalise([1 10 10; 20 10 2; 2 3 4]) == [0? 