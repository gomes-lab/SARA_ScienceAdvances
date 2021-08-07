module TestUtil
using Test
using SARA
using SARA: interpolate_xrd_stripe

@testset "util" begin
    n = 128
    x = range(-1, 1, length = n)
    m = 5
    Y = randn(n, m)
    Yx = interpolate_xrd_stripe(x, Y, x)
    @test Yx ≈ Y
end

end
