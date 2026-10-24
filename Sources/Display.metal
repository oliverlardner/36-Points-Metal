#include <metal_stdlib>
using namespace metal;

struct BlitOut {
    float4 position [[position]];
};

struct Blit {
    float width;
    float height;
    float invert;
};

vertex BlitOut blitVert(uint id [[vertex_id]]) {
    float2 p = float2(float((id << 1u) & 2u), float(id & 2u));
    BlitOut out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

// The original draws the square field twice, side by side, and scales that
// wide image until it covers the window.
fragment float4 blitFrag(
    BlitOut in [[stage_in]],
    texture2d<float, access::sample> field [[texture(0)]],
    constant Blit &blit [[buffer(0)]]
) {
    float2 size = max(float2(blit.width, blit.height), float2(1.0));
    float side = max(size.y, size.x * 0.5);
    float2 origin = 0.5 * (size - float2(side * 2.0, side));
    float2 uv = (in.position.xy - origin) / side;
    uv.x = fract(uv.x);
    constexpr sampler smp(coord::normalized, address::clamp_to_edge, filter::linear);
    float3 color = clamp(field.sample(smp, uv).rgb, 0.0, 1.0);
    if (blit.invert > 0.5) color = float3(1.0) - color;
    return float4(color, 1.0);
}

fragment float4 fieldCopyFrag(
    BlitOut in [[stage_in]],
    texture2d<float, access::sample> source [[texture(0)]],
    constant float2 &size [[buffer(0)]]
) {
    constexpr sampler smp(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv = in.position.xy / max(size, float2(1.0));
    return source.sample(smp, uv);
}
