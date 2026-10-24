#include <metal_stdlib>
using namespace metal;

// Metal port of the 36 Points simulation (2019–2022) by Sage Jenson.
// https://www.sagejenson.com/36points/
// The steering, deposit, blur, and fade below follow that work.
// Licensed CC BY-NC-SA 3.0: https://creativecommons.org/licenses/by-nc-sa/3.0/
// This file is not covered by the MIT license on the rest of the application.

struct Particle {
    float x;
    float y;
    float age;
    float theta;
};

struct FillOut {
    float4 position [[position]];
};

struct PaintOut {
    float4 position [[position]];
    float pointSize [[point_size]];
};

struct PaintUniforms {
    float opacity;
    float pointSize;
    float dotSize;
};

struct BlurUniforms {
    float decay;
    float size;
};

float2 pointsWrap(float2 pos) {
    pos *= 0.5;
    pos += float2(0.5);
    pos -= floor(pos);
    pos -= float2(0.5);
    pos *= 2.0;
    return pos;
}

// Trail samples are taken in the simulation's y-up space. Metal stores the
// top of the texture at v = 0, so the wrapped coordinate is flipped on y.
float2 trailUV(float2 coord) {
    float2 t = pointsWrap(coord);
    t -= floor(t);
    t.y = 1.0 - t.y;
    return t;
}

float pointsNoise(float2 coordinate, float seed) {
    return fract(tan(distance(coordinate * (seed + 0.118446744073709551614),
        float2(0.118446744073709551614, 0.314159265358979323846264))) * 0.141421356237309504880169);
}

float trailSample(texture2d<float, access::sample> trail, float2 coord) {
    constexpr sampler smp(coord::normalized, address::repeat, filter::linear);
    return trail.sample(smp, trailUV(coord)).r;
}

kernel void pointsUpdate(
    device Particle *particles [[buffer(0)]],
    constant float *v [[buffer(1)]],
    constant float &simSize [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    texture2d<float, access::sample> trail [[texture(0)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    Particle p = particles[id];
    float2 position = float2(p.x, p.y);
    float theta = p.theta;
    float hd = simSize * 0.5;
    float2 dir = float2(cos(theta), sin(theta));
    float2 sp = 0.5 * (position + float2(1.0));

    float sv = trailSample(trail, sp + v[13] / hd * dir + float2(0.0, v[12] / hd));
    sv = max(sv, 0.000000001);
    float sd = v[0] / hd + v[2] * pow(sv, v[1]) * 250.0 / hd;
    float md = v[9] / hd + v[11] * pow(sv, v[10]) * 250.0 / hd;
    float sa = v[3] + v[5] * pow(sv, v[4]);
    float ra = v[6] + v[8] * pow(sv, v[7]);

    float mid = trailSample(trail, sp + sd * float2(cos(theta), sin(theta)));
    float left = trailSample(trail, sp + sd * float2(cos(theta + sa), sin(theta + sa)));
    float right = trailSample(trail, sp + sd * float2(cos(theta - sa), sin(theta - sa)));
    float heading = theta;
    if (mid > left && mid > right) {
    } else if (mid < left && mid < right) {
        if (pointsNoise(position * 1332.4324, theta) > 0.5) heading += ra;
        else heading -= ra;
    } else if (left < right) {
        heading -= ra;
    } else if (left > right) {
        heading += ra;
    }

    float2 nextDir = float2(cos(heading), sin(heading));
    float2 wrapped = pointsWrap(position + nextDir * md);
    particles[id].x = wrapped.x;
    particles[id].y = wrapped.y;
    particles[id].age = fract(p.age + 0.0005);
    particles[id].theta = heading;
}

vertex FillOut pointsFillVert(uint id [[vertex_id]]) {
    float2 p = float2(float((id << 1u) & 2u), float(id & 2u));
    FillOut out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

vertex PaintOut pointsPaintVert(
    uint id [[vertex_id]],
    device Particle *particles [[buffer(0)]],
    constant PaintUniforms &paint [[buffer(1)]]
) {
    Particle p = particles[id];
    PaintOut out;
    out.position = float4(p.x, p.y, 0.0, 1.0);
    out.pointSize = id == 0u ? paint.dotSize : paint.pointSize;
    return out;
}

fragment float4 pointsPaintFrag(
    PaintOut in [[stage_in]],
    float2 coord [[point_coord]],
    constant PaintUniforms &paint [[buffer(0)]]
) {
    if (dot(coord - 0.5, coord - 0.5) > 0.25) discard_fragment();
    return float4(1.0, 1.0, 1.0, paint.opacity);
}

fragment float4 pointsBlurFrag(
    FillOut in [[stage_in]],
    texture2d<float, access::sample> source [[texture(0)]],
    constant BlurUniforms &blur [[buffer(0)]]
) {
    constexpr sampler smp(coord::normalized, address::repeat, filter::linear);
    float2 size = float2(max(blur.size, 1.0));
    float2 uv = in.position.xy / size;
    float2 px = 1.0 / size;
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            sum += source.sample(smp, uv + float2(x, y) * px).r;
        }
    }
    return float4(sum / 9.0 * blur.decay, 0.0, 0.0, 1.0);
}

fragment float4 pointsFadeFrag(constant float &amount [[buffer(0)]]) {
    return float4(0.0, 0.0, 0.0, amount);
}
