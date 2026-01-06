// ---------------------------------------------
// Shadertoy-style minimal fragment shader in Slang
// ---------------------------------------------

// Slang uses HLSL-like semantics, but feels close to GLSL.
// We explicitly model the "Shadertoy contract".

cbuffer Uniforms
{
    float2 iResolution;   // viewport resolution (pixels)
    float  iTime;         // time in seconds
    float  _pad;          // padding (cbuffer alignment)
};

// Fullscreen triangle vertex shader
struct VSOut
{
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
};

VSOut vsMain(uint vertexID : SV_VertexID)
{
    // Fullscreen triangle (no vertex buffer)
    float2 pos[3] = {
        float2(-1.0, -1.0),
        float2(-1.0,  3.0),
        float2( 3.0, -1.0)
    };

    float2 uv = (pos[vertexID] + 1.0) * 0.5;

    VSOut o;
    o.position = float4(pos[vertexID], 0.0, 1.0);
    o.uv = uv;
    return o;
}

// Shadertoy-style "mainImage"
float4 mainImage(float2 fragCoord)
{
    float2 uv = fragCoord / iResolution;
    float2 p  = uv * 2.0 - 1.0;

    float t = iTime;

    // Simple analytic field
    float r = length(p);
    float a = atan2(p.y, p.x);

    float v = sin(10.0 * r - 3.0 * t + sin(a * 3.0));

    float3 col = 0.5 + 0.5 * cos(
        float3(0.0, 2.0, 4.0) + v + t
    );

    return float4(col, 1.0);
}

// Pixel shader
/*
// slang comes from a HLSL Legacy: Not suitable for WebGPU / WGSL-compatible. Use the one below instead.
float4 psMain(VSOut i) : SV_Target
{
    float2 fragCoord = i.uv * iResolution;
    return mainImage(fragCoord);
}
*/
float4 psMain(
    float2 uv : TEXCOORD0
) : SV_Target {
    float2 fragCoord = uv * iResolution;
    return mainImage(fragCoord);
}

