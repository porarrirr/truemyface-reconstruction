#include <metal_stdlib>
using namespace metal;

struct DepthPreprocessParams {
    uint width;
    uint height;
    uint hasMask;
    uint fillOnly;
    float sigmaR;
};

kernel void tmf_initializeDepthValidity(device const float *depth [[buffer(0)]],
                                        device uchar *valid [[buffer(1)]],
                                        constant uint &count [[buffer(2)]],
                                        uint gid [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }
    const float z = depth[gid];
    valid[gid] = (isfinite(z) && z > 0.0f) ? 1 : 0;
}

kernel void tmf_erodeMask(device const uchar *input [[buffer(0)]],
                          device uchar *output [[buffer(1)]],
                          constant uint2 &dimensions [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    const uint width = dimensions.x;
    const uint height = dimensions.y;
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    const uint index = gid.y * width + gid.x;
    if (input[index] == 0) {
        output[index] = 0;
        return;
    }

    for (int dy = -1; dy <= 1; dy++) {
        const int yy = int(gid.y) + dy;
        if (yy < 0 || yy >= int(height)) {
            output[index] = 0;
            return;
        }
        for (int dx = -1; dx <= 1; dx++) {
            const int xx = int(gid.x) + dx;
            if (xx < 0 || xx >= int(width) || input[uint(yy) * width + uint(xx)] == 0) {
                output[index] = 0;
                return;
            }
        }
    }
    output[index] = 1;
}

kernel void tmf_preprocessDepthPass(device const float *current [[buffer(0)]],
                                    device float *out [[buffer(1)]],
                                    device const uchar *valid [[buffer(2)]],
                                    device uchar *outValid [[buffer(3)]],
                                    device const uchar *mask [[buffer(4)]],
                                    device const float *spatial [[buffer(5)]],
                                    constant DepthPreprocessParams &params [[buffer(6)]],
                                    device atomic_uint *filledCount [[buffer(7)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    const uint width = params.width;
    const uint height = params.height;
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    const uint index = gid.y * width + gid.x;
    out[index] = current[index];
    outValid[index] = valid[index];

    if (params.hasMask != 0 && mask[index] == 0) {
        return;
    }

    const bool centerValid = valid[index] != 0;
    if (params.fillOnly != 0 && centerValid) {
        return;
    }

    const int radius = 2;
    const float center = current[index];
    const float sigmaR2 = 2.0f * params.sigmaR * params.sigmaR;
    float weightSum = 0.0f;
    float valueSum = 0.0f;
    uint validNeighbors = 0;

    for (int dy = -radius; dy <= radius; dy++) {
        const int yy = int(gid.y) + dy;
        if (yy < 0 || yy >= int(height)) {
            continue;
        }
        for (int dx = -radius; dx <= radius; dx++) {
            const int xx = int(gid.x) + dx;
            if (xx < 0 || xx >= int(width)) {
                continue;
            }

            const uint neighbor = uint(yy) * width + uint(xx);
            if (valid[neighbor] == 0 || (params.hasMask != 0 && mask[neighbor] == 0)) {
                continue;
            }
            validNeighbors += 1;

            float weight = spatial[uint(dy + radius) * 5u + uint(dx + radius)];
            if (centerValid) {
                const float delta = current[neighbor] - center;
                weight *= exp(-(delta * delta) / sigmaR2);
            }
            weightSum += weight;
            valueSum += weight * current[neighbor];
        }
    }

    if (centerValid) {
        if (weightSum > 0.0f) {
            out[index] = valueSum / weightSum;
        }
        return;
    }

    if (validNeighbors >= 6u && weightSum > 0.0f) {
        out[index] = valueSum / weightSum;
        outValid[index] = 1;
        atomic_fetch_add_explicit(filledCount, 1u, memory_order_relaxed);
    }
}

struct DepthDeprojectionParams {
    uint width;
    uint height;
    uint strideStep;
    uint hasMask;
    uint hasLens;
    uint usesInverseTable;
    uint tableCount;
    float fx;
    float fy;
    float cx;
    float cy;
    float2 lensCenter;
    float2 targetToLookupScale;
    float maxRadius;
};

static float tmf_interpolateLensTable(device const float *table, uint count, float normalized) {
    if (count == 0u) {
        return normalized;
    }
    if (count == 1u) {
        return table[0];
    }
    const float clamped = clamp(normalized, 0.0f, 1.0f);
    const float scaled = clamped * float(count - 1u);
    const uint lower = uint(floor(scaled));
    const uint upper = min(lower + 1u, count - 1u);
    const float fraction = scaled - float(lower);
    return table[lower] * (1.0f - fraction) + table[upper] * fraction;
}

static float2 tmf_applyRadialMagnification(float2 point,
                                           device const float *table,
                                           constant DepthDeprojectionParams &params) {
    const float2 delta = point - params.lensCenter;
    const float radius = length(delta * params.targetToLookupScale);
    if (radius <= 1.0e-6f) {
        return point;
    }
    const float normalized = min(radius / max(params.maxRadius, 1.0e-6f), 1.0f);
    const float magnification = tmf_interpolateLensTable(table, params.tableCount, normalized);
    return params.lensCenter + delta * (1.0f + magnification);
}

static float tmf_mappedRadius(float radius,
                              device const float *table,
                              constant DepthDeprojectionParams &params) {
    const float normalized = min(radius / max(params.maxRadius, 1.0e-6f), 1.0f);
    const float magnification = tmf_interpolateLensTable(table, params.tableCount, normalized);
    return radius * (1.0f + magnification);
}

static float2 tmf_invertRadialMagnification(float2 target,
                                            device const float *table,
                                            constant DepthDeprojectionParams &params) {
    const float2 delta = target - params.lensCenter;
    const float targetRadius = length(delta * params.targetToLookupScale);
    if (targetRadius <= 1.0e-6f) {
        return target;
    }

    const float lastScale = 1.0f + table[params.tableCount - 1u];
    if (!isfinite(lastScale) || lastScale <= 0.0f ||
        !isfinite(params.maxRadius) || params.maxRadius <= 0.0f) {
        return float2(NAN);
    }

    const float mappedMaximumRadius = tmf_mappedRadius(params.maxRadius, table, params);
    if (targetRadius > mappedMaximumRadius) {
        const float sourceRadius = targetRadius / lastScale;
        return params.lensCenter + delta * (sourceRadius / targetRadius);
    }

    float lower = 0.0f;
    float upper = params.maxRadius;
    const float lookupTolerance = max(
        1.0e-6f,
        1.0e-4f * min(params.targetToLookupScale.x, params.targetToLookupScale.y)
    );
    float sourceRadius = (lower + upper) * 0.5f;
    bool converged = false;
    for (uint i = 0u; i < 32u; i++) {
        const float midpoint = (lower + upper) * 0.5f;
        const float mapped = tmf_mappedRadius(midpoint, table, params);
        sourceRadius = midpoint;
        if (abs(mapped - targetRadius) <= lookupTolerance || upper - lower <= lookupTolerance) {
            converged = true;
            break;
        }
        if (mapped < targetRadius) {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    if (!converged) {
        return float2(NAN);
    }
    return params.lensCenter + delta * (sourceRadius / targetRadius);
}

static float2 tmf_rectifiedPixel(float2 observed,
                                 device const float *table,
                                 constant DepthDeprojectionParams &params) {
    if (params.hasLens == 0u || params.tableCount == 0u) {
        return observed;
    }
    if (params.usesInverseTable != 0u) {
        // inverseLensDistortionLookupTable は observed → rectified（直接適用）。
        return tmf_applyRadialMagnification(observed, table, params);
    }
    // lensDistortionLookupTable は rectified → observed なので数値反転する。
    return tmf_invertRadialMagnification(observed, table, params);
}

kernel void tmf_deprojectDepthCloud(device const float *depth [[buffer(0)]],
                                    device const uchar *mask [[buffer(1)]],
                                    device const float *lensTable [[buffer(2)]],
                                    device float4 *points [[buffer(3)]],
                                    device uchar *valid [[buffer(4)]],
                                    constant DepthDeprojectionParams &params [[buffer(5)]],
                                    constant float4x4 &faceFromCamera [[buffer(6)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    const uint width = params.width;
    const uint height = params.height;
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    const uint index = gid.y * width + gid.x;
    valid[index] = 0;
    points[index] = float4(0.0f);

    if ((gid.x % params.strideStep) != 0u || (gid.y % params.strideStep) != 0u) {
        return;
    }
    if (params.hasMask != 0u && mask[index] == 0) {
        return;
    }

    const float z = depth[index];
    if (!isfinite(z) || z <= 0.0f || z >= 1.5f) {
        return;
    }

    const float2 rectified = tmf_rectifiedPixel(float2(float(gid.x), float(gid.y)), lensTable, params);
    const float3 depthOptical = float3(
        (rectified.x - params.cx) * z / params.fx,
        (rectified.y - params.cy) * z / params.fy,
        z
    );
    const float3 camera = float3(depthOptical.x, -depthOptical.y, -depthOptical.z);
    const float4 face4 = faceFromCamera * float4(camera, 1.0f);
    points[index] = float4(face4.xyz, 0.0f);
    valid[index] = 1;
}

struct CylindricalGridSmoothParams {
    uint columns;
    uint rows;
    uint radius;
    float sigmaS;
    float sigmaR;
};

struct CylindricalDiffusionParams {
    uint columns;
    uint rows;
    int iteration;
};

kernel void tmf_initializeCylindricalFill(device const float *grid [[buffer(0)]],
                                          device int *fillDepth [[buffer(1)]],
                                          constant CylindricalDiffusionParams &params [[buffer(2)]],
                                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.columns || gid.y >= params.rows) {
        return;
    }
    const uint index = gid.y * params.columns + gid.x;
    fillDepth[index] = isfinite(grid[index]) ? 0 : -1;
}

kernel void tmf_diffuseCylindricalGrid(device const float *grid [[buffer(0)]],
                                       device const int *fillDepth [[buffer(1)]],
                                       device float *nextGrid [[buffer(2)]],
                                       device int *nextFillDepth [[buffer(3)]],
                                       device atomic_uint *filledCount [[buffer(4)]],
                                       constant CylindricalDiffusionParams &params [[buffer(5)]],
                                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.columns || gid.y >= params.rows) {
        return;
    }

    const uint index = gid.y * params.columns + gid.x;
    const float current = grid[index];
    nextGrid[index] = current;
    nextFillDepth[index] = fillDepth[index];
    if (isfinite(current)) {
        return;
    }

    float sum = 0.0f;
    uint count = 0u;
    for (int dy = -1; dy <= 1; dy++) {
        const int yy = int(gid.y) + dy;
        if (yy < 0 || yy >= int(params.rows)) {
            continue;
        }
        for (int dx = -1; dx <= 1; dx++) {
            const int xx = int(gid.x) + dx;
            if (xx < 0 || xx >= int(params.columns) || (dx == 0 && dy == 0)) {
                continue;
            }
            const uint neighborIndex = uint(yy) * params.columns + uint(xx);
            const float neighbor = grid[neighborIndex];
            if (isfinite(neighbor)) {
                sum += neighbor;
                count += 1u;
            }
        }
    }

    if (count >= 3u) {
        nextGrid[index] = sum / float(count);
        nextFillDepth[index] = params.iteration;
        atomic_fetch_add_explicit(filledCount, 1u, memory_order_relaxed);
    }
}

kernel void tmf_smoothCylindricalGrid(device const float *grid [[buffer(0)]],
                                      device float *output [[buffer(1)]],
                                      constant CylindricalGridSmoothParams &params [[buffer(2)]],
                                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.columns || gid.y >= params.rows) {
        return;
    }

    const uint index = gid.y * params.columns + gid.x;
    const float center = grid[index];
    if (!isfinite(center)) {
        output[index] = center;
        return;
    }

    float weightSum = 0.0f;
    float valueSum = 0.0f;
    const int radius = int(params.radius);
    for (int dy = -radius; dy <= radius; dy++) {
        const int yy = int(gid.y) + dy;
        if (yy < 0 || yy >= int(params.rows)) {
            continue;
        }
        for (int dx = -radius; dx <= radius; dx++) {
            const int xx = int(gid.x) + dx;
            if (xx < 0 || xx >= int(params.columns)) {
                continue;
            }
            const uint neighborIndex = uint(yy) * params.columns + uint(xx);
            const float neighbor = grid[neighborIndex];
            if (!isfinite(neighbor)) {
                continue;
            }
            const float delta = neighbor - center;
            const float spatial = exp(-float(dx * dx + dy * dy) / (2.0f * params.sigmaS * params.sigmaS));
            const float range = exp(-(delta * delta) / (2.0f * params.sigmaR * params.sigmaR));
            const float weight = spatial * range;
            weightSum += weight;
            valueSum += weight * neighbor;
        }
    }

    output[index] = weightSum > 0.0f ? valueSum / weightSum : center;
}

struct CrossMaskParams {
    uint columns;
    uint rows;
};

kernel void tmf_erodeCrossMask(device const uchar *input [[buffer(0)]],
                               device uchar *output [[buffer(1)]],
                               constant CrossMaskParams &params [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.columns || gid.y >= params.rows) {
        return;
    }

    const uint index = gid.y * params.columns + gid.x;
    if (input[index] == 0) {
        output[index] = 0;
        return;
    }

    const bool up = gid.y > 0u && input[index - params.columns] != 0;
    const bool down = gid.y + 1u < params.rows && input[index + params.columns] != 0;
    const bool left = gid.x > 0u && input[index - 1u] != 0;
    const bool right = gid.x + 1u < params.columns && input[index + 1u] != 0;
    output[index] = (up && down && left && right) ? 1 : 0;
}

kernel void tmf_dilateCrossMask(device const uchar *input [[buffer(0)]],
                                device uchar *output [[buffer(1)]],
                                constant CrossMaskParams &params [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.columns || gid.y >= params.rows) {
        return;
    }

    const uint index = gid.y * params.columns + gid.x;
    if (input[index] != 0) {
        output[index] = 1;
        return;
    }

    const bool up = gid.y > 0u && input[index - params.columns] != 0;
    const bool down = gid.y + 1u < params.rows && input[index + params.columns] != 0;
    const bool left = gid.x > 0u && input[index - 1u] != 0;
    const bool right = gid.x + 1u < params.columns && input[index + 1u] != 0;
    output[index] = (up || down || left || right) ? 1 : 0;
}

kernel void tmf_intersectMasks(device const uchar *lhs [[buffer(0)]],
                               device const uchar *rhs [[buffer(1)]],
                               device uchar *output [[buffer(2)]],
                               constant CrossMaskParams &params [[buffer(3)]],
                               uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.columns || gid.y >= params.rows) {
        return;
    }
    const uint index = gid.y * params.columns + gid.x;
    output[index] = (lhs[index] != 0 && rhs[index] != 0) ? 1 : 0;
}

// MARK: - ロバスト MLS 収縮（MeasuredPointCloudMLS.swift）
//
// CompactUniformGrid（CPU 側で counting sort 構築）のフラット索引を受け取り、
// 1) 局所平面フィット（共分散 + Jacobi）で点法線を推定し、
// 2) 各点を法線方向オフセットの「中央値 + MAD 棄却の加重平均」へ動かす。
// アルゴリズム・走査順・perCellCap 打ち切りは旧 CPU 実装（テストの
// legacyMlsCollapsed 参照実装）と同一。

