import Foundation
import Metal
import simd

enum MeasuredDepthDeprojector {
    static func deprojectedCloud(keyframe: FaceDepthKeyframe, stride strideStep: Int) throws -> [SIMD3<Float>] {
        try MeasuredDepthDeprojectorMetal.instance().deprojectedCloud(
            keyframe: keyframe,
            stride: strideStep
        )
    }
}

private final class MeasuredDepthDeprojectorMetal {
    private static let cache = MeasuredMetalEngineCache<MeasuredDepthDeprojectorMetal>()

    static func instance() throws -> MeasuredDepthDeprojectorMetal {
        try cache.get { try MeasuredDepthDeprojectorMetal() }
    }

    private struct DepthDeprojectionParams {
        var width: UInt32
        var height: UInt32
        var strideStep: UInt32
        var hasMask: UInt32
        var hasLens: UInt32
        var usesInverseTable: UInt32
        var tableCount: UInt32
        var fx: Float
        var fy: Float
        var cx: Float
        var cy: Float
        var lensCenter: SIMD2<Float>
        var targetToLookupScale: SIMD2<Float>
        var maxRadius: Float
    }

    private let context: MeasuredMetalContext
    private let deprojectPipeline: MTLComputePipelineState

    private init() throws {
        let context = try MeasuredMetalContext(operation: "measured depth deprojection")
        self.context = context
        self.deprojectPipeline = try context.makePipeline(named: "tmf_deprojectDepthCloud")
    }

    func deprojectedCloud(keyframe: FaceDepthKeyframe, stride strideStep: Int) throws -> [SIMD3<Float>] {
        let width = keyframe.width
        let height = keyframe.height
        guard width > 0, height > 0, keyframe.depthValues.count == width * height else { return [] }

        let count = width * height
        let hasMask = keyframe.faceMask.count == count
        var depth = keyframe.depthValues
        var mask = hasMask ? keyframe.faceMask.map { UInt8($0 ? 1 : 0) } : [UInt8](repeating: 1, count: count)
        let lens = keyframe.lens
        var table = lens?.table ?? []
        if table.isEmpty {
            table = [0]
        }
        var faceFromCamera = simd_inverse(keyframe.cameraFromFace)
        var params = DepthDeprojectionParams(
            width: UInt32(width),
            height: UInt32(height),
            strideStep: UInt32(max(1, strideStep)),
            hasMask: hasMask ? 1 : 0,
            hasLens: lens == nil ? 0 : 1,
            usesInverseTable: lens?.usesInverseTable == false ? 0 : 1,
            tableCount: UInt32(lens?.table.count ?? 0),
            fx: keyframe.intrinsics.fx,
            fy: keyframe.intrinsics.fy,
            cx: keyframe.intrinsics.cx,
            cy: keyframe.intrinsics.cy,
            lensCenter: lens?.center ?? .zero,
            targetToLookupScale: lens?.targetToLookupScale ?? SIMD2<Float>(repeating: 1),
            maxRadius: lens?.maxRadius ?? 1
        )

        let depthBuffer = try context.makeBuffer(resource: "depth deprojection input") {
            context.device.makeBuffer(bytes: &depth, length: count * MemoryLayout<Float>.stride, options: [])
        }
        let maskBuffer = try context.makeBuffer(resource: "depth deprojection mask") {
            context.device.makeBuffer(bytes: &mask, length: count, options: [])
        }
        let tableBuffer = try context.makeBuffer(resource: "lens distortion table") {
            context.device.makeBuffer(bytes: &table, length: table.count * MemoryLayout<Float>.stride, options: [])
        }
        let pointsBuffer = try context.makeBuffer(resource: "deprojected points") {
            context.device.makeBuffer(length: count * MemoryLayout<SIMD4<Float>>.stride, options: [])
        }
        let validBuffer = try context.makeBuffer(resource: "deprojected point validity") {
            context.device.makeBuffer(length: count, options: [])
        }

        let commandBuffer = try context.makeCommandBuffer()
        let encoder = try context.makeEncoder(commandBuffer: commandBuffer, stage: "depth deprojection")
        encoder.setComputePipelineState(deprojectPipeline)
        encoder.setBuffer(depthBuffer, offset: 0, index: 0)
        encoder.setBuffer(maskBuffer, offset: 0, index: 1)
        encoder.setBuffer(tableBuffer, offset: 0, index: 2)
        encoder.setBuffer(pointsBuffer, offset: 0, index: 3)
        encoder.setBuffer(validBuffer, offset: 0, index: 4)
        encoder.setBytes(&params, length: MemoryLayout<DepthDeprojectionParams>.stride, index: 5)
        encoder.setBytes(
            &faceFromCamera,
            length: MemoryLayout<simd_float4x4>.stride,
            index: 6
        )
        let threadsPerThreadgroup = MTLSize(
            width: min(deprojectPipeline.maxTotalThreadsPerThreadgroup, 16),
            height: 1,
            depth: 1
        )
        encoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: threadsPerThreadgroup
        )
        encoder.endEncoding()
        try context.waitForCompletion(commandBuffer, stage: "depth deprojection")

        let valid = validBuffer.contents().bindMemory(to: UInt8.self, capacity: count)
        let points = pointsBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        var result: [SIMD3<Float>] = []
        result.reserveCapacity(count / max(1, strideStep * strideStep))
        var y = 0
        while y < height {
            var x = 0
            while x < width {
                let index = y * width + x
                if valid[index] != 0 {
                    let point = points[index]
                    result.append(SIMD3<Float>(point.x, point.y, point.z))
                }
                x += max(1, strideStep)
            }
            y += max(1, strideStep)
        }
        return result
    }
}
