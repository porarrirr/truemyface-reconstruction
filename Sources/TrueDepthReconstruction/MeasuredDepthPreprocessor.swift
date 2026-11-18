import Foundation
import Metal

enum MeasuredDepthPreprocessor {
    static func erodedMask(_ mask: [Bool], width: Int, height: Int, iterations: Int) throws -> [Bool] {
        try MeasuredDepthPreprocessorMetal.instance().erodedMask(
            mask,
            width: width,
            height: height,
            iterations: iterations
        )
    }

    struct FaceDepthLayer: Equatable {
        var mask: [Bool]
        /// 顔層より完全に奥にあるとして除外した画素数。
        var removedBehindPixelCount: Int
    }

    /// 顔マスク内の実測画素を深度の連続性で連結成分に分け、最大成分（顔面）より
    /// 「完全に奥」にある成分だけをマスクから外す。
    ///
    /// 前景マスク（追跡器の投影や画像 ROI 由来）は実際の顔輪郭より数 px はみ出し、その画素は
    /// 顔の後方（壁など、実測で +40〜60cm）を測っている。この背景点は TSDF の領域推定を
    /// 数十 cm 広げてボクセルを粗くし（実測 1.5mm → 2.43mm）、ICP にも入る。
    /// 顔より手前の成分（前髪など）や、奥行きが顔層と重なる成分は残す。
    /// 輪郭の判定は深度の段差のみで行い、顔形状テンプレートは使わない。
    static func faceDepthLayer(depth: [Float],
                               mask: [Bool],
                               width: Int,
                               height: Int,
                               maximumDepth: Float,
                               jumpMeters: Float) -> FaceDepthLayer {
        let count = width * height
        guard width > 0, height > 0, depth.count == count, mask.count == count,
              jumpMeters.isFinite, jumpMeters > 0 else {
            return FaceDepthLayer(mask: mask, removedBehindPixelCount: 0)
        }
        var parent = [Int32](repeating: -1, count: count)
        depth.withUnsafeBufferPointer { d in
            mask.withUnsafeBufferPointer { m in
                parent.withUnsafeMutableBufferPointer { p in
                    func isMeasured(_ i: Int) -> Bool {
                        m[i] && d[i].isFinite && d[i] > 0 && d[i] < maximumDepth
                    }
                    func find(_ i: Int) -> Int {
                        var root = i
                        while Int(p[root]) != root { root = Int(p[root]) }
                        var node = i
                        while Int(p[node]) != root {
                            let next = Int(p[node])
                            p[node] = Int32(root)
                            node = next
                        }
                        return root
                    }
                    func union(_ a: Int, _ b: Int) {
                        let ra = find(a), rb = find(b)
                        if ra != rb { p[max(ra, rb)] = Int32(min(ra, rb)) }
                    }
                    for i in 0..<count where isMeasured(i) { p[i] = Int32(i) }
                    for y in 0..<height {
                        for x in 0..<width {
                            let i = y * width + x
                            guard p[i] >= 0 else { continue }
                            if x + 1 < width, p[i + 1] >= 0, abs(d[i + 1] - d[i]) <= jumpMeters {
                                union(i, i + 1)
                            }
                            if y + 1 < height, p[i + width] >= 0, abs(d[i + width] - d[i]) <= jumpMeters {
                                union(i, i + width)
                            }
                        }
                    }
                    for i in 0..<count where p[i] >= 0 { p[i] = Int32(find(i)) }
                }
            }
        }

        var sizes: [Int32: Int] = [:]
        var minimumDepth: [Int32: Float] = [:]
        var maximumDepthByRoot: [Int32: Float] = [:]
        for i in 0..<count where parent[i] >= 0 {
            let root = parent[i]
            sizes[root, default: 0] += 1
            minimumDepth[root] = min(minimumDepth[root] ?? .infinity, depth[i])
            maximumDepthByRoot[root] = max(maximumDepthByRoot[root] ?? -.infinity, depth[i])
        }
        // 同数の場合は root 番号の小さい方を選び、入力だけで結果が決まるようにする。
        guard let main = sizes.max(by: { lhs, rhs in
            lhs.value != rhs.value ? lhs.value < rhs.value : lhs.key > rhs.key
        })?.key, let mainMaximum = maximumDepthByRoot[main] else {
            return FaceDepthLayer(mask: mask, removedBehindPixelCount: 0)
        }
        let behind = Set(minimumDepth.compactMap { root, minimum in
            minimum > mainMaximum ? root : nil
        })
        guard !behind.isEmpty else {
            return FaceDepthLayer(mask: mask, removedBehindPixelCount: 0)
        }
        var out = mask
        var removed = 0
        for i in 0..<count where parent[i] >= 0 && behind.contains(parent[i]) {
            out[i] = false
            removed += 1
        }
        return FaceDepthLayer(mask: out, removedBehindPixelCount: removed)
    }

    static func preprocessedDepth(_ depth: [Float],
                                  mask: [Bool],
                                  width: Int,
                                  height: Int,
                                  configuration: MeasuredFaceReconstructor.Configuration) throws -> [Float] {
        try MeasuredDepthPreprocessorMetal.instance().preprocessedDepth(
            depth,
            mask: mask,
            width: width,
            height: height,
            configuration: configuration
        )
    }
}

private final class MeasuredDepthPreprocessorMetal {
    private static let cache = MeasuredMetalEngineCache<MeasuredDepthPreprocessorMetal>()

    static func instance() throws -> MeasuredDepthPreprocessorMetal {
        try cache.get { try MeasuredDepthPreprocessorMetal() }
    }

    private struct DepthPreprocessParams {
        var width: UInt32
        var height: UInt32
        var hasMask: UInt32
        var fillOnly: UInt32
        var sigmaR: Float
    }

    private let context: MeasuredMetalContext
    private let initializeValidityPipeline: MTLComputePipelineState
    private let erodeMaskPipeline: MTLComputePipelineState
    private let preprocessPassPipeline: MTLComputePipelineState
    private let spatialWeights: [Float]

    private init() throws {
        let context = try MeasuredMetalContext(operation: "measured depth preprocessing")
        self.context = context
        self.initializeValidityPipeline = try context.makePipeline(named: "tmf_initializeDepthValidity")
        self.erodeMaskPipeline = try context.makePipeline(named: "tmf_erodeMask")
        self.preprocessPassPipeline = try context.makePipeline(named: "tmf_preprocessDepthPass")

        let radius = 2
        let sigmaS: Float = 1.6
        var weights: [Float] = []
        weights.reserveCapacity((2 * radius + 1) * (2 * radius + 1))
        for dy in -radius...radius {
            for dx in -radius...radius {
                weights.append(expf(-Float(dx * dx + dy * dy) / (2 * sigmaS * sigmaS)))
            }
        }
        self.spatialWeights = weights
    }

    func erodedMask(_ mask: [Bool], width: Int, height: Int, iterations: Int) throws -> [Bool] {
        guard iterations > 0, mask.count == width * height else { return mask }
        let count = mask.count
        var inputMask = mask.map { UInt8($0 ? 1 : 0) }
        let input = try context.makeBuffer(resource: "eroded mask input") {
            context.device.makeBuffer(bytes: &inputMask, length: count, options: [])
        }
        let output = try context.makeBuffer(resource: "eroded mask output") {
            context.device.makeBuffer(length: count, options: [])
        }

        var source = input
        var destination = output
        for _ in 0..<iterations {
            var dimensions = SIMD2<UInt32>(UInt32(width), UInt32(height))
            try encode(
                pipeline: erodeMaskPipeline,
                bufferSetup: { encoder in
                    encoder.setBuffer(source, offset: 0, index: 0)
                    encoder.setBuffer(destination, offset: 0, index: 1)
                    encoder.setBytes(&dimensions, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                },
                width: width,
                height: height
            )
            swap(&source, &destination)
        }

        let raw = source.contents().bindMemory(to: UInt8.self, capacity: count)
        return (0..<count).map { raw[$0] != 0 }
    }

    func preprocessedDepth(_ depth: [Float],
                           mask: [Bool],
                           width: Int,
                           height: Int,
                           configuration: MeasuredFaceReconstructor.Configuration) throws -> [Float] {
        guard depth.count == width * height else { return depth }

        let count = depth.count
        let hasMask = mask.count == count
        var inputDepth = depth
        var maskBytes = hasMask ? mask.map { UInt8($0 ? 1 : 0) } : [UInt8](repeating: 1, count: count)
        var spatial = spatialWeights

        let currentDepth = try context.makeBuffer(resource: "current depth") {
            context.device.makeBuffer(bytes: &inputDepth, length: count * MemoryLayout<Float>.stride, options: [])
        }
        let nextDepth = try context.makeBuffer(resource: "next depth") {
            context.device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: [])
        }
        let currentValid = try context.makeBuffer(resource: "current depth validity") {
            context.device.makeBuffer(length: count, options: [])
        }
        let nextValid = try context.makeBuffer(resource: "next depth validity") {
            context.device.makeBuffer(length: count, options: [])
        }
        let maskBuffer = try context.makeBuffer(resource: "depth mask") {
            context.device.makeBuffer(bytes: &maskBytes, length: count, options: [])
        }
        let spatialBuffer = try context.makeBuffer(resource: "spatial weights") {
            context.device.makeBuffer(bytes: &spatial, length: spatial.count * MemoryLayout<Float>.stride, options: [])
        }

        try initializeValidity(depthBuffer: currentDepth, validBuffer: currentValid, count: count)

        var sourceDepth = currentDepth
        var destinationDepth = nextDepth
        var sourceValid = currentValid
        var destinationValid = nextValid

        _ = try preprocessPass(
            sourceDepth: sourceDepth,
            destinationDepth: destinationDepth,
            sourceValid: sourceValid,
            destinationValid: destinationValid,
            mask: maskBuffer,
            spatial: spatialBuffer,
            width: width,
            height: height,
            hasMask: hasMask,
            fillOnly: false,
            sigmaR: configuration.bilateralRangeSigma
        )
        swap(&sourceDepth, &destinationDepth)
        swap(&sourceValid, &destinationValid)

        for _ in 1..<max(1, configuration.inpaintIterations) {
            let filled = try preprocessPass(
                sourceDepth: sourceDepth,
                destinationDepth: destinationDepth,
                sourceValid: sourceValid,
                destinationValid: destinationValid,
                mask: maskBuffer,
                spatial: spatialBuffer,
                width: width,
                height: height,
                hasMask: hasMask,
                fillOnly: true,
                sigmaR: configuration.bilateralRangeSigma
            )
            swap(&sourceDepth, &destinationDepth)
            swap(&sourceValid, &destinationValid)
            if filled == 0 { break }
        }

        let raw = sourceDepth.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: raw, count: count))
    }

    private func initializeValidity(depthBuffer: MTLBuffer, validBuffer: MTLBuffer, count: Int) throws {
        var pixelCount = UInt32(count)
        try encode(
            pipeline: initializeValidityPipeline,
            bufferSetup: { encoder in
                encoder.setBuffer(depthBuffer, offset: 0, index: 0)
                encoder.setBuffer(validBuffer, offset: 0, index: 1)
                encoder.setBytes(&pixelCount, length: MemoryLayout<UInt32>.stride, index: 2)
            },
            width: count,
            height: 1
        )
    }

    private func preprocessPass(sourceDepth: MTLBuffer,
                                destinationDepth: MTLBuffer,
                                sourceValid: MTLBuffer,
                                destinationValid: MTLBuffer,
                                mask: MTLBuffer,
                                spatial: MTLBuffer,
                                width: Int,
                                height: Int,
                                hasMask: Bool,
                                fillOnly: Bool,
                                sigmaR: Float) throws -> UInt32 {
        var params = DepthPreprocessParams(
            width: UInt32(width),
            height: UInt32(height),
            hasMask: hasMask ? 1 : 0,
            fillOnly: fillOnly ? 1 : 0,
            sigmaR: sigmaR
        )
        var filled = UInt32(0)
        let filledBuffer = try context.makeBuffer(resource: "depth fill counter") {
            context.device.makeBuffer(bytes: &filled, length: MemoryLayout<UInt32>.stride, options: [])
        }
        try encode(
            pipeline: preprocessPassPipeline,
            bufferSetup: { encoder in
                encoder.setBuffer(sourceDepth, offset: 0, index: 0)
                encoder.setBuffer(destinationDepth, offset: 0, index: 1)
                encoder.setBuffer(sourceValid, offset: 0, index: 2)
                encoder.setBuffer(destinationValid, offset: 0, index: 3)
                encoder.setBuffer(mask, offset: 0, index: 4)
                encoder.setBuffer(spatial, offset: 0, index: 5)
                encoder.setBytes(&params, length: MemoryLayout<DepthPreprocessParams>.stride, index: 6)
                encoder.setBuffer(filledBuffer, offset: 0, index: 7)
            },
            width: width,
            height: height
        )
        return filledBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee
    }

    private func encode(pipeline: MTLComputePipelineState,
                        bufferSetup: (MTLComputeCommandEncoder) -> Void,
                        width: Int,
                        height: Int) throws {
        let commandBuffer = try context.makeCommandBuffer()
        let encoder = try context.makeEncoder(commandBuffer: commandBuffer, stage: "depth preprocessing")
        encoder.setComputePipelineState(pipeline)
        bufferSetup(encoder)
        let threadsPerThreadgroup = MTLSize(
            width: min(pipeline.maxTotalThreadsPerThreadgroup, 16),
            height: 1,
            depth: 1
        )
        let threads = MTLSize(width: max(width, 1), height: max(height, 1), depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        try context.waitForCompletion(commandBuffer, stage: "depth preprocessing")
    }
}
