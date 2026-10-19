import Foundation
import Metal

enum MeasuredCylindricalGridProcessor {
    struct DiffusedGrid {
        var grid: [Float]
        var fillDepth: [Int]
    }

    static func diffusedFill(_ grid: [Float], columns: Int, rows: Int, maxIterations: Int) throws -> DiffusedGrid {
        guard maxIterations > 0 else {
            return DiffusedGrid(
                grid: grid,
                fillDepth: grid.map { $0.isFinite ? 0 : -1 }
            )
        }
        precondition(grid.count == columns * rows)
        return try MeasuredCylindricalGridProcessorMetal.instance().diffusedFill(
            grid,
            columns: columns,
            rows: rows,
            maxIterations: maxIterations
        )
    }

    static func smoothedGrid(
        _ grid: [Float],
        columns: Int,
        rows: Int,
        iterations: Int,
        radius: Int,
        sigmaS: Float,
        sigmaR: Float
    ) throws -> [Float] {
        guard iterations > 0 else { return grid }
        precondition(grid.count == columns * rows)
        return try MeasuredCylindricalGridProcessorMetal.instance().smoothedGrid(
            grid,
            columns: columns,
            rows: rows,
            iterations: iterations,
            radius: radius,
            sigmaS: sigmaS,
            sigmaR: sigmaR
        )
    }

    static func openedMask2D(_ mask: [Bool], columns: Int, rows: Int, iterations: Int) throws -> [Bool] {
        guard iterations > 0 else { return mask }
        precondition(mask.count == columns * rows)
        return try MeasuredCylindricalGridProcessorMetal.instance().openedMask2D(
            mask,
            columns: columns,
            rows: rows,
            iterations: iterations
        )
    }
}

private final class MeasuredCylindricalGridProcessorMetal {
    private static let cache = MeasuredMetalEngineCache<MeasuredCylindricalGridProcessorMetal>()

    static func instance() throws -> MeasuredCylindricalGridProcessorMetal {
        try cache.get { try MeasuredCylindricalGridProcessorMetal() }
    }

    private struct GridSmoothParams {
        var columns: UInt32
        var rows: UInt32
        var radius: UInt32
        var sigmaS: Float
        var sigmaR: Float
    }

    private struct MaskParams {
        var columns: UInt32
        var rows: UInt32
    }

    private struct DiffusionParams {
        var columns: UInt32
        var rows: UInt32
        var iteration: Int32
    }

    private let context: MeasuredMetalContext
    private let fillInitializePipeline: MTLComputePipelineState
    private let fillPassPipeline: MTLComputePipelineState
    private let smoothPipeline: MTLComputePipelineState
    private let erodePipeline: MTLComputePipelineState
    private let dilatePipeline: MTLComputePipelineState
    private let intersectPipeline: MTLComputePipelineState

    private init() throws {
        let context = try MeasuredMetalContext(operation: "measured cylindrical grid processing")
        self.context = context
        self.fillInitializePipeline = try context.makePipeline(named: "tmf_initializeCylindricalFill")
        self.fillPassPipeline = try context.makePipeline(named: "tmf_diffuseCylindricalGrid")
        self.smoothPipeline = try context.makePipeline(named: "tmf_smoothCylindricalGrid")
        self.erodePipeline = try context.makePipeline(named: "tmf_erodeCrossMask")
        self.dilatePipeline = try context.makePipeline(named: "tmf_dilateCrossMask")
        self.intersectPipeline = try context.makePipeline(named: "tmf_intersectMasks")
    }

    func diffusedFill(_ grid: [Float], columns: Int, rows: Int, maxIterations: Int) throws -> MeasuredCylindricalGridProcessor.DiffusedGrid {
        guard !grid.isEmpty else {
            return MeasuredCylindricalGridProcessor.DiffusedGrid(grid: [], fillDepth: [])
        }
        var source = grid
        let gridA = try context.makeBuffer(resource: "cylindrical diffusion source") {
            context.device.makeBuffer(bytes: &source, length: source.count * MemoryLayout<Float>.stride, options: [])
        }
        let gridB = try context.makeBuffer(resource: "cylindrical diffusion destination") {
            context.device.makeBuffer(length: source.count * MemoryLayout<Float>.stride, options: [])
        }
        let depthA = try context.makeBuffer(resource: "cylindrical fill depth source") {
            context.device.makeBuffer(length: source.count * MemoryLayout<Int32>.stride, options: [])
        }
        let depthB = try context.makeBuffer(resource: "cylindrical fill depth destination") {
            context.device.makeBuffer(length: source.count * MemoryLayout<Int32>.stride, options: [])
        }

        var currentGrid: MTLBuffer = gridA
        var nextGrid: MTLBuffer = gridB
        var currentDepth: MTLBuffer = depthA
        var nextDepth: MTLBuffer = depthB
        var params = DiffusionParams(columns: UInt32(columns), rows: UInt32(rows), iteration: 0)

        let initCommandBuffer = try context.makeCommandBuffer()
        let initEncoder = try context.makeEncoder(commandBuffer: initCommandBuffer, stage: "cylindrical fill initialization")
        initEncoder.setComputePipelineState(fillInitializePipeline)
        initEncoder.setBuffer(currentGrid, offset: 0, index: 0)
        initEncoder.setBuffer(currentDepth, offset: 0, index: 1)
        initEncoder.setBytes(&params, length: MemoryLayout<DiffusionParams>.stride, index: 2)
        dispatch2D(initEncoder, pipeline: fillInitializePipeline, columns: columns, rows: rows)
        initEncoder.endEncoding()
        try context.waitForCompletion(initCommandBuffer, stage: "cylindrical fill initialization")

        for iteration in 1...maxIterations {
            var filledCount: UInt32 = 0
            params.iteration = Int32(iteration)
            let countBuffer = try context.makeBuffer(resource: "cylindrical diffusion fill count") {
                context.device.makeBuffer(bytes: &filledCount, length: MemoryLayout<UInt32>.stride, options: [])
            }
            let commandBuffer = try context.makeCommandBuffer()
            let encoder = try context.makeEncoder(commandBuffer: commandBuffer, stage: "cylindrical diffusion")
            encoder.setComputePipelineState(fillPassPipeline)
            encoder.setBuffer(currentGrid, offset: 0, index: 0)
            encoder.setBuffer(currentDepth, offset: 0, index: 1)
            encoder.setBuffer(nextGrid, offset: 0, index: 2)
            encoder.setBuffer(nextDepth, offset: 0, index: 3)
            encoder.setBuffer(countBuffer, offset: 0, index: 4)
            encoder.setBytes(&params, length: MemoryLayout<DiffusionParams>.stride, index: 5)
            dispatch2D(encoder, pipeline: fillPassPipeline, columns: columns, rows: rows)
            encoder.endEncoding()
            try context.waitForCompletion(commandBuffer, stage: "cylindrical diffusion")

            let filled = countBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)[0]
            swap(&currentGrid, &nextGrid)
            swap(&currentDepth, &nextDepth)
            if filled == 0 {
                break
            }
        }

        let valuePointer = currentGrid.contents().bindMemory(to: Float.self, capacity: grid.count)
        let depthPointer = currentDepth.contents().bindMemory(to: Int32.self, capacity: grid.count)
        return MeasuredCylindricalGridProcessor.DiffusedGrid(
            grid: Array(UnsafeBufferPointer(start: valuePointer, count: grid.count)),
            fillDepth: (0..<grid.count).map { Int(depthPointer[$0]) }
        )
    }

    func smoothedGrid(
        _ grid: [Float],
        columns: Int,
        rows: Int,
        iterations: Int,
        radius: Int,
        sigmaS: Float,
        sigmaR: Float
    ) throws -> [Float] {
        guard !grid.isEmpty else { return [] }
        var source = grid
        var params = GridSmoothParams(
            columns: UInt32(columns),
            rows: UInt32(rows),
            radius: UInt32(max(0, radius)),
            sigmaS: sigmaS,
            sigmaR: sigmaR
        )

        let sourceBuffer = try context.makeBuffer(resource: "cylindrical smoothing source") {
            context.device.makeBuffer(bytes: &source, length: source.count * MemoryLayout<Float>.stride, options: [])
        }
        let scratchBuffer = try context.makeBuffer(resource: "cylindrical smoothing scratch") {
            context.device.makeBuffer(length: source.count * MemoryLayout<Float>.stride, options: [])
        }
        let commandBuffer = try context.makeCommandBuffer()

        var currentBuffer: MTLBuffer = sourceBuffer
        var nextBuffer: MTLBuffer = scratchBuffer
        for _ in 0..<iterations {
            let encoder = try context.makeEncoder(commandBuffer: commandBuffer, stage: "cylindrical smoothing")
            encoder.setComputePipelineState(smoothPipeline)
            encoder.setBuffer(currentBuffer, offset: 0, index: 0)
            encoder.setBuffer(nextBuffer, offset: 0, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<GridSmoothParams>.stride, index: 2)
            dispatch2D(
                encoder,
                pipeline: smoothPipeline,
                columns: columns,
                rows: rows
            )
            encoder.endEncoding()
            swap(&currentBuffer, &nextBuffer)
        }
        try context.waitForCompletion(commandBuffer, stage: "cylindrical smoothing")

        let values = currentBuffer.contents().bindMemory(to: Float.self, capacity: grid.count)
        return Array(UnsafeBufferPointer(start: values, count: grid.count))
    }

    func openedMask2D(_ mask: [Bool], columns: Int, rows: Int, iterations: Int) throws -> [Bool] {
        guard !mask.isEmpty else { return [] }
        var original = mask.map { UInt8($0 ? 1 : 0) }
        var params = MaskParams(columns: UInt32(columns), rows: UInt32(rows))

        let originalBuffer = try context.makeBuffer(resource: "mask opening original") {
            context.device.makeBuffer(bytes: &original, length: original.count, options: [])
        }
        let scratchA = try context.makeBuffer(resource: "mask opening scratch A") {
            context.device.makeBuffer(bytes: &original, length: original.count, options: [])
        }
        let scratchB = try context.makeBuffer(resource: "mask opening scratch B") {
            context.device.makeBuffer(length: original.count, options: [])
        }
        let commandBuffer = try context.makeCommandBuffer()

        var currentBuffer: MTLBuffer = scratchA
        var nextBuffer: MTLBuffer = scratchB
        for _ in 0..<iterations {
            try encodeMaskPass(
                commandBuffer: commandBuffer,
                pipeline: erodePipeline,
                input: currentBuffer,
                output: nextBuffer,
                params: &params,
                columns: columns,
                rows: rows
            )
            swap(&currentBuffer, &nextBuffer)
        }
        for _ in 0..<iterations {
            try encodeMaskPass(
                commandBuffer: commandBuffer,
                pipeline: dilatePipeline,
                input: currentBuffer,
                output: nextBuffer,
                params: &params,
                columns: columns,
                rows: rows
            )
            swap(&currentBuffer, &nextBuffer)
        }
        let encoder = try context.makeEncoder(commandBuffer: commandBuffer, stage: "mask intersection")
        encoder.setComputePipelineState(intersectPipeline)
        encoder.setBuffer(currentBuffer, offset: 0, index: 0)
        encoder.setBuffer(originalBuffer, offset: 0, index: 1)
        encoder.setBuffer(nextBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<MaskParams>.stride, index: 3)
        dispatch2D(encoder, pipeline: intersectPipeline, columns: columns, rows: rows)
        encoder.endEncoding()
        currentBuffer = nextBuffer

        try context.waitForCompletion(commandBuffer, stage: "mask opening")

        let values = currentBuffer.contents().bindMemory(to: UInt8.self, capacity: mask.count)
        return (0..<mask.count).map { values[$0] != 0 }
    }

    private func encodeMaskPass(
        commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        input: MTLBuffer,
        output: MTLBuffer,
        params: inout MaskParams,
        columns: Int,
        rows: Int
    ) throws {
        let encoder = try context.makeEncoder(commandBuffer: commandBuffer, stage: "mask morphology")
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<MaskParams>.stride, index: 2)
        dispatch2D(encoder, pipeline: pipeline, columns: columns, rows: rows)
        encoder.endEncoding()
    }

    private func dispatch2D(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        columns: Int,
        rows: Int
    ) {
        let width = min(16, pipeline.threadExecutionWidth)
        let height = max(1, min(16, pipeline.maxTotalThreadsPerThreadgroup / max(1, width)))
        encoder.dispatchThreads(
            MTLSize(width: columns, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1)
        )
    }
}
