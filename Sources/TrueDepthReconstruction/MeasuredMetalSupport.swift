import Foundation
import Metal

enum MeasuredReconstructionError: Error, Equatable {
    case metalDeviceUnavailable(operation: String)
    case metalCommandQueueCreationFailed(operation: String)
    case metalLibraryLoadFailed(operation: String)
    case metalKernelMissing(operation: String, kernel: String)
    case metalPipelineCreationFailed(operation: String, kernel: String, detail: String)
    case metalBufferAllocationFailed(operation: String, resource: String)
    case metalCommandBufferCreationFailed(operation: String)
    case metalEncoderCreationFailed(operation: String, stage: String)
    case metalExecutionFailed(operation: String, stage: String, detail: String)

    var diagnosticDescription: String {
        switch self {
        case .metalDeviceUnavailable(let operation):
            return "Metal device unavailable during \(operation)"
        case .metalCommandQueueCreationFailed(let operation):
            return "Metal command queue creation failed during \(operation)"
        case .metalLibraryLoadFailed(let operation):
            return "Metal default library load failed during \(operation)"
        case .metalKernelMissing(let operation, let kernel):
            return "Metal kernel \(kernel) is missing during \(operation)"
        case .metalPipelineCreationFailed(let operation, let kernel, let detail):
            return "Metal pipeline \(kernel) creation failed during \(operation): \(detail)"
        case .metalBufferAllocationFailed(let operation, let resource):
            return "Metal buffer allocation failed for \(resource) during \(operation)"
        case .metalCommandBufferCreationFailed(let operation):
            return "Metal command buffer creation failed during \(operation)"
        case .metalEncoderCreationFailed(let operation, let stage):
            return "Metal encoder creation failed for \(stage) during \(operation)"
        case .metalExecutionFailed(let operation, let stage, let detail):
            return "Metal execution failed for \(stage) during \(operation): \(detail)"
        }
    }
}

extension MeasuredReconstructionError: LocalizedError {
    var errorDescription: String? { diagnosticDescription }
}

/// Measured 再構築の Metal リソース生成と実行エラーを一つの契約にまとめる。
/// `failureInjector` は unit test で各障害点を決定的に再現するためだけに使用する。
struct MeasuredMetalContext {
    enum FailurePoint: Equatable {
        case device
        case commandQueue
        case library
        case pipeline(kernel: String)
        case buffer(resource: String)
        case commandBuffer
        case encoder(stage: String)
        case execution(stage: String)
    }

    typealias FailureInjector = (FailurePoint) -> Bool

    let operation: String
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary
    private let failureInjector: FailureInjector

    init(
        operation: String,
        deviceProvider: () -> MTLDevice? = { MTLCreateSystemDefaultDevice() },
        failureInjector: @escaping FailureInjector = { _ in false }
    ) throws {
        self.operation = operation
        self.failureInjector = failureInjector
        guard !failureInjector(.device), let device = deviceProvider() else {
            throw MeasuredReconstructionError.metalDeviceUnavailable(operation: operation)
        }
        guard !failureInjector(.commandQueue), let commandQueue = device.makeCommandQueue() else {
            throw MeasuredReconstructionError.metalCommandQueueCreationFailed(operation: operation)
        }
        guard !failureInjector(.library),
              let sourceURL = Bundle.module.url(forResource: "MeasuredDepthPreprocessor", withExtension: "metal"),
              let source = try? String(contentsOf: sourceURL, encoding: .utf8),
              let library = try? device.makeLibrary(source: source, options: nil) else {
            throw MeasuredReconstructionError.metalLibraryLoadFailed(operation: operation)
        }
        self.device = device
        self.commandQueue = commandQueue
        self.library = library
    }

    func makePipeline(named name: String) throws -> MTLComputePipelineState {
        guard !failureInjector(.pipeline(kernel: name)) else {
            throw MeasuredReconstructionError.metalPipelineCreationFailed(
                operation: operation,
                kernel: name,
                detail: "injected failure"
            )
        }
        guard let function = library.makeFunction(name: name) else {
            throw MeasuredReconstructionError.metalKernelMissing(operation: operation, kernel: name)
        }
        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            throw MeasuredReconstructionError.metalPipelineCreationFailed(
                operation: operation,
                kernel: name,
                detail: error.localizedDescription
            )
        }
    }

    func makeBuffer(resource: String, _ make: () -> MTLBuffer?) throws -> MTLBuffer {
        guard !failureInjector(.buffer(resource: resource)), let buffer = make() else {
            throw MeasuredReconstructionError.metalBufferAllocationFailed(
                operation: operation,
                resource: resource
            )
        }
        return buffer
    }

    func makeCommandBuffer() throws -> MTLCommandBuffer {
        guard !failureInjector(.commandBuffer), let buffer = commandQueue.makeCommandBuffer() else {
            throw MeasuredReconstructionError.metalCommandBufferCreationFailed(operation: operation)
        }
        return buffer
    }

    func makeEncoder(commandBuffer: MTLCommandBuffer, stage: String) throws -> MTLComputeCommandEncoder {
        guard !failureInjector(.encoder(stage: stage)),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MeasuredReconstructionError.metalEncoderCreationFailed(operation: operation, stage: stage)
        }
        return encoder
    }

    func waitForCompletion(_ commandBuffer: MTLCommandBuffer, stage: String) throws {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if failureInjector(.execution(stage: stage)) {
            throw MeasuredReconstructionError.metalExecutionFailed(
                operation: operation,
                stage: stage,
                detail: "injected failure"
            )
        }
        if let error = commandBuffer.error {
            throw MeasuredReconstructionError.metalExecutionFailed(
                operation: operation,
                stage: stage,
                detail: error.localizedDescription
            )
        }
    }
}

/// 初期化失敗は保持せず、次の再撮影で初期化を再試行できる成功時のみのキャッシュ。
final class MeasuredMetalEngineCache<Engine> {
    private let lock = NSLock()
    private var cached: Engine?

    func get(create: () throws -> Engine) throws -> Engine {
        lock.lock()
        defer { lock.unlock() }
        if let cached { return cached }
        let engine = try create()
        cached = engine
        return engine
    }
}
