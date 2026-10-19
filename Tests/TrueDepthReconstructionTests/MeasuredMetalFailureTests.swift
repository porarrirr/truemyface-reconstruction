import Metal
import XCTest
@testable import TrueDepthReconstruction

final class MeasuredMetalFailureTests: XCTestCase {
    func testEngineCacheRetriesAfterFailureAndCachesOnlySuccess() throws {
        let cache = MeasuredMetalEngineCache<NSObject>()
        var attempts = 0

        XCTAssertThrowsError(try cache.get {
            attempts += 1
            throw MeasuredReconstructionError.metalDeviceUnavailable(operation: "test")
        })

        let expected = try cache.get {
            attempts += 1
            return NSObject()
        }
        let cached = try cache.get {
            attempts += 1
            return NSObject()
        }

        XCTAssertTrue(expected === cached)
        XCTAssertEqual(attempts, 2)
    }

    func testInitializationFailuresAreTypedAndDoNotCreateAnEngine() throws {
        XCTAssertThrowsError(
            try MeasuredMetalContext(operation: "test", deviceProvider: { nil })
        ) { error in
            XCTAssertEqual(
                error as? MeasuredReconstructionError,
                .metalDeviceUnavailable(operation: "test")
            )
        }

        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal is unavailable on this test destination")
        }
        XCTAssertThrowsError(
            try MeasuredMetalContext(operation: "test", failureInjector: { point in point == .commandQueue })
        ) { error in
            XCTAssertEqual(
                error as? MeasuredReconstructionError,
                .metalCommandQueueCreationFailed(operation: "test")
            )
        }
        XCTAssertThrowsError(
            try MeasuredMetalContext(operation: "test", failureInjector: { point in point == .library })
        ) { error in
            XCTAssertEqual(
                error as? MeasuredReconstructionError,
                .metalLibraryLoadFailed(operation: "test")
            )
        }
    }

    func testPipelineAndBufferFailuresPreserveTheFailingResource() throws {
        let pipelineContext = try MeasuredMetalContext(operation: "test", failureInjector: { point in
            point == .pipeline(kernel: "tmf_erodeMask")
        })
        XCTAssertThrowsError(try pipelineContext.makePipeline(named: "tmf_erodeMask")) { error in
            XCTAssertEqual(
                error as? MeasuredReconstructionError,
                .metalPipelineCreationFailed(
                    operation: "test",
                    kernel: "tmf_erodeMask",
                    detail: "injected failure"
                )
            )
        }

        let bufferContext = try MeasuredMetalContext(operation: "test", failureInjector: { point in
            point == .buffer(resource: "depth input")
        })
        XCTAssertThrowsError(
            try bufferContext.makeBuffer(resource: "depth input") {
                bufferContext.device.makeBuffer(length: 4, options: [])
            }
        ) { error in
            XCTAssertEqual(
                error as? MeasuredReconstructionError,
                .metalBufferAllocationFailed(operation: "test", resource: "depth input")
            )
        }
    }

    func testCommandEncodingAndExecutionFailuresRemainDistinct() throws {
        let commandContext = try MeasuredMetalContext(operation: "test", failureInjector: { point in
            point == .commandBuffer
        })
        XCTAssertThrowsError(try commandContext.makeCommandBuffer()) { error in
            XCTAssertEqual(
                error as? MeasuredReconstructionError,
                .metalCommandBufferCreationFailed(operation: "test")
            )
        }

        let encoderContext = try MeasuredMetalContext(operation: "test", failureInjector: { point in
            point == .encoder(stage: "preprocess")
        })
        let encoderCommandBuffer = try encoderContext.makeCommandBuffer()
        XCTAssertThrowsError(
            try encoderContext.makeEncoder(commandBuffer: encoderCommandBuffer, stage: "preprocess")
        ) { error in
            XCTAssertEqual(
                error as? MeasuredReconstructionError,
                .metalEncoderCreationFailed(operation: "test", stage: "preprocess")
            )
        }

        let executionContext = try MeasuredMetalContext(operation: "test", failureInjector: { point in
            point == .execution(stage: "preprocess")
        })
        let executionCommandBuffer = try executionContext.makeCommandBuffer()
        XCTAssertThrowsError(
            try executionContext.waitForCompletion(executionCommandBuffer, stage: "preprocess")
        ) { error in
            XCTAssertEqual(
                error as? MeasuredReconstructionError,
                .metalExecutionFailed(
                    operation: "test",
                    stage: "preprocess",
                    detail: "injected failure"
                )
            )
        }
    }
}
