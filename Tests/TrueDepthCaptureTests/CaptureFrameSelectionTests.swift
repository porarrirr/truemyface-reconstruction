import Foundation
import XCTest
import simd
import TrueDepthReconstruction
@testable import TrueDepthCapture

final class CaptureFrameSelectionTests: XCTestCase {
    private func frame(_ time: Double, yaw: Double = 0, pitch: Double = 0) -> FaceDepthKeyframe {
        FaceDepthKeyframe(timestamp: time, pose: FaceScanPose(yawDegrees: yaw, pitchDegrees: pitch),
                          depthValues: [0.3, 0.3, 0.3, 0.3], width: 2, height: 2,
                          intrinsics: CameraIntrinsics(fx: 20, fy: 20, cx: 0.5, cy: 0.5),
                          cameraFromFace: matrix_identity_float4x4, faceMask: [true, true, true, true])
    }

    func testBalancedSelectionIsBoundedAndKeepsAllDirections() {
        var selection = CaptureFrameSelection()
        for i in 0..<100 { selection.append(frame(Double(i))) }
        XCTAssertEqual(selection.frames.count, 2)
        XCTAssertFalse(selection.canReconstruct)
        let poses: [(Double, Double)] = [(-15, 0), (15, 0), (0, -15), (0, 15)]
        for (index, pose) in poses.enumerated() {
            for t in 0..<3 { selection.append(frame(100 + Double(index * 3 + t), yaw: pose.0, pitch: pose.1)) }
        }
        XCTAssertEqual(selection.frames.count, 10)
        XCTAssertTrue(selection.isComplete)
        XCTAssertTrue(selection.canReconstruct)
        XCTAssertEqual(selection.counts.count, 5)
        XCTAssertTrue(selection.counts.values.allSatisfy { $0 == 2 })
        XCTAssertEqual(selection.frames.map(\.timestamp), selection.frames.map(\.timestamp).sorted())
    }

    func testCoverageRequiresFrontAndAtLeastThreeDirections() {
        var selection = CaptureFrameSelection()
        for i in 0..<2 {
            selection.append(frame(Double(i), yaw: -15))
            selection.append(frame(Double(i + 2), yaw: 15))
            selection.append(frame(Double(i + 4), pitch: 15))
        }
        XCTAssertFalse(selection.canReconstruct)
        selection.append(frame(10))
        XCTAssertTrue(selection.canReconstruct)
        XCTAssertFalse(selection.isComplete)
    }

    func testExtremeAndNonfinitePosesAreRejected() {
        for pose in [FaceScanPose(yawDegrees: 60), FaceScanPose(pitchDegrees: -60), FaceScanPose(yawDegrees: .nan)] {
            XCTAssertNil(CaptureViewpoint.classify(pose))
        }
    }

    func testDepthTimingRejectsMissingStaleDuplicateAndTooFrequentSamples() {
        XCTAssertTrue(CaptureFrameTiming.accepts(imageTime: 10, depthTime: 10.05, previousDepthTime: 9.8, previousAcceptedTime: 9.5))
        for depthTime in [0, .nan, 9.5, 10.3] {
            XCTAssertFalse(CaptureFrameTiming.accepts(imageTime: 10, depthTime: depthTime, previousDepthTime: nil, previousAcceptedTime: nil))
        }
        XCTAssertFalse(CaptureFrameTiming.accepts(imageTime: 10, depthTime: 10, previousDepthTime: 10, previousAcceptedTime: nil))
        XCTAssertFalse(CaptureFrameTiming.accepts(imageTime: 10, depthTime: 10, previousDepthTime: 9.9, previousAcceptedTime: 9.9))
    }

    func testIngressBoundsRetainedFramesAndInvalidatesOldEpochs() {
        let ingress = CaptureFrameIngress()
        XCTAssertNil(ingress.acquire())
        let first = UUID(), second = UUID()
        ingress.setEpoch(first)
        XCTAssertEqual(ingress.acquire(), first)
        DispatchQueue.concurrentPerform(iterations: 64) { _ in XCTAssertNil(ingress.acquire()) }
        ingress.setEpoch(second)
        XCTAssertNil(ingress.acquire())
        ingress.release()
        XCTAssertEqual(ingress.acquire(), second)
        ingress.setEpoch(nil); ingress.release()
        XCTAssertNil(ingress.acquire())
    }

    func testPortraitPoseCorrectsSensorRotationWithoutChangingSurfaceCoordinates() {
        let portrait = simd_float4x4(simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(0, 0, 1)))
        for angle: Float in [-15, 0, 15] {
            var transform = portrait * simd_float4x4(simd_quatf(angle: angle * .pi / 180, axis: SIMD3<Float>(0, 1, 0)))
            transform[3] = SIMD4(0, 0, -0.3, 1)
            let pose = CapturePose.portrait(transform)
            XCTAssertEqual(pose.yawDegrees, -Double(angle), accuracy: 0.001)
            XCTAssertEqual(pose.pitchDegrees, 0, accuracy: 0.001)
        }
    }
    func testPortraitPitchClassifiesLookingUpAndDownLikeOriginalCapture() {
        let portrait = simd_float4x4(simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(0, 0, 1)))
        for angle: Float in [-15, 15] {
            var transform = portrait * simd_float4x4(simd_quatf(angle: angle * .pi / 180, axis: SIMD3<Float>(1, 0, 0)))
            transform[3] = SIMD4(0, 0, -0.3, 1)
            let pose = CapturePose.portrait(transform)
            XCTAssertEqual(pose.pitchDegrees, -Double(angle), accuracy: 0.001)
            XCTAssertEqual(CaptureViewpoint.classify(pose), angle < 0 ? .up : .down)
            var selection = CaptureFrameSelection()
            selection.append(frame(1, yaw: pose.yawDegrees, pitch: pose.pitchDegrees))
            XCTAssertEqual(selection.counts[angle < 0 ? .up : .down], 1)
        }
    }

}
