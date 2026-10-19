import Foundation
import XCTest
import simd
@testable import TrueDepthCapture

final class CaptureMotionGateTests: XCTestCase {
    private func transform(x: Float = 0, angle: Float = 0) -> simd_float4x4 {
        var result = simd_float4x4(simd_quatf(angle: angle, axis: SIMD3<Float>(0, 1, 0)))
        result[3] = SIMD4(x, 0, -0.3, 1)
        return result
    }

    func testRequiresObservedMotionBeforeAcceptingEvenAlignedDepth() {
        var gate = CaptureMotionGate(); let face = UUID()
        gate.observe(cameraFromFace: transform(), timestamp: 1, trackingID: face)
        XCTAssertFalse(gate.accepts(imageTime: 1, depthTime: 1))
        gate.observe(cameraFromFace: transform(), timestamp: 1.05, trackingID: face)
        XCTAssertTrue(gate.accepts(imageTime: 1.05, depthTime: 1.05))
    }

    func testOriginalBaseLinearVelocityRejectsFastRelativeMotion() {
        for displacement: Float in [0.004, 0.0055] {
            var gate = CaptureMotionGate(); let face = UUID()
            gate.observe(cameraFromFace: transform(), timestamp: 1, trackingID: face)
            gate.observe(cameraFromFace: transform(x: displacement), timestamp: 1.1, trackingID: face)
            XCTAssertEqual(gate.accepts(imageTime: 1.1, depthTime: 1.08), displacement == 0.004)
        }
    }

    func testOriginalBaseAngularVelocityRejectsFastTurns() {
        for angle: Float in [0.05, 0.09] {
            var gate = CaptureMotionGate(); let face = UUID()
            gate.observe(cameraFromFace: transform(), timestamp: 1, trackingID: face)
            gate.observe(cameraFromFace: transform(angle: angle), timestamp: 1.1, trackingID: face)
            XCTAssertEqual(gate.accepts(imageTime: 1.1, depthTime: 1.08), angle == 0.05)
        }
    }

    func testDelayedDepthChecksEarlierMotionAfterHeadStops() {
        var gate = CaptureMotionGate(); let face = UUID()
        gate.observe(cameraFromFace: transform(), timestamp: 1, trackingID: face)
        gate.observe(cameraFromFace: transform(x: 0.01), timestamp: 1.05, trackingID: face)
        gate.observe(cameraFromFace: transform(x: 0.01), timestamp: 1.10, trackingID: face)
        gate.observe(cameraFromFace: transform(x: 0.01), timestamp: 1.15, trackingID: face)
        XCTAssertFalse(gate.accepts(imageTime: 1.15, depthTime: 1.04))
        gate.observe(cameraFromFace: transform(x: 0.01), timestamp: 1.20, trackingID: face)
        XCTAssertTrue(gate.accepts(imageTime: 1.20, depthTime: 1.12))
    }

    func testMissingTrackingAndChangedIdentityResetMotionEvidence() {
        for mode in 0..<2 {
            var gate = CaptureMotionGate(); let first = UUID(), second = UUID()
            gate.observe(cameraFromFace: transform(), timestamp: 1, trackingID: first)
            gate.observe(cameraFromFace: transform(), timestamp: 1.05, trackingID: first)
            if mode == 0 { gate.observe(cameraFromFace: nil, timestamp: 1.06, trackingID: nil) }
            let identity = mode == 0 ? first : second
            gate.observe(cameraFromFace: transform(), timestamp: 1.1, trackingID: identity)
            XCTAssertFalse(gate.accepts(imageTime: 1.1, depthTime: 1.08))
            gate.observe(cameraFromFace: transform(), timestamp: 1.15, trackingID: identity)
            XCTAssertTrue(gate.accepts(imageTime: 1.15, depthTime: 1.12))
        }
    }

    func testUnknownFutureDepthAndUnobservedPoseGapsAreRejected() {
        var gate = CaptureMotionGate(); let face = UUID()
        gate.observe(cameraFromFace: transform(), timestamp: 1, trackingID: face)
        gate.observe(cameraFromFace: transform(), timestamp: 1.05, trackingID: face)
        for depth in [0, .nan, 1.06, 0.95] {
            XCTAssertFalse(gate.accepts(imageTime: 1.05, depthTime: depth))
        }
        gate.observe(cameraFromFace: transform(), timestamp: 1.5, trackingID: face)
        XCTAssertFalse(gate.accepts(imageTime: 1.5, depthTime: 1.48))
        gate.observe(cameraFromFace: transform(), timestamp: 1.55, trackingID: face)
        XCTAssertTrue(gate.accepts(imageTime: 1.55, depthTime: 1.52))
    }

    func testNonfiniteAndOutOfOrderTrackingCannotSupplyStableEvidence() {
        var gate = CaptureMotionGate(); let face = UUID()
        gate.observe(cameraFromFace: transform(), timestamp: 1, trackingID: face)
        gate.observe(cameraFromFace: transform(), timestamp: 1.05, trackingID: face)
        gate.observe(cameraFromFace: transform(x: .nan), timestamp: 1.1, trackingID: face)
        XCTAssertFalse(gate.accepts(imageTime: 1.1, depthTime: 1.08))
        gate.observe(cameraFromFace: transform(), timestamp: 1.15, trackingID: face)
        gate.observe(cameraFromFace: transform(), timestamp: 1.14, trackingID: face)
        XCTAssertFalse(gate.accepts(imageTime: 1.14, depthTime: 1.14))
    }

    func testIngressObservesDroppedFrameMotionAndIgnoresOldSessions() throws {
        let ingress = CaptureFrameIngress(), session = NSObject(), oldSession = NSObject()
        let token = UUID(), face = UUID(), id = ObjectIdentifier(session)
        ingress.setEpoch(token, sessionIdentifier: id)
        let first = try XCTUnwrap(ingress.acquire(sessionIdentifier: id, cameraFromFace: transform(),
            trackingID: face, imageTime: 1, depthTime: 1))
        XCTAssertFalse(first.motionAccepted)
        XCTAssertNil(ingress.acquire(sessionIdentifier: id, cameraFromFace: transform(x: 0.01),
            trackingID: face, imageTime: 1.05, depthTime: 1.04))
        XCTAssertNil(ingress.acquire(sessionIdentifier: id, cameraFromFace: transform(x: 0.01),
            trackingID: face, imageTime: 1.1, depthTime: 1.08))
        ingress.release()
        let delayed = try XCTUnwrap(ingress.acquire(sessionIdentifier: id, cameraFromFace: transform(x: 0.01),
            trackingID: face, imageTime: 1.15, depthTime: 1.04))
        XCTAssertEqual(delayed.token, token); XCTAssertFalse(delayed.motionAccepted)
        ingress.release()
        XCTAssertNil(ingress.acquire(sessionIdentifier: ObjectIdentifier(oldSession), cameraFromFace: transform(x: 10),
            trackingID: face, imageTime: 1.16, depthTime: 1.16))
        let stable = try XCTUnwrap(ingress.acquire(sessionIdentifier: id, cameraFromFace: transform(x: 0.01),
            trackingID: face, imageTime: 1.2, depthTime: 1.12))
        XCTAssertTrue(stable.motionAccepted)
        ingress.release()
        ingress.setEpoch(UUID(), sessionIdentifier: id)
        let reset = try XCTUnwrap(ingress.acquire(sessionIdentifier: id, cameraFromFace: transform(x: 0.01),
            trackingID: face, imageTime: 1.25, depthTime: 1.2))
        XCTAssertFalse(reset.motionAccepted)
        ingress.release()
    }
}
