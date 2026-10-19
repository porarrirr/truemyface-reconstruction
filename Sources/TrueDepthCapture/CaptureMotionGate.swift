import Foundation
import simd

/// Local motion measurement extracted from the application's capture-readiness calculation.
/// Use its conservative base limits, without app-mode/adaptive-window dependencies.
struct CaptureMotionGate {
    static let maximumLinearVelocity: Float = 0.045 // meters / second
    static let maximumAngularVelocity: Float = 0.80 // radians / second
    private static let maximumSampleGap: TimeInterval = 0.12

    private struct Sample {
        var transform: simd_float4x4
        var timestamp: TimeInterval
        var trackingID: UUID
    }
    private struct Segment {
        var start: TimeInterval
        var end: TimeInterval
        var linearVelocity: Float
        var angularVelocity: Float
        var isStable: Bool {
            linearVelocity.isFinite && angularVelocity.isFinite
                && linearVelocity <= CaptureMotionGate.maximumLinearVelocity
                && angularVelocity <= CaptureMotionGate.maximumAngularVelocity
        }
    }
    private var previous: Sample?
    private var segments: [Segment] = []

    mutating func observe(cameraFromFace: simd_float4x4?, timestamp: TimeInterval, trackingID: UUID?) {
        guard let transform = cameraFromFace, let trackingID,
              timestamp.isFinite, timestamp > 0,
              (0..<4).allSatisfy({ column in (0..<4).allSatisfy { transform[column][$0].isFinite } }),
              abs(simd_determinant(transform) - 1) < 0.001 else {
            self = Self(); return
        }
        let current = Sample(transform: transform, timestamp: timestamp, trackingID: trackingID)
        defer { self.previous = current }
        guard let previous, previous.trackingID == trackingID,
              timestamp > previous.timestamp,
              timestamp - previous.timestamp <= Self.maximumSampleGap else {
            segments.removeAll(keepingCapacity: true); return
        }
        // Same quaternion-delta/translation-distance calculation as the original readiness gate.
        let dt = max(1e-3, Float(timestamp - previous.timestamp))
        let delta = simd_quatf(transform) * simd_quatf(previous.transform).inverse
        let angularVelocity = 2 * acosf(min(1, abs(delta.real))) / dt
        let before = SIMD3(previous.transform[3].x, previous.transform[3].y, previous.transform[3].z)
        let after = SIMD3(transform[3].x, transform[3].y, transform[3].z)
        segments.append(Segment(start: previous.timestamp, end: timestamp,
                                linearVelocity: simd_distance(before, after) / dt,
                                angularVelocity: angularVelocity))
        segments.removeAll { $0.end < timestamp - 0.5 }
        if segments.count > 64 { segments.removeFirst(segments.count - 64) }
    }

    func accepts(imageTime: TimeInterval, depthTime: TimeInterval) -> Bool {
        guard imageTime.isFinite, depthTime.isFinite, depthTime > 0,
              depthTime <= imageTime, imageTime - depthTime <= 0.12,
              previous?.timestamp == imageTime, let latest = segments.last else { return false }
        // Cover the full RGB/depth interval AND at least the latest observed motion segment.
        // A newly stationary pose must not hide motion that occurred after an older depth frame.
        let start = min(depthTime, latest.start)
        guard let first = segments.first, first.start <= start else { return false }
        return segments.lazy.filter { $0.end > start }.allSatisfy(\.isStable)
    }
}
