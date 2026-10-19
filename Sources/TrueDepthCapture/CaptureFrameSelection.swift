import Foundation
import simd
import TrueDepthReconstruction

public enum CaptureViewpoint: String, CaseIterable {
    case front, left, right, up, down

    static func classify(_ pose: FaceScanPose) -> Self? {
        let yaw = pose.yawDegrees, pitch = pose.pitchDegrees
        guard yaw.isFinite, pitch.isFinite, abs(yaw) <= 30, abs(pitch) <= 25 else { return nil }
        if abs(yaw) <= 7, abs(pitch) <= 7 { return .front }
        if abs(pitch) <= 12, yaw <= -8 { return .left }
        if abs(pitch) <= 12, yaw >= 8 { return .right }
        if abs(yaw) <= 12, pitch <= -8 { return .up }
        if abs(yaw) <= 12, pitch >= 8 { return .down }
        return nil
    }
}

/// Bounded, balanced selection: two measured frames per viewpoint, ten frames in total.
struct CaptureFrameSelection {
    private var buckets: [CaptureViewpoint: [FaceDepthKeyframe]] = [:]
    var frames: [FaceDepthKeyframe] {
        CaptureViewpoint.allCases.flatMap { buckets[$0] ?? [] }.sorted { $0.timestamp < $1.timestamp }
    }
    var counts: [CaptureViewpoint: Int] { buckets.mapValues(\.count) }
    var isComplete: Bool { CaptureViewpoint.allCases.allSatisfy { buckets[$0]?.count == 2 } }
    var canReconstruct: Bool { counts[.front, default: 0] >= 1 && counts.count >= 3 && frames.count >= 5 }

    @discardableResult
    mutating func append(_ frame: FaceDepthKeyframe) -> Bool {
        guard let viewpoint = CaptureViewpoint.classify(frame.pose) else { return false }
        var pool = buckets[viewpoint, default: []]
        // Preserve a clean first reference. Choose the other frame by angular/temporal diversity.
        if pool.count < 2 {
            pool.append(frame)
        } else {
            func distance(_ f: FaceDepthKeyframe) -> Double {
                let yaw = f.pose.yawDegrees - pool[0].pose.yawDegrees
                let pitch = f.pose.pitchDegrees - pool[0].pose.pitchDegrees
                let time = 0.5 * (f.timestamp - pool[0].timestamp)
                return yaw * yaw + pitch * pitch + time * time
            }
            if distance(frame) > distance(pool[1]) { pool[1] = frame }
        }
        buckets[viewpoint] = pool
        return true
    }
}

enum CaptureFrameTiming {
    static func accepts(imageTime: TimeInterval, depthTime: TimeInterval,
                        previousDepthTime: TimeInterval?, previousAcceptedTime: TimeInterval?) -> Bool {
        guard imageTime.isFinite, depthTime.isFinite, imageTime > 0, depthTime > 0,
              abs(imageTime - depthTime) <= 0.12 else { return false }
        if let previousDepthTime, depthTime <= previousDepthTime { return false }
        if let previousAcceptedTime, depthTime - previousAcceptedTime < 0.25 { return false }
        return true
    }
}

/// Keeps at most one retained camera frame while Vision/depth work runs elsewhere.
final class CaptureFrameIngress {
    private let lock = NSLock()
    private var epoch: UUID?
    private var busy = false
    func setEpoch(_ epoch: UUID?) { lock.lock(); defer { lock.unlock() }; self.epoch = epoch }
    func acquire() -> UUID? {
        lock.lock(); defer { lock.unlock() }
        guard !busy, let epoch else { return nil }
        busy = true; return epoch
    }
    func release() { lock.lock(); defer { lock.unlock() }; busy = false }
}

enum CapturePose {
    static func portrait(_ transform: simd_float4x4) -> FaceScanPose {
        let correction = simd_float4x4(simd_quatf(angle: -.pi / 2, axis: SIMD3<Float>(0, 0, 1)))
        let upright = correction * transform
        let z = SIMD3(upright[2].x, upright[2].y, upright[2].z)
        let position = SIMD3(upright[3].x, upright[3].y, upright[3].z)
        let outward = simd_dot(z, -position) >= 0 ? z : -z
        return FaceScanPose(yawDegrees: -Double(atan2(outward.x, outward.z) * 180 / .pi),
                            pitchDegrees: -Double(atan2(-outward.y, hypotf(outward.x, outward.z)) * 180 / .pi))
    }
}
