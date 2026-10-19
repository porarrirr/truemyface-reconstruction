import Foundation
import simd

/// Head orientation is used to seed reconstruction order; it supplies no facial shape.
public struct FaceScanPose: Equatable {
    public var yawDegrees: Double
    public var pitchDegrees: Double
    public init(yawDegrees: Double = 0, pitchDegrees: Double = 0) {
        self.yawDegrees = yawDegrees; self.pitchDegrees = pitchDegrees
    }
}

/// Row-major measured depth plus calibrated rigid registration. No image or template mesh.
public struct FaceDepthKeyframe {
    var measuredDepthSupport: [Bool]? = nil
    public var timestamp: TimeInterval
    public var pose: FaceScanPose
    public var depthValues: [Float]
    public var width: Int
    public var height: Int
    public var intrinsics: CameraIntrinsics
    public var lens: LensDistortionModel?
    /// Rigid face-local to camera transform: camera +x right, +y up, -z forward.
    public var cameraFromFace: simd_float4x4
    /// True means a depth pixel belongs to the facial foreground.
    public var faceMask: [Bool]

    public init(timestamp: TimeInterval, pose: FaceScanPose = FaceScanPose(),
                depthValues: [Float], width: Int, height: Int,
                intrinsics: CameraIntrinsics, lens: LensDistortionModel? = nil,
                cameraFromFace: simd_float4x4, faceMask: [Bool]) {
        self.timestamp = timestamp; self.pose = pose
        self.depthValues = depthValues; self.width = width; self.height = height
        self.intrinsics = intrinsics; self.lens = lens
        self.cameraFromFace = cameraFromFace; self.faceMask = faceMask
    }
}
