import Foundation
import simd

public enum ReconstructionInputError: Error, Equatable {
    case invalidDimensions(frame: Int)
    case invalidIntrinsics(frame: Int)
    case invalidPose(frame: Int)
    case invalidLens(frame: Int)
}

public struct ReconstructionResult {
    public let mesh: FaceMesh
    public let frameCount: Int
    public let inputPointCount: Int
    public let medianPoseCorrectionMeters: Float
    public let voxelSizeMeters: Float
}

/// Synchronous, local reconstruction. Run away from the main/UI thread.
public enum Reconstruction {
    /// Returns nil when measured support cannot produce a surface. Metal failures throw.
    /// Default production reconstruction settings are used without reducing quality budgets.
    public static func reconstruct(frames: [FaceDepthKeyframe]) throws -> ReconstructionResult? {
        for (index, frame) in frames.enumerated() {
            let (count, overflow) = frame.width.multipliedReportingOverflow(by: frame.height)
            guard frame.width >= 2, frame.height >= 2, !overflow,
                  count <= Int(UInt32.max), frame.depthValues.count == count,
                  frame.faceMask.count == count else {
                throw ReconstructionInputError.invalidDimensions(frame: index)
            }
            let k = frame.intrinsics
            guard [k.fx, k.fy, k.cx, k.cy].allSatisfy(\.isFinite), k.fx > 0, k.fy > 0 else {
                throw ReconstructionInputError.invalidIntrinsics(frame: index)
            }
            let m = frame.cameraFromFace
            let rotation = simd_float3x3(SIMD3(m[0].x, m[0].y, m[0].z),
                                        SIMD3(m[1].x, m[1].y, m[1].z),
                                        SIMD3(m[2].x, m[2].y, m[2].z))
            let gram = rotation.transpose * rotation
            var rigid = true
            for row in 0..<3 {
                for column in 0..<3 {
                    rigid = rigid && abs(gram[column][row] - (column == row ? 1 : 0)) < 0.001
                }
            }
            guard (0..<4).allSatisfy({ column in (0..<4).allSatisfy { m[column][$0].isFinite } }),
                  abs(m[0].w) < 0.00001, abs(m[1].w) < 0.00001,
                  abs(m[2].w) < 0.00001, abs(m[3].w - 1) < 0.00001,
                  rigid, abs(simd_determinant(rotation) - 1) < 0.001,
                  frame.timestamp.isFinite, frame.pose.yawDegrees.isFinite, frame.pose.pitchDegrees.isFinite else {
                throw ReconstructionInputError.invalidPose(frame: index)
            }
            if let lens = frame.lens {
                guard TrueDepthCalibrationValidator.validate(lens, intrinsics: k).isValid else {
                    throw ReconstructionInputError.invalidLens(frame: index)
                }
            }
        }
        guard let result = try MeasuredFaceReconstructor.reconstruct(keyframes: frames) else { return nil }
        return ReconstructionResult(mesh: result.mesh,
                                    frameCount: result.diagnostics.keyframesIngested,
                                    inputPointCount: result.diagnostics.cloudPointCount,
                                    medianPoseCorrectionMeters: result.diagnostics.icpMedianCorrectionMeters,
                                    voxelSizeMeters: result.diagnostics.tsdfVoxelSize)
    }
}
