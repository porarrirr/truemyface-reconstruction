import Foundation
import simd

public enum TrueDepthCalibrationError: Error, Equatable {
    case invalidDimensions
    case invalidIntrinsics
    case invalidLookupTableByteCount
    case invalidLens(issues: [String])
}

/// Local projection parameters and validation results. No storage or transmission behavior.
public struct TrueDepthProjectionModel {
    public let intrinsics: CameraIntrinsics
    public let lens: LensDistortionModel
    public let validation: LensCalibrationValidation
}

/// Builds projection parameters from the calibration attached to the measured depth image.
public enum TrueDepthCalibration {
    /// LUT bytes contain native Float32 radial magnifications, as supplied by the camera SDK.
    /// Missing or invalid lens calibration throws; no uncalibrated fallback is substituted.
    public static func makeProjectionModel(
        intrinsicMatrix: simd_float3x3,
        referenceDimensions: SIMD2<Float>,
        lensDistortionCenter: SIMD2<Float>,
        forwardLookupTable: Data?,
        inverseLookupTable: Data?,
        depthImageSize: SIMD2<Float>
    ) throws -> TrueDepthProjectionModel {
        guard validSize(referenceDimensions), validSize(depthImageSize) else {
            throw TrueDepthCalibrationError.invalidDimensions
        }
        guard (0..<3).allSatisfy({ column in (0..<3).allSatisfy { intrinsicMatrix[column][$0].isFinite } }),
              intrinsicMatrix[0][0] > 0, intrinsicMatrix[1][1] > 0 else {
            throw TrueDepthCalibrationError.invalidIntrinsics
        }
        let intrinsics = CameraIntrinsics.scaled(matrix: intrinsicMatrix,
                                                 referenceSize: referenceDimensions,
                                                 targetSize: depthImageSize)
        guard [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy].allSatisfy(\.isFinite),
              intrinsics.fx > 0, intrinsics.fy > 0 else {
            throw TrueDepthCalibrationError.invalidIntrinsics
        }
        let scale = depthImageSize / referenceDimensions
        let dx = max(lensDistortionCenter.x, referenceDimensions.x - lensDistortionCenter.x)
        let dy = max(lensDistortionCenter.y, referenceDimensions.y - lensDistortionCenter.y)
        let lens = LensDistortionModel(
            center: lensDistortionCenter * scale,
            forwardTable: try floatTable(forwardLookupTable),
            inverseTable: try floatTable(inverseLookupTable),
            maxRadius: max(hypotf(dx, dy), 1),
            targetToLookupScale: referenceDimensions / depthImageSize,
            referenceDimensions: referenceDimensions,
            imageDimensions: depthImageSize
        )
        let validation = TrueDepthCalibrationValidator.validate(
            lens, intrinsics: intrinsics,
            forwardTableByteCount: forwardLookupTable?.count,
            inverseTableByteCount: inverseLookupTable?.count
        )
        guard validation.isValid else {
            throw TrueDepthCalibrationError.invalidLens(issues: validation.issues)
        }
        return TrueDepthProjectionModel(intrinsics: intrinsics, lens: lens, validation: validation)
    }

    private static func validSize(_ size: SIMD2<Float>) -> Bool {
        size.x.isFinite && size.y.isFinite && size.x >= 1 && size.y >= 1
            && size.x.rounded() == size.x && size.y.rounded() == size.y
    }

    private static func floatTable(_ data: Data?) throws -> [Float]? {
        guard let data else { return nil }
        guard !data.isEmpty, data.count.isMultiple(of: MemoryLayout<Float>.size) else {
            throw TrueDepthCalibrationError.invalidLookupTableByteCount
        }
        return data.withUnsafeBytes { bytes in
            stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
                bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
            }
        }
    }
}
