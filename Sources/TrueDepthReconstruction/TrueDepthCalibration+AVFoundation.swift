import AVFoundation
import simd

extension TrueDepthCalibration {
    /// Reads only local SDK calibration fields; does not start capture or perform any I/O.
    /// Use the calibration attached to the corresponding measured depth image.
    public static func makeProjectionModel(
        calibration: AVCameraCalibrationData,
        depthImageSize: SIMD2<Float>
    ) throws -> TrueDepthProjectionModel {
        let reference = calibration.intrinsicMatrixReferenceDimensions
        let center = calibration.lensDistortionCenter
        return try makeProjectionModel(
            intrinsicMatrix: calibration.intrinsicMatrix,
            referenceDimensions: SIMD2(Float(reference.width), Float(reference.height)),
            lensDistortionCenter: SIMD2(Float(center.x), Float(center.y)),
            forwardLookupTable: calibration.lensDistortionLookupTable,
            inverseLookupTable: calibration.inverseLensDistortionLookupTable,
            depthImageSize: depthImageSize
        )
    }
}
