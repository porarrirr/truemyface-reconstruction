import Foundation
import XCTest
import simd
@testable import TrueDepthReconstruction

final class TrueDepthCalibrationTests: XCTestCase {
    private func bytes(_ values: [Float]) -> Data {
        values.withUnsafeBytes { Data($0) }
    }

    private func makeProjection(reference: SIMD2<Float> = SIMD2(800, 600),
                                image: SIMD2<Float> = SIMD2(200, 100),
                                forward: Data? = nil, inverse: Data? = nil) throws -> TrueDepthProjectionModel {
        let matrix = simd_float3x3(SIMD3(600, 0, 0), SIMD3(0, 600, 0), SIMD3(400, 300, 1))
        return try TrueDepthCalibration.makeProjectionModel(
            intrinsicMatrix: matrix, referenceDimensions: reference,
            lensDistortionCenter: SIMD2(400, 300),
            forwardLookupTable: forward, inverseLookupTable: inverse, depthImageSize: image)
    }

    func testProjectionScalesIntrinsicsAndLensInReferenceCoordinates() throws {
        let projection = try makeProjection(inverse: bytes([0, 0.05, 0.1]))
        XCTAssertEqual(projection.intrinsics, CameraIntrinsics(fx: 150, fy: 100, cx: 100, cy: 50))
        XCTAssertEqual(projection.lens.center, SIMD2(100, 50))
        XCTAssertEqual(projection.lens.targetToLookupScale, SIMD2(4, 6))
        XCTAssertEqual(projection.lens.maxRadius, 500, accuracy: 0.0001)
        XCTAssertEqual(projection.lens.referenceDimensions, SIMD2(800, 600))
        XCTAssertEqual(projection.lens.imageDimensions, SIMD2(200, 100))
        XCTAssertTrue(projection.validation.isValid)
        XCTAssertTrue(projection.validation.usedNumericalInverseForward)
        XCTAssertLessThan(projection.lens.roundTripPixelError(px: 199, py: 99), 0.001)
    }

    func testForwardOnlyCalibrationUsesNumericalRectification() throws {
        let projection = try makeProjection(forward: bytes([0.2, 0.2]))
        XCTAssertTrue(projection.validation.usedNumericalForwardInverse)
        XCTAssertFalse(projection.lens.usesInverseTable)
        let p = projection.lens.rectifiedPixel(px: 160, py: 50)
        XCTAssertEqual(p.x, 150, accuracy: 0.0001)
    }

    func testBothTablesKeepDirectCompositionWarningsObservable() throws {
        let valid = try makeProjection(forward: bytes([0.2, 0.2]), inverse: bytes([-1 / 6, -1 / 6]))
        XCTAssertTrue(valid.validation.warnings.isEmpty)
        let mismatch = try makeProjection(forward: bytes([0.2, 0.2]), inverse: bytes([0.2, 0.2]))
        XCTAssertTrue(mismatch.validation.isValid)
        XCTAssertEqual(mismatch.validation.warnings, ["forwardInverseCompositionError"])
        XCTAssertGreaterThan(try XCTUnwrap(mismatch.validation.directCompositionErrorP95), 1)
    }

    func testInvalidRawTableLengthsAreRejectedBeforeDecoding() {
        for data in [Data(), Data([0, 1, 2]), Data(repeating: 0, count: 7)] {
            XCTAssertThrowsError(try makeProjection(inverse: data)) {
                XCTAssertEqual($0 as? TrueDepthCalibrationError, .invalidLookupTableByteCount)
            }
        }
    }

    func testMissingOrInvalidCalibrationDoesNotFallBackToPinhole() {
        XCTAssertThrowsError(try makeProjection()) {
            XCTAssertEqual($0 as? TrueDepthCalibrationError, .invalidLens(issues: ["missingLensLookupTable"]))
        }
        for values: [Float] in [[0, .nan], [-2, 2], [0, 1, -0.9]] {
            XCTAssertThrowsError(try makeProjection(inverse: bytes(values))) {
                guard case .invalidLens = $0 as? TrueDepthCalibrationError else { return XCTFail("Unexpected error: \($0)") }
            }
        }
    }

    func testInvalidImageAndReferenceSizesAreRejected() {
        for size: SIMD2<Float> in [SIMD2(0, 100), SIMD2(0.5, 100), SIMD2(100.5, 100), SIMD2(.nan, 100), SIMD2(200, .infinity)] {
            XCTAssertThrowsError(try makeProjection(image: size, inverse: bytes([0]))) {
                XCTAssertEqual($0 as? TrueDepthCalibrationError, .invalidDimensions)
            }
            XCTAssertThrowsError(try makeProjection(reference: size, inverse: bytes([0]))) {
                XCTAssertEqual($0 as? TrueDepthCalibrationError, .invalidDimensions)
            }
        }
    }

    func testInvalidIntrinsicMatrixIsRejected() {
        for focal: Float in [0, -1, .nan] {
            var matrix = matrix_identity_float3x3; matrix[0][0] = focal
            XCTAssertThrowsError(try TrueDepthCalibration.makeProjectionModel(
                intrinsicMatrix: matrix, referenceDimensions: SIMD2(800, 600), lensDistortionCenter: SIMD2(400, 300),
                forwardLookupTable: nil, inverseLookupTable: bytes([0]), depthImageSize: SIMD2(200, 100))) {
                XCTAssertEqual($0 as? TrueDepthCalibrationError, .invalidIntrinsics)
            }
        }
    }

    func testReconstructionRejectsFoldoverAndInvalidCoordinateBasisBeforeGPUWork() {
        for mode in 0..<2 {
            let lens = LensDistortionModel(center: SIMD2(1, 1), forwardTable: mode == 0 ? [0, 1, -0.9] : [0, 0.1],
                                           inverseTable: nil, maxRadius: 10,
                                           referenceDimensions: mode == 1 ? SIMD2(4, 4) : nil)
            let frame = FaceDepthKeyframe(timestamp: 0, depthValues: [0.3, 0.3, 0.3, 0.3], width: 2, height: 2,
                                          intrinsics: CameraIntrinsics(fx: 20, fy: 20, cx: 0.5, cy: 0.5),
                                          lens: lens, cameraFromFace: matrix_identity_float4x4,
                                          faceMask: [true, true, true, true])
            XCTAssertThrowsError(try Reconstruction.reconstruct(frames: [frame])) {
                XCTAssertEqual($0 as? ReconstructionInputError, .invalidLens(frame: 0))
            }
        }
    }
}
