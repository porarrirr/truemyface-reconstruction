import XCTest
import simd
@testable import TrueDepthReconstruction

final class TrueDepthCalibrationValidatorTests: XCTestCase {
    func testValidatorAcceptsMatchingForwardAndInverseTables() throws {
        let lens = LensDistortionModel(
            center: SIMD2<Float>(160, 120),
            forwardTable: [Float](repeating: 0.20, count: 16),
            inverseTable: [Float](repeating: -1.0 / 6.0, count: 16),
            maxRadius: 200
        )

        let validation = TrueDepthCalibrationValidator.validate(lens)

        XCTAssertTrue(validation.isValid, validation.issues.joined(separator: ","))
        XCTAssertLessThan(try XCTUnwrap(validation.directCompositionErrorP95), 0.5)
        XCTAssertFalse(validation.usedNumericalForwardInverse)
        XCTAssertFalse(validation.usedNumericalInverseForward)
    }

    func testValidatorMarksWhichDirectionRequiresNumericalInversion() {
        let forwardOnly = LensDistortionModel(
            center: .zero,
            table: [0, 0.1],
            maxRadius: 100,
            usesInverseTable: false
        )
        let inverseOnly = LensDistortionModel(
            center: .zero,
            table: [0, -0.05],
            maxRadius: 100,
            usesInverseTable: true
        )

        let forwardValidation = TrueDepthCalibrationValidator.validate(forwardOnly)
        let inverseValidation = TrueDepthCalibrationValidator.validate(inverseOnly)

        // forward LUT は rectified → observed なので、observed → rectified 側が数値反転になる。
        XCTAssertTrue(forwardValidation.isValid)
        XCTAssertTrue(forwardValidation.usedNumericalForwardInverse)
        XCTAssertFalse(forwardValidation.usedNumericalInverseForward)
        XCTAssertTrue(inverseValidation.isValid)
        XCTAssertTrue(inverseValidation.usedNumericalInverseForward)
        XCTAssertFalse(inverseValidation.usedNumericalForwardInverse)
    }

    func testValidatorRejectsNonFiniteAndNonMonotonicTables() {
        let nonFinite = LensDistortionModel(
            center: .zero,
            table: [0, .nan],
            maxRadius: 100,
            usesInverseTable: false
        )
        let nonMonotonic = LensDistortionModel(
            center: .zero,
            table: [0, 1, -0.9],
            maxRadius: 100,
            usesInverseTable: false
        )

        XCTAssertEqual(
            TrueDepthCalibrationValidator.validate(nonFinite).issues,
            ["forwardTableNonFinite"]
        )
        XCTAssertEqual(
            TrueDepthCalibrationValidator.validate(nonMonotonic).issues,
            ["forwardTableNonMonotonic"]
        )
    }

    func testValidatorRejectsFlatAndBetweenKnotFoldoverMappings() {
        let flat = LensDistortionModel(
            center: .zero, table: [-1, -1], maxRadius: 100, usesInverseTable: false
        )
        // 節点の写像は 0 -> 0, 1 -> 3 と増えるが、区間内では一度負方向へ折り返す。
        let foldover = LensDistortionModel(
            center: .zero, table: [-2, 2], maxRadius: 100, usesInverseTable: false
        )

        XCTAssertEqual(
            TrueDepthCalibrationValidator.validate(flat).issues,
            ["forwardTableNonMonotonic"]
        )
        XCTAssertEqual(
            TrueDepthCalibrationValidator.validate(foldover).issues,
            ["forwardTableNonMonotonic"]
        )
    }

    func testValidatorWarnsAboutMismatchedDirectTablesWithoutRejectingStructurallyValidCalibration() {
        let lens = LensDistortionModel(
            center: SIMD2<Float>(100, 100),
            forwardTable: [Float](repeating: 0.2, count: 16),
            inverseTable: [Float](repeating: 0.2, count: 16),
            maxRadius: 140
        )

        let validation = TrueDepthCalibrationValidator.validate(lens)

        XCTAssertTrue(validation.isValid)
        XCTAssertTrue(validation.issues.isEmpty)
        XCTAssertTrue(validation.warnings.contains("forwardInverseCompositionError"))
        XCTAssertGreaterThan(validation.forwardInverseCompositionErrorP95 ?? 0, 1)
        XCTAssertGreaterThan(validation.inverseForwardCompositionErrorP95 ?? 0, 1)
    }

    func testValidatorChecksIntrinsicsCoordinateBasisAndRawTableSize() {
        let lens = LensDistortionModel(
            center: SIMD2<Float>(50, 50),
            forwardTable: [0, 0.1],
            inverseTable: nil,
            maxRadius: 100,
            targetToLookupScale: SIMD2<Float>(2, 0),
            referenceDimensions: SIMD2<Float>(200, 200),
            imageDimensions: SIMD2<Float>(100, 100)
        )
        let validation = TrueDepthCalibrationValidator.validate(
            lens,
            intrinsics: CameraIntrinsics(fx: 0, fy: 100, cx: 50, cy: 50),
            forwardTableByteCount: 7
        )

        XCTAssertTrue(validation.issues.contains("invalidIntrinsics"))
        XCTAssertTrue(validation.issues.contains("invalidLookupCoordinateScale"))
        XCTAssertTrue(validation.issues.contains("forwardTableInvalidByteCount"))
    }

    func testValidatorRejectsEmptyTableAndInvalidRadius() {
        let empty = LensDistortionModel(
            center: .zero, table: [], maxRadius: 100, usesInverseTable: false
        )
        let invalidRadius = LensDistortionModel(
            center: .zero, table: [0, 0.1], maxRadius: .infinity, usesInverseTable: false
        )

        XCTAssertEqual(TrueDepthCalibrationValidator.validate(empty).issues, ["forwardTableEmpty"])
        XCTAssertTrue(
            TrueDepthCalibrationValidator.validate(invalidRadius).issues.contains("invalidMaximumRadius")
        )
    }

    func testValidatorRejectsMissingLookupTables() {
        let validation = TrueDepthCalibrationValidator.validate(nil)

        XCTAssertFalse(validation.isValid)
        XCTAssertEqual(validation.issues, ["missingLensLookupTable"])
        XCTAssertEqual(validation.fingerprint, "none")
    }

    func testCalibrationFingerprintChangesWithTableOrGeometry() {
        let baseline = LensDistortionModel(
            center: SIMD2<Float>(10, 20), table: [0, 0.1], maxRadius: 100, usesInverseTable: false
        )
        let changedTable = LensDistortionModel(
            center: SIMD2<Float>(10, 20), table: [0, 0.2], maxRadius: 100, usesInverseTable: false
        )
        let changedCenter = LensDistortionModel(
            center: SIMD2<Float>(11, 20), table: [0, 0.1], maxRadius: 100, usesInverseTable: false
        )

        let fingerprint = TrueDepthCalibrationValidator.fingerprint(baseline)
        XCTAssertNotEqual(fingerprint, TrueDepthCalibrationValidator.fingerprint(changedTable))
        XCTAssertNotEqual(fingerprint, TrueDepthCalibrationValidator.fingerprint(changedCenter))
    }


}
