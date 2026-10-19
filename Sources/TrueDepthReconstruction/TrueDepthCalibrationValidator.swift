import Foundation
import simd

/// Local validation of lens tables, calibration coordinates and projection consistency.
/// Structural errors reject calibration; direct-table composition discrepancies are warnings.
public struct LensCalibrationValidation: Equatable {
    public var isValid: Bool
    public var issues: [String]
    public var warnings: [String] = []
    public var directCompositionErrorP95: Float?
    public var directCompositionErrorMax: Float? = nil
    public var forwardInverseCompositionErrorP95: Float? = nil
    public var forwardInverseCompositionErrorMax: Float? = nil
    public var inverseForwardCompositionErrorP95: Float? = nil
    public var inverseForwardCompositionErrorMax: Float? = nil
    public var fingerprint: String
    public var usedNumericalForwardInverse: Bool
    public var usedNumericalInverseForward: Bool
}

public enum TrueDepthCalibrationValidator {
    public static let maximumDirectCompositionErrorP95: Float = 1.0
    public static let maximumDirectCompositionErrorMax: Float = 2.0
    private static let sampleCount = 65
    private static let angularSampleCount = 8

    public static func validate(_ lens: LensDistortionModel?,
                         intrinsics: CameraIntrinsics? = nil,
                         forwardTableByteCount: Int? = nil,
                         inverseTableByteCount: Int? = nil) -> LensCalibrationValidation {
        guard let lens else {
            return LensCalibrationValidation(
                isValid: false,
                issues: ["missingLensLookupTable"],
                directCompositionErrorP95: nil,
                fingerprint: "none",
                usedNumericalForwardInverse: false,
                usedNumericalInverseForward: false
            )
        }

        var issues: [String] = []
        var warnings: [String] = []
        if let intrinsics,
           !intrinsics.fx.isFinite || !intrinsics.fy.isFinite ||
            !intrinsics.cx.isFinite || !intrinsics.cy.isFinite ||
            intrinsics.fx <= 0 || intrinsics.fy <= 0 {
            issues.append("invalidIntrinsics")
        }
        if !lens.center.x.isFinite || !lens.center.y.isFinite {
            issues.append("nonFiniteCenter")
        }
        if !lens.maxRadius.isFinite || lens.maxRadius <= 0 {
            issues.append("invalidMaximumRadius")
        }
        if !lens.targetToLookupScale.x.isFinite || !lens.targetToLookupScale.y.isFinite ||
            lens.targetToLookupScale.x <= 0 || lens.targetToLookupScale.y <= 0 {
            issues.append("invalidLookupCoordinateScale")
        }
        if let referenceDimensions = lens.referenceDimensions,
           !isValidDimensions(referenceDimensions) {
            issues.append("invalidReferenceDimensions")
        }
        if let imageDimensions = lens.imageDimensions {
            if !isValidDimensions(imageDimensions) {
                issues.append("invalidImageDimensions")
            } else if lens.center.x < 0 || lens.center.y < 0 ||
                        lens.center.x > imageDimensions.x || lens.center.y > imageDimensions.y {
                issues.append("distortionCenterOutsideImage")
            }
        }
        if (lens.referenceDimensions == nil) != (lens.imageDimensions == nil) {
            issues.append("incompleteLookupCoordinateBasis")
        }
        if let byteCount = forwardTableByteCount,
           byteCount == 0 || byteCount % MemoryLayout<Float>.size != 0 {
            issues.append("forwardTableInvalidByteCount")
        }
        if let byteCount = inverseTableByteCount,
           byteCount == 0 || byteCount % MemoryLayout<Float>.size != 0 {
            issues.append("inverseTableInvalidByteCount")
        }
        let forwardValid = validateTable(lens.forwardTable, name: "forward", issues: &issues)
        let inverseValid = validateTable(lens.inverseTable, name: "inverse", issues: &issues)
        if lens.forwardTable == nil, lens.inverseTable == nil {
            issues.append("missingLensLookupTable")
        }

        var compositionP95: Float?
        var compositionMax: Float?
        var forwardInverseP95: Float?
        var forwardInverseMax: Float?
        var inverseForwardP95: Float?
        var inverseForwardMax: Float?
        if forwardValid, inverseValid, lens.maxRadius.isFinite, lens.maxRadius > 0 {
            let points = compositionSamplePoints(lens: lens)
            let forwardInverseErrors = points.map { point -> Float in
                let rectified = lens.rectifiedPixel(px: point.x, py: point.y)
                let restored = lens.distortedPixel(px: rectified.x, py: rectified.y)
                return simd_length(restored - point)
            }.filter(\.isFinite).sorted()
            let inverseForwardErrors = points.map { point -> Float in
                let observed = lens.distortedPixel(px: point.x, py: point.y)
                let restored = lens.rectifiedPixel(px: observed.x, py: observed.y)
                return simd_length(restored - point)
            }.filter(\.isFinite).sorted()
            if forwardInverseErrors.count == points.count,
               inverseForwardErrors.count == points.count {
                forwardInverseP95 = percentile(forwardInverseErrors, fraction: 0.95)
                forwardInverseMax = forwardInverseErrors.last
                inverseForwardP95 = percentile(inverseForwardErrors, fraction: 0.95)
                inverseForwardMax = inverseForwardErrors.last
                compositionP95 = max(forwardInverseP95 ?? 0, inverseForwardP95 ?? 0)
                compositionMax = max(forwardInverseMax ?? 0, inverseForwardMax ?? 0)
                if (compositionP95 ?? .infinity) > maximumDirectCompositionErrorP95 ||
                    (compositionMax ?? .infinity) > maximumDirectCompositionErrorMax {
                    warnings.append("forwardInverseCompositionError")
                }
            } else {
                issues.append("nonFiniteComposition")
            }
        }

        return LensCalibrationValidation(
            isValid: issues.isEmpty,
            issues: issues,
            warnings: warnings,
            directCompositionErrorP95: compositionP95,
            directCompositionErrorMax: compositionMax,
            forwardInverseCompositionErrorP95: forwardInverseP95,
            forwardInverseCompositionErrorMax: forwardInverseMax,
            inverseForwardCompositionErrorP95: inverseForwardP95,
            inverseForwardCompositionErrorMax: inverseForwardMax,
            fingerprint: fingerprint(lens),
            usedNumericalForwardInverse: lens.inverseTable == nil && lens.forwardTable != nil,
            usedNumericalInverseForward: lens.forwardTable == nil && lens.inverseTable != nil
        )
    }

    public static func fingerprint(_ lens: LensDistortionModel) -> String {
        var hash: UInt64 = 14_695_981_039_346_656_037
        func mix(_ value: UInt32) {
            var littleEndian = value.littleEndian
            Swift.withUnsafeBytes(of: &littleEndian) { bytes in
                for byte in bytes {
                    hash ^= UInt64(byte)
                    hash &*= 1_099_511_628_211
                }
            }
        }
        mix(lens.center.x.bitPattern)
        mix(lens.center.y.bitPattern)
        mix(lens.maxRadius.bitPattern)
        mix(lens.targetToLookupScale.x.bitPattern)
        mix(lens.targetToLookupScale.y.bitPattern)
        mix(lens.referenceDimensions?.x.bitPattern ?? UInt32.max)
        mix(lens.referenceDimensions?.y.bitPattern ?? UInt32.max)
        mix(lens.imageDimensions?.x.bitPattern ?? UInt32.max)
        mix(lens.imageDimensions?.y.bitPattern ?? UInt32.max)
        mix(UInt32(lens.forwardTable?.count ?? 0))
        lens.forwardTable?.forEach { mix($0.bitPattern) }
        mix(UInt32(lens.inverseTable?.count ?? 0))
        lens.inverseTable?.forEach { mix($0.bitPattern) }
        return String(format: "%016llx", hash)
    }

    private static func validateTable(_ table: [Float]?,
                                      name: String,
                                      issues: inout [String]) -> Bool {
        guard let table else { return false }
        guard !table.isEmpty else {
            issues.append("\(name)TableEmpty")
            return false
        }
        guard table.allSatisfy(\.isFinite) else {
            issues.append("\(name)TableNonFinite")
            return false
        }

        if table.count == 1 {
            guard 1 + table[0] > 0 else {
                issues.append("\(name)TableInvalidRadius")
                return false
            }
            return true
        }

        // 線形補間された magnification m(r) に対し、半径写像は
        // f(r)=r(1+m(r))。節点だけでなく各区間両端の f'(r) を検査する。
        let interval = 1 / Float(table.count - 1)
        for index in 0..<(table.count - 1) {
            let r0 = Float(index) * interval
            let r1 = Float(index + 1) * interval
            let slope = (table[index + 1] - table[index]) / interval
            let derivative0 = 1 + table[index] + r0 * slope
            let derivative1 = 1 + table[index + 1] + r1 * slope
            if !derivative0.isFinite || !derivative1.isFinite ||
                derivative0 <= 0 || derivative1 <= 0 {
                issues.append("\(name)TableNonMonotonic")
                return false
            }
        }
        return true
    }

    private static func isValidDimensions(_ dimensions: SIMD2<Float>) -> Bool {
        dimensions.x.isFinite && dimensions.y.isFinite && dimensions.x > 0 && dimensions.y > 0
    }

    private static func compositionSamplePoints(lens: LensDistortionModel) -> [SIMD2<Float>] {
        var points: [SIMD2<Float>] = []
        points.reserveCapacity(sampleCount * angularSampleCount)
        for radiusIndex in 0..<sampleCount {
            let radius = lens.maxRadius * Float(radiusIndex) / Float(sampleCount - 1)
            for angleIndex in 0..<angularSampleCount {
                let angle = 2 * Float.pi * Float(angleIndex) / Float(angularSampleCount)
                let lookupDelta = SIMD2<Float>(cos(angle), sin(angle)) * radius
                let targetDelta = lookupDelta / lens.targetToLookupScale
                points.append(lens.center + targetDelta)
            }
        }
        return points
    }

    private static func percentile(_ sorted: [Float], fraction: Float) -> Float? {
        guard !sorted.isEmpty else { return nil }
        let index = Int((Float(sorted.count - 1) * min(1, max(0, fraction))).rounded(.up))
        return sorted[min(index, sorted.count - 1)]
    }
}
