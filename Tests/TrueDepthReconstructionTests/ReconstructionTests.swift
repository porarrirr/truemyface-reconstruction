import Metal
import XCTest
import simd
@testable import TrueDepthReconstruction

final class ReconstructionTests: XCTestCase {
    private func frame() -> FaceDepthKeyframe {
        FaceDepthKeyframe(timestamp: 0, depthValues: [0.3, 0.3, 0.3, 0.3], width: 2, height: 2,
                          intrinsics: CameraIntrinsics(fx: 20, fy: 20, cx: 0.5, cy: 0.5),
                          cameraFromFace: matrix_identity_float4x4, faceMask: [true, true, true, true])
    }

    func testEmptyInputDoesNotInventGeometry() throws {
        XCTAssertNil(try Reconstruction.reconstruct(frames: []))
    }

    func testInputRejectsWrongBufferSizeAndOverflowBeforeGPUWork() {
        for mode in 0..<3 {
            var f = frame()
            if mode == 0 { f.depthValues.removeLast() }
            if mode == 1 { f.faceMask.removeLast() }
            if mode == 2 { f.width = Int.max }
            XCTAssertThrowsError(try Reconstruction.reconstruct(frames: [f])) {
                XCTAssertEqual($0 as? ReconstructionInputError, .invalidDimensions(frame: 0))
            }
        }
    }

    func testInputRejectsNonFiniteOrZeroFocalLength() {
        for focal: Float in [0, -.infinity, .nan] {
            var f = frame(); f.intrinsics.fx = focal
            XCTAssertThrowsError(try Reconstruction.reconstruct(frames: [f])) {
                XCTAssertEqual($0 as? ReconstructionInputError, .invalidIntrinsics(frame: 0))
            }
        }
    }

    func testInputRejectsScaleReflectionAndNonFinitePose() {
        for mode in 0..<3 {
            var f = frame()
            if mode == 0 { f.cameraFromFace[0].x = 2 }
            if mode == 1 { f.cameraFromFace[0].x = -1 }
            if mode == 2 { f.cameraFromFace[3].z = .nan }
            XCTAssertThrowsError(try Reconstruction.reconstruct(frames: [f])) {
                XCTAssertEqual($0 as? ReconstructionInputError, .invalidPose(frame: 0))
            }
        }
    }

    func testMetalDeprojectionMatchesCPUWithLensAndRegistration() throws {
        var f = frame()
        f.depthValues = [0.3, .nan, 0.4, 0.5]
        f.faceMask[3] = false
        f.cameraFromFace[3].z = -0.35
        f.lens = LensDistortionModel(center: SIMD2(0.5, 0.5), table: [0.02, 0.04], maxRadius: 2)
        let points = try MeasuredDepthDeprojector.deprojectedCloud(keyframe: f, stride: 1)
        XCTAssertEqual(points.count, 2)
        for (outputIndex, pixel) in [(0, 0), (1, 2)] {
            let optical = LensDistortionModel.unproject(px: Float(pixel % 2), py: Float(pixel / 2),
                                                        z: f.depthValues[pixel], intrinsics: f.intrinsics, lens: f.lens)
            let camera = LensDistortionModel.opticalToARKitCamera(optical)
            let expected = simd_inverse(f.cameraFromFace) * SIMD4(camera, 1)
            XCTAssertLessThan(simd_distance(points[outputIndex], SIMD3(expected.x, expected.y, expected.z)), 0.000001)
        }
    }

    func testLensRoundTripWithNonUniformCalibrationScale() {
        let lens = LensDistortionModel(center: SIMD2(100, 80), forwardTable: nil,
                                       inverseTable: [0.01, 0.08, 0.2], maxRadius: 250,
                                       targetToLookupScale: SIMD2(2, 3))
        for point: SIMD2<Float> in [SIMD2(0, 0), SIMD2(100, 80), SIMD2(199, 159)] {
            XCTAssertLessThan(lens.roundTripPixelError(px: point.x, py: point.y), 0.001)
        }
    }

    func testICPRestoresKnownRigidTransformOnCurvedSurface() {
        var target: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        for row in 0..<35 {
            let phi = -0.9 + Float(row) * 1.8 / 34
            for column in 0..<45 {
                let theta = -1.0 + Float(column) * 2.0 / 44
                let p = SIMD3<Float>(0.08 * cos(phi) * sin(theta), 0.11 * sin(phi), 0.06 * cos(phi) * cos(theta))
                target.append(p)
                normals.append(simd_normalize(SIMD3(p.x / (0.08 * 0.08), p.y / (0.11 * 0.11), p.z / (0.06 * 0.06))))
            }
        }
        let rotation = simd_float3x3(simd_quatf(angle: 0.015, axis: simd_normalize(SIMD3<Float>(1, 2, 3))))
        let translation = SIMD3<Float>(0.001, -0.0015, 0.001)
        let source = target.map { rotation * $0 + translation }
        let model = MeasuredFaceReconstructor.OrientedPointSet(points: target, normals: normals, cellSize: 0.012)
        let correction = MeasuredFaceReconstructor.icpAlign(source: source, model: model, initial: .identity,
                                                            scales: [(0.012, 20), (0.005, 20), (0.002, 30)])
        let error = zip(source, target).reduce(Float(0)) { $0 + simd_distance(correction.apply($1.0), $1.1) } / Float(target.count)
        XCTAssertLessThan(error, 0.0001)
    }

    // MARK: - Face depth layer (background removal)

    /// Removes only the background measured by projected-mask overshoot pixels (entirely behind the face)
    /// and keeps separate components in front of the face (hair fringe) or overlapping it in depth.
    func testFaceDepthLayerRemovesOnlyComponentsEntirelyBehindTheFace() {
        let width = 40, height = 40
        var depth = [Float](repeating: 0, count: width * height)
        var mask = [Bool](repeating: false, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let i = y * width + x
                mask[i] = true
                // Face: recedes smoothly from 0.24 m at the center outward (neighbor step < 2 cm).
                let r = hypotf(Float(x - 20), Float(y - 20))
                depth[i] = 0.24 + 0.002 * r
            }
        }
        // Two left columns: background wall measured by overshoot pixels (+50 cm).
        for y in 0..<height { for x in 0..<2 { depth[y * width + x] = 0.75 } }
        // Part of the top three rows: fringe in front of the face (separated by a > 2 cm step).
        for y in 0..<3 { for x in 15..<25 { depth[y * width + x] = 0.21 } }
        // Bottom-right patch: separated by a step but overlapping the face layer in depth (hair, ear).
        for y in 36..<40 { for x in 36..<40 { depth[y * width + x] = 0.25 } }

        let layer = MeasuredDepthPreprocessor.faceDepthLayer(
            depth: depth, mask: mask, width: width, height: height,
            maximumDepth: 1.5, jumpMeters: 0.02)

        XCTAssertEqual(layer.removedBehindPixelCount, 2 * height)
        for y in 0..<height {
            XCTAssertFalse(layer.mask[y * width], "background wall is removed")
            XCTAssertFalse(layer.mask[y * width + 1], "background wall is removed")
        }
        XCTAssertTrue(layer.mask[0 * width + 20], "components in front of the face are kept")
        XCTAssertTrue(layer.mask[38 * width + 38], "components overlapping the face depth range are kept")
        XCTAssertTrue(layer.mask[20 * width + 20], "face is kept")
    }

    func testFaceDepthLayerKeepsMaskWhenNothingLiesBehindTheFace() {
        let width = 16, height = 16
        let depth = (0..<(width * height)).map { 0.30 + Float($0 % width) * 0.001 }
        let mask = [Bool](repeating: true, count: width * height)
        let layer = MeasuredDepthPreprocessor.faceDepthLayer(
            depth: depth, mask: mask, width: width, height: height,
            maximumDepth: 1.5, jumpMeters: 0.02)
        XCTAssertEqual(layer.removedBehindPixelCount, 0)
        XCTAssertEqual(layer.mask, mask)
    }

    // MARK: - Radii sampled directly from the TSDF field

    /// Recovers the surface radius of an analytic 80 mm cylinder SDF from priors offset by 1 mm,
    /// without picking up a floater outside the ±2-voxel search window.
    func testFieldSampledRadiiRecoversAnalyticSurfaceAndIgnoresDistantFloaters() {
        let voxel: Float = 0.0015
        let truncation: Float = 0.0075
        let radius: Float = 0.08
        let floaterRadius: Float = 0.088
        let origin = SIMD3<Float>(-0.06, -0.03, 0.02)
        let dims = SIMD3<Int>(81, 41, 61)
        var values = [Float](repeating: 1, count: dims.x * dims.y * dims.z)
        for z in 0..<dims.z {
            for y in 0..<dims.y {
                for x in 0..<dims.x {
                    let p = origin + SIMD3<Float>(Float(x), Float(y), Float(z)) * voxel
                    let r = hypotf(p.x, p.z)
                    // Main surface (+ outside) plus a 2 mm thick floater shell (negative inside) 8 mm outside it.
                    var value = max(-1, min(1, (r - radius) / truncation))
                    if abs(r - floaterRadius) < 0.001 { value = -0.5 }
                    values[(z * dims.y + y) * dims.x + x] = value
                }
            }
        }
        let field = MeasuredTSDFSurfaceReconstructor.Field(
            values: values, valid: [Bool](repeating: true, count: values.count),
            dims: dims, origin: origin, voxel: voxel)
        let binTheta = Float(0.65) * .pi / 180
        let frame = CylindricalFaceFrame(
            axisX: 0, axisZ: 0, thetaOrigin: -0.4, yOrigin: -0.01,
            binTheta: binTheta, binHeight: 0.001, columns: Int(0.8 / binTheta), rows: 20)
        let count = frame.columns * frame.rows
        let prior = [Float](repeating: radius + 0.001, count: count)
        var fillDepth = [Int](repeating: 0, count: count)
        fillDepth[0] = 5  // bins far from surface points are not searched

        let sampled = MeasuredFaceReconstructor.fieldSampledRadii(
            prior: prior, fillDepth: fillDepth, frame: frame, field: field, maximumPriorFillDepth: 2)

        XCTAssertTrue(sampled[0].isNaN)
        let recovered = sampled.dropFirst().filter(\.isFinite)
        XCTAssertEqual(recovered.count, count - 1)
        for r in recovered {
            XCTAssertEqual(r, radius, accuracy: 0.0001)
        }
    }
}
