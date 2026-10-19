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
}
