import XCTest
import simd
@testable import TrueDepthReconstruction

final class MeasuredTSDFSurfaceReconstructorTests: XCTestCase {
    func testProjectiveIntegrationUsesVoxelPositionAndOneVotePerFrame() {
        let frame = makeKeyframe(timestamp: 0, validRatio: 1, maskCoverage: 1)
        var sums = [Float](repeating: 0, count: 3)
        var weights = sums
        for expectedSupport in 1...2 {
            MeasuredTSDFSurfaceReconstructor.integrate(
                keyframe: frame, correction: .identity,
                origin: SIMD3<Float>(0, 0, -0.301), dims: SIMD3<Int>(1, 1, 3),
                voxel: 0.001, truncation: 0.0075, depthTruncation: 1.5,
                tsdfSum: &sums, weights: &weights)
            for index in 0..<3 {
                XCTAssertEqual(weights[index], Float(expectedSupport))
                let expected = Float(index - 1) * 0.001 / 0.0075
                XCTAssertEqual(sums[index] / weights[index], expected, accuracy: 0.00001)
            }
        }
    }

    func testProjectiveIntegrationUndoesRegistrationBeforeCameraProjection() {
        var frame = makeKeyframe(timestamp: 0, validRatio: 1, maskCoverage: 1)
        frame.cameraFromFace.columns.3.z = -0.1
        let correction = MeasuredFaceReconstructor.RigidCorrection(
            rotation: simd_float3x3(simd_quatf(angle: 0.35, axis: SIMD3<Float>(0, 1, 0))),
            translation: SIMD3<Float>(0.008, -0.006, 0.003))
        var sums: [Float] = [0]
        var weights: [Float] = [0]
        MeasuredTSDFSurfaceReconstructor.integrate(
            keyframe: frame, correction: correction,
            origin: correction.apply(SIMD3<Float>(0, 0, -0.2)), dims: SIMD3<Int>(1, 1, 1),
            voxel: 0.001, truncation: 0.0075, depthTruncation: 1.5,
            tsdfSum: &sums, weights: &weights)
        XCTAssertEqual(weights, [1])
        XCTAssertEqual(sums[0], 0, accuracy: 0.00001)
    }

    func testProjectiveIntegrationRejectsDepthDiscontinuitiesAndMaskedNeighbors() {
        for mode in 0..<3 {
            var frame = makeKeyframe(timestamp: 0, validRatio: 1, maskCoverage: 1)
            if mode == 0 { frame.depthValues[0] = 0.8 }
            if mode == 1 { frame.faceMask[0] = false }
            if mode == 2 { frame.measuredDepthSupport = [false, true, true, true] }
            var sums: [Float] = [0]
            var weights: [Float] = [0]
            MeasuredTSDFSurfaceReconstructor.integrate(
                keyframe: frame, correction: .identity,
                origin: SIMD3<Float>(0, 0, -0.3), dims: SIMD3<Int>(1, 1, 1),
                voxel: 0.001, truncation: 0.0075, depthTruncation: 1.5,
                tsdfSum: &sums, weights: &weights)
            XCTAssertEqual(weights, [0])
        }
    }

    func testMeasuredGridExtentKeepsProtrusionBeyondLegacyTrackingBounds() throws {
        // 旧 ARKit cropBounds の前端 70mm + margin 10mm を越える、96mm の実測鼻尖。
        // TSDF 領域はこの追跡メッシュ境界ではなく、実測点群だけから決まること。
        var cloud: [SIMD3<Float>] = []
        cloud.reserveCapacity(5_000)
        for y in 0..<50 {
            for x in 0..<100 {
                let px = (Float(x) - 49.5) * 0.0014
                let py = (Float(y) - 24.5) * 0.0028
                let nose = max(0, 1 - (px * px + py * py) / (0.012 * 0.012))
                let z: Float = 0.055 + nose * 0.041
                cloud.append(SIMD3<Float>(px, py, z))
            }
        }

        let extent = try XCTUnwrap(MeasuredTSDFSurfaceReconstructor.gridExtent(
            deprojectedClouds: [cloud],
            corrections: [.identity],
            configuration: MeasuredTSDFSurfaceReconstructor.Configuration()
        ))

        XCTAssertGreaterThan(extent.diagnostics.measuredMaximum.z, 0.093)
        XCTAssertGreaterThan(extent.diagnostics.gridMaximum.z, 0.100)
        XCTAssertEqual(extent.diagnostics.surfacePointCount, cloud.count)
        XCTAssertEqual(extent.diagnostics.surfacePointsOutsideGrid, 0)
    }

    func testMeasuredGridExtentAppliesRigidCorrectionsBeforeChoosingBounds() throws {
        var cloud: [SIMD3<Float>] = []
        cloud.reserveCapacity(1_000)
        for index in 0..<1_000 {
            let point = SIMD3<Float>(
                Float(index % 10) * 0.001,
                Float((index / 10) % 10) * 0.001,
                0.040 + Float(index / 100) * 0.001
            )
            cloud.append(point)
        }
        let correction = MeasuredFaceReconstructor.RigidCorrection(
            rotation: matrix_identity_float3x3,
            translation: SIMD3<Float>(0.003, -0.004, 0.025)
        )

        let extent = try XCTUnwrap(MeasuredTSDFSurfaceReconstructor.gridExtent(
            deprojectedClouds: [cloud],
            corrections: [correction],
            configuration: MeasuredTSDFSurfaceReconstructor.Configuration()
        ))

        XCTAssertEqual(extent.diagnostics.measuredMinimum.x, 0.003, accuracy: 0.000_001)
        XCTAssertEqual(extent.diagnostics.measuredMinimum.y, -0.004, accuracy: 0.000_001)
        XCTAssertEqual(extent.diagnostics.measuredMinimum.z, 0.065, accuracy: 0.000_001)
        XCTAssertEqual(extent.diagnostics.measuredMaximum.z, 0.074, accuracy: 0.000_001)
    }

    func testPhysicalFrontCropThresholdDoesNotDependOnRearExtent() throws {
        let nearRear = makeStripMesh(rearZ: 0.020)
        let farRear = makeStripMesh(rearZ: -0.080)

        let near = try XCTUnwrap(MeasuredTSDFSurfaceReconstructor.croppedToMeasuredFront(
            nearRear,
            depthMeters: 0.060,
            referencePercentile: 0.995
        ))
        let far = try XCTUnwrap(MeasuredTSDFSurfaceReconstructor.croppedToMeasuredFront(
            farRear,
            depthMeters: 0.060,
            referencePercentile: 0.995
        ))

        XCTAssertEqual(near.diagnostics.frontReferenceZ, 0.100, accuracy: 0.000_001)
        XCTAssertEqual(far.diagnostics.frontReferenceZ, 0.100, accuracy: 0.000_001)
        XCTAssertEqual(near.diagnostics.thresholdZ, 0.040, accuracy: 0.000_001)
        XCTAssertEqual(far.diagnostics.thresholdZ, 0.040, accuracy: 0.000_001)
    }

    func testPhysicalFrontCropKeepsSixtyMillimetersAndRejectsDeeperVertices() throws {
        let mesh = makeDepthBoundaryMesh()

        let result = try XCTUnwrap(MeasuredTSDFSurfaceReconstructor.croppedToMeasuredFront(
            mesh,
            depthMeters: 0.060,
            referencePercentile: 1
        ))

        XCTAssertEqual(result.diagnostics.frontReferenceZ, 0.100, accuracy: 0.000_001)
        XCTAssertEqual(result.diagnostics.thresholdZ, 0.040, accuracy: 0.000_001)
        XCTAssertEqual(result.diagnostics.inputVertexCount, 6)
        XCTAssertEqual(result.diagnostics.retainedVertexCount, 3)
        XCTAssertTrue(result.mesh.vertices.allSatisfy { $0.z >= 0.040 })
        XCTAssertTrue(result.mesh.vertices.contains { abs($0.z - 0.040) < 0.000_001 })
        XCTAssertFalse(result.mesh.vertices.contains { abs($0.z - 0.039) < 0.000_001 })
    }

    func testPhysicalFrontCropRejectsInvalidConfigurationAndNonFiniteGeometry() {
        let mesh = makeDepthBoundaryMesh()
        XCTAssertNil(MeasuredTSDFSurfaceReconstructor.croppedToMeasuredFront(
            mesh,
            depthMeters: 0,
            referencePercentile: 0.995
        ))
        XCTAssertNil(MeasuredTSDFSurfaceReconstructor.croppedToMeasuredFront(
            mesh,
            depthMeters: 0.060,
            referencePercentile: 1.1
        ))

        var invalid = mesh
        invalid.vertices[0].z = .nan
        XCTAssertNil(MeasuredTSDFSurfaceReconstructor.croppedToMeasuredFront(
            invalid,
            depthMeters: 0.060,
            referencePercentile: 0.995
        ))
    }

    func testMeasuredGridExtentRejectsInvalidMeasuredInputWithoutFallback() {
        XCTAssertNil(MeasuredTSDFSurfaceReconstructor.gridExtent(
            deprojectedClouds: [[]],
            corrections: [.identity],
            configuration: MeasuredTSDFSurfaceReconstructor.Configuration()
        ))
        XCTAssertNil(MeasuredTSDFSurfaceReconstructor.gridExtent(
            deprojectedClouds: [[SIMD3<Float>(repeating: .nan)]],
            corrections: [.identity],
            configuration: MeasuredTSDFSurfaceReconstructor.Configuration()
        ))
    }

    private func makeStripMesh(rearZ: Float) -> FaceMesh {
        var vertices: [SIMD3<Float>] = []
        vertices.reserveCapacity(1_001)
        for index in 0..<996 {
            let x = Float(index % 32) * 0.001
            let y = Float(index / 32) * 0.001
            let z = index == 0 ? rearZ : Float(index) / 995 * 0.100
            vertices.append(SIMD3<Float>(x, y, z))
        }
        for index in 0..<5 {
            vertices.append(SIMD3<Float>(Float(index) * 0.001, 0, 0.200 + Float(index) * 0.010))
        }
        return FaceMesh(
            vertices: vertices,
            triangleIndices: sequentialTriangles(vertexCount: vertices.count)
        )
    }

    private func makeDepthBoundaryMesh() -> FaceMesh {
        FaceMesh(
            vertices: [
                SIMD3<Float>(0, 0, 0.100),
                SIMD3<Float>(0.001, 0, 0.080),
                SIMD3<Float>(0, 0.001, 0.040),
                SIMD3<Float>(0.001, 0.001, 0.039),
                SIMD3<Float>(0.002, 0, 0.020),
                SIMD3<Float>(0.002, 0.001, -0.010)
            ],
            triangleIndices: [0, 1, 2, 3, 4, 5]
        )
    }

    private func sequentialTriangles(vertexCount: Int) -> [UInt32] {
        let triangleVertexCount = vertexCount - vertexCount % 3
        return (0..<triangleVertexCount).map(UInt32.init)
    }
    private func makeKeyframe(timestamp: TimeInterval,
                              validRatio: Double, maskCoverage: Double) -> FaceDepthKeyframe {
        FaceDepthKeyframe(timestamp: timestamp,
                          depthValues: [0.3, 0.3, 0.3, 0.3], width: 2, height: 2,
                          intrinsics: CameraIntrinsics(fx: 20, fy: 20, cx: 0.5, cy: 0.5),
                          cameraFromFace: matrix_identity_float4x4,
                          faceMask: [true, true, true, true])
    }
}
