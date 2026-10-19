import Foundation
import simd
import TrueDepthReconstruction

// Analytic ellipsoid ray intersections, generated in memory; no captured face data.
let width = 160, height = 192
let focal: Float = 300
let distance: Float = 0.35
let a: Float = 0.075, b: Float = 0.100, c: Float = 0.070
let matrix = simd_float3x3(SIMD3(focal, 0, 0), SIMD3(0, focal, 0),
                           SIMD3(Float(width - 1) / 2, Float(height - 1) / 2, 1))
let zeroMagnification = [Float](repeating: 0, count: 16).withUnsafeBytes { Data($0) }
let projection = try TrueDepthCalibration.makeProjectionModel(
    intrinsicMatrix: matrix, referenceDimensions: SIMD2(Float(width), Float(height)),
    lensDistortionCenter: SIMD2(matrix[2].x, matrix[2].y),
    forwardLookupTable: nil, inverseLookupTable: zeroMagnification,
    depthImageSize: SIMD2(Float(width), Float(height)))
let intrinsics = projection.intrinsics
var depth = [Float](repeating: .nan, count: width * height)
var mask = [Bool](repeating: false, count: depth.count)
for y in 0..<height {
    for x in 0..<width {
        let rx = (Float(x) - intrinsics.cx) / focal
        let ry = (Float(y) - intrinsics.cy) / focal
        let qa = rx * rx / (a * a) + ry * ry / (b * b) + 1 / (c * c)
        let qb = -2 * distance / (c * c)
        let qc = distance * distance / (c * c) - 1
        let discriminant = qb * qb - 4 * qa * qc
        if discriminant > 0 {
            depth[y * width + x] = (-qb - sqrt(discriminant)) / (2 * qa)
            mask[y * width + x] = true
        }
    }
}
var cameraFromFace = matrix_identity_float4x4
cameraFromFace.columns.3.z = -distance
let frames = (0..<3).map {
    FaceDepthKeyframe(timestamp: Double($0), depthValues: depth, width: width, height: height,
                      intrinsics: intrinsics, lens: projection.lens, cameraFromFace: cameraFromFace, faceMask: mask)
}
guard let result = try Reconstruction.reconstruct(frames: frames) else {
    fatalError("Synthetic reconstruction produced no supported surface")
}
precondition(result.mesh.vertexCount > 2_000 && result.mesh.triangleCount > 0)
precondition(result.mesh.vertices.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite })
precondition(result.mesh.triangleIndices.allSatisfy { Int($0) < result.mesh.vertexCount })
print("Synthetic ellipsoid: \(result.mesh.vertexCount) vertices, \(result.mesh.triangleCount) triangles")
print("Frames: \(result.frameCount); TSDF voxel: \(result.voxelSizeMeters * 1000) mm")
if CommandLine.arguments.count > 1 {
    let path = CommandLine.arguments[1]
    var obj = "# Synthetic ellipsoid; meters; no captured face data\no synthetic_ellipsoid\n"
    for v in result.mesh.vertices { obj += "v \(v.x) \(v.y) \(v.z)\n" }
    for i in stride(from: 0, to: result.mesh.triangleIndices.count, by: 3) {
        obj += "f \(result.mesh.triangleIndices[i] + 1) \(result.mesh.triangleIndices[i + 1] + 1) \(result.mesh.triangleIndices[i + 2] + 1)\n"
    }
    try obj.write(toFile: path, atomically: true, encoding: .utf8)
    print("Wrote \(path)")
}
