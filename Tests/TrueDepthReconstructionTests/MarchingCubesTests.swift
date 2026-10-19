import XCTest
import simd
@testable import TrueDepthReconstruction

/// `MarchingCubes` のテーブル/アルゴリズムを解析的な球 SDF で検証する。

/// 等値面抽出の不具合を即座に検出する（半径誤差 < 1 voxel・面積比 ~1・非多様体辺 0）。
final class MarchingCubesTests: XCTestCase {
    func testSphereSurfaceIsClosedAndCorrectRadius() {
        let radius: Float = 0.05
        let n = 48
        let pad: Float = 0.02
        let half = radius + pad
        let spacing = (2 * half) / Float(n - 1)
        let dims = SIMD3<Int>(n, n, n)
        let origin = SIMD3<Float>(-half, -half, -half)

        var field = [Float](repeating: 0, count: n * n * n)
        let valid = [Bool](repeating: true, count: n * n * n)
        for z in 0..<n {
            let wz = origin.z + Float(z) * spacing
            for y in 0..<n {
                let wy = origin.y + Float(y) * spacing
                for x in 0..<n {
                    let wx = origin.x + Float(x) * spacing
                    let idx = (z * n + y) * n + x
                    field[idx] = sqrt(wx * wx + wy * wy + wz * wz) - radius
                }
            }
        }

        let mesh = MarchingCubes.surface(
            field: field, valid: valid, dims: dims,
            origin: origin, spacing: spacing, iso: 0)

        XCTAssertGreaterThan(mesh.vertices.count, 1000)
        XCTAssertEqual(mesh.triangles.count % 3, 0)

        // すべての頂点が半径 ~radius にある（< 1 voxel）。
        var maxRadiusError: Float = 0
        for v in mesh.vertices {
            maxRadiusError = max(maxRadiusError, abs(simd_length(v) - radius))
        }
        XCTAssertLessThan(maxRadiusError, spacing, "radius error \(maxRadiusError * 1000)mm >= voxel")

        // 面積が 4*pi*r^2 に一致（±3%）。
        var area: Float = 0
        var i = 0
        while i + 2 < mesh.triangles.count {
            let a = mesh.vertices[Int(mesh.triangles[i])]
            let b = mesh.vertices[Int(mesh.triangles[i + 1])]
            let c = mesh.vertices[Int(mesh.triangles[i + 2])]
            area += 0.5 * simd_length(simd_cross(b - a, c - a))
            i += 3
        }
        let areaRef = 4 * Float.pi * radius * radius
        XCTAssertEqual(area / areaRef, 1.0, accuracy: 0.03, "area ratio \(area / areaRef)")

        // 閉曲面: すべての無向辺がちょうど 2 つの三角形に共有される。
        var edgeCounts: [Int64: Int] = [:]
        func key(_ a: UInt32, _ b: UInt32) -> Int64 {
            let lo = Int64(min(a, b))
            let hi = Int64(max(a, b))
            return lo << 32 | hi
        }
        i = 0
        while i + 2 < mesh.triangles.count {
            let t0 = mesh.triangles[i], t1 = mesh.triangles[i + 1], t2 = mesh.triangles[i + 2]
            edgeCounts[key(t0, t1), default: 0] += 1
            edgeCounts[key(t1, t2), default: 0] += 1
            edgeCounts[key(t2, t0), default: 0] += 1
            i += 3
        }
        let boundaryEdges = edgeCounts.values.filter { $0 != 2 }.count
        XCTAssertEqual(boundaryEdges, 0, "non-manifold/boundary edges = \(boundaryEdges)")
    }
}

