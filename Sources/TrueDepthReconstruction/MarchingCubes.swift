import Foundation
import simd

/// Extracts the zero crossing of an observed scalar field on a regular 3D grid.
/// A cell is skipped if any corner has no measurement support.
/// Field storage: field[z * ny * nx + y * nx + x].
enum MarchingCubes {
    struct Mesh {
        var vertices: [SIMD3<Float>]
        var triangles: [UInt32]
    }

    /// - Parameters:
    ///   - field: スカラ場（z-major, 長さ nx*ny*nz）。
    ///   - valid: 各ボクセルが観測済みか（同じ並び）。
    ///   - dims: (nx, ny, nz)。
    ///   - origin: ボクセル (0,0,0) 中心のワールド座標（x,y,z）。
    ///   - spacing: ボクセル一辺（メートル）。
    ///   - iso: 等値面レベル（TSDF は 0）。
    static func surface(field: [Float],
                        valid: [Bool],
                        dims: SIMD3<Int>,
                        origin: SIMD3<Float>,
                        spacing: Float,
                        iso: Float = 0) -> Mesh {
        let nx = dims.x
        let ny = dims.y
        let nz = dims.z
        guard nx > 1, ny > 1, nz > 1,
              field.count == nx * ny * nz,
              valid.count == field.count else {
            return Mesh(vertices: [], triangles: [])
        }

        @inline(__always)
        func at(_ x: Int, _ y: Int, _ z: Int) -> Int { (z * ny + y) * nx + x }

        let corner = MarchingCubesTables.corner
        let edgeVertex = MarchingCubesTables.edgeVertex
        let triTable = MarchingCubesTables.triTable

        var vertices: [SIMD3<Float>] = []
        var triangles: [UInt32] = []
        // 共有辺で頂点を重複させないためのキャッシュ（辺の両端ボクセル座標で識別）。
        var vertexCache: [Int64: UInt32] = [:]

        // 辺の頂点キー: 両端グローバル格子座標を昇順にして 1 つの Int64 へ詰める。
        @inline(__always)
        func edgeKey(_ ax: Int, _ ay: Int, _ az: Int, _ bx: Int, _ by: Int, _ bz: Int) -> Int64 {
            func pack(_ x: Int, _ y: Int, _ z: Int) -> Int64 {
                (Int64(x) << 42) | (Int64(y) << 21) | Int64(z)
            }
            let a = pack(ax, ay, az)
            let b = pack(bx, by, bz)
            let lo = min(a, b)
            let hi = max(a, b)
            return lo &* 1_000_003 &+ hi
        }

        var cornerVals = [Float](repeating: 0, count: 8)

        for z in 0..<(nz - 1) {
            for y in 0..<(ny - 1) {
                for x in 0..<(nx - 1) {
                    var cubeIndex = 0
                    var allValid = true
                    for c in 0..<8 {
                        let off = corner[c]
                        let idx = at(x + Int(off.x), y + Int(off.y), z + Int(off.z))
                        if !valid[idx] { allValid = false; break }
                        let v = field[idx]
                        cornerVals[c] = v
                        if v < iso { cubeIndex |= (1 << c) }
                    }
                    guard allValid, cubeIndex != 0, cubeIndex != 255 else { continue }

                    @inline(__always)
                    func vertexOnEdge(_ edge: Int) -> UInt32 {
                        let (a, b) = edgeVertex[edge]
                        let oa = corner[a]
                        let ob = corner[b]
                        let ax = x + Int(oa.x), ay = y + Int(oa.y), az = z + Int(oa.z)
                        let bx = x + Int(ob.x), by = y + Int(ob.y), bz = z + Int(ob.z)
                        let key = edgeKey(ax, ay, az, bx, by, bz)
                        if let cached = vertexCache[key] { return cached }
                        let va = cornerVals[a]
                        let vb = cornerVals[b]
                        let denom = vb - va
                        let t: Float = abs(denom) < 1e-12 ? 0.5 : (iso - va) / denom
                        let pa = SIMD3<Float>(Float(ax), Float(ay), Float(az))
                        let pb = SIMD3<Float>(Float(bx), Float(by), Float(bz))
                        let p = pa + t * (pb - pa)
                        let world = origin + p * spacing
                        let index = UInt32(vertices.count)
                        vertices.append(world)
                        vertexCache[key] = index
                        return index
                    }

                    let row = triTable[cubeIndex]
                    var e = 0
                    while row[e] != -1 {
                        let v0 = vertexOnEdge(row[e])
                        let v1 = vertexOnEdge(row[e + 1])
                        let v2 = vertexOnEdge(row[e + 2])
                        triangles.append(v0)
                        triangles.append(v1)
                        triangles.append(v2)
                        e += 3
                    }
                }
            }
        }

        return Mesh(vertices: vertices, triangles: triangles)
    }
}

