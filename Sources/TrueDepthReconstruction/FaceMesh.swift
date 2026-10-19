import Foundation
import simd

/// Triangle surface in face coordinates, with distances in meters.
public struct FaceMesh: Equatable {
    public var vertices: [SIMD3<Float>]
    public var triangleIndices: [UInt32]
    public var vertexCount: Int { vertices.count }
    public var triangleCount: Int { triangleIndices.count / 3 }
    public init(vertices: [SIMD3<Float>], triangleIndices: [UInt32]) {
        self.vertices = vertices
        self.triangleIndices = triangleIndices
    }
}
struct MeshBounds {
    var min: SIMD3<Float>
    var max: SIMD3<Float>
    var size: SIMD3<Float> { max - min }
}
extension FaceMesh {
    var bounds: MeshBounds {
        guard let first = vertices.first else {
            return MeshBounds(min: .zero, max: .zero)
        }
        var lo = first
        var hi = first
        for v in vertices {
            lo = simd.min(lo, v)
            hi = simd.max(hi, v)
        }
        return MeshBounds(min: lo, max: hi)
    }

    func taubinSmoothed(iterations: Int, lambda: Float = 0.5, mu: Float = -0.53) -> FaceMesh {
        guard iterations > 0, triangleCount > 0 else { return self }

        var adj = [[Int]](repeating: [], count: vertexCount)
        var i = 0
        while i + 2 < triangleIndices.count {
            let a = Int(triangleIndices[i]), b = Int(triangleIndices[i + 1]), c = Int(triangleIndices[i + 2])
            guard a < vertexCount, b < vertexCount, c < vertexCount else {
                i += 3
                continue
            }
            adj[a] += [b, c]; adj[b] += [a, c]; adj[c] += [a, b]
            i += 3
        }
        adj = adj.map { Array(Set($0)) }

        func laplacianPass(_ verts: [SIMD3<Float>], _ factor: Float) -> [SIMD3<Float>] {
            var next = verts
            for v in 0..<vertexCount {
                let neighbors = adj[v]
                guard !neighbors.isEmpty else { continue }
                let avg = neighbors.reduce(SIMD3<Float>.zero) { $0 + verts[$1] } / Float(neighbors.count)
                next[v] = verts[v] + factor * (avg - verts[v])
            }
            return next
        }

        var verts = vertices
        for _ in 0..<iterations {          // 各反復 = λ 収縮 → μ 膨張（Taubin）
            verts = laplacianPass(verts, lambda)
            verts = laplacianPass(verts, mu)
        }
        return FaceMesh(
            vertices: verts,
            triangleIndices: triangleIndices
        )
    }

    func decimated(targetVertexCount target: Int) -> FaceMesh {
        guard target > 0, vertexCount > target, triangleCount > 0 else { return self }

        let b = bounds
        let size = b.size
        let volume = Swift.max(size.x * size.y * size.z, 1e-9)
        let cellSize = cbrtf(volume / Float(target))
        guard cellSize.isFinite, cellSize > 0 else { return self }

        struct CellKey: Hashable {
            var x: Int
            var y: Int
            var z: Int
        }
        struct Accumulator {
            var sum: SIMD3<Float>
            var count: Int
            var index: UInt32
        }

        var cells: [CellKey: Accumulator] = [:]
        var remap = [UInt32](repeating: 0, count: vertexCount)
        for (i, v) in vertices.enumerated() {
            let p = (v - b.min) / cellSize
            let key = CellKey(x: Int(floorf(p.x)), y: Int(floorf(p.y)), z: Int(floorf(p.z)))
            if var acc = cells[key] {
                acc.sum += v
                acc.count += 1
                remap[i] = acc.index
                cells[key] = acc
            } else {
                let index = UInt32(cells.count)
                cells[key] = Accumulator(sum: v, count: 1, index: index)
                remap[i] = index
            }
        }

        var newVertices = [SIMD3<Float>](repeating: .zero, count: cells.count)
        for acc in cells.values {
            newVertices[Int(acc.index)] = acc.sum / Float(acc.count)
        }

        var newIndices: [UInt32] = []
        newIndices.reserveCapacity(triangleIndices.count)
        var i = 0
        while i + 2 < triangleIndices.count {
            let ia = Int(triangleIndices[i])
            let ib = Int(triangleIndices[i + 1])
            let ic = Int(triangleIndices[i + 2])
            guard ia < remap.count, ib < remap.count, ic < remap.count else {
                i += 3
                continue
            }
            let a = remap[ia]
            let b = remap[ib]
            let c = remap[ic]
            if a != b && b != c && a != c {
                newIndices += [a, b, c]
            }
            i += 3
        }

        guard !newVertices.isEmpty, !newIndices.isEmpty else { return self }
        return FaceMesh(
            vertices: newVertices,
            triangleIndices: newIndices
        )
    }

    func keepingVertices(_ selected: Set<Int>) -> FaceMesh {
        guard !selected.isEmpty else {
            return FaceMesh(
                vertices: [],
                triangleIndices: []
            )
        }

        var remap = [UInt32](repeating: .max, count: vertexCount)
        var newVertices: [SIMD3<Float>] = []
        newVertices.reserveCapacity(Swift.min(vertexCount, selected.count))

        var nextIndex: UInt32 = 0
        for index in vertices.indices where selected.contains(index) {
            remap[index] = nextIndex
            newVertices.append(vertices[index])
            nextIndex += 1
        }

        var newIndices: [UInt32] = []
        newIndices.reserveCapacity(triangleIndices.count)
        var i = 0
        while i + 2 < triangleIndices.count {
            let ia = Int(triangleIndices[i])
            let ib = Int(triangleIndices[i + 1])
            let ic = Int(triangleIndices[i + 2])
            guard ia < remap.count, ib < remap.count, ic < remap.count else {
                i += 3
                continue
            }
            let a = remap[ia]
            let b = remap[ib]
            let c = remap[ic]
            if a != .max, b != .max, c != .max {
                newIndices += [a, b, c]
            }
            i += 3
        }

        return FaceMesh(
            vertices: newVertices,
            triangleIndices: newIndices
        )
    }

    func largestConnectedComponent() -> (mesh: FaceMesh, count: Int) {
        guard vertexCount > 0, triangleCount > 0 else { return (self, 0) }
        var adjacency = [[Int]](repeating: [], count: vertexCount)
        var i = 0
        while i + 2 < triangleIndices.count {
            let a = Int(triangleIndices[i])
            let b = Int(triangleIndices[i + 1])
            let c = Int(triangleIndices[i + 2])
            guard a < vertexCount, b < vertexCount, c < vertexCount else {
                i += 3
                continue
            }
            adjacency[a].append(b); adjacency[a].append(c)
            adjacency[b].append(a); adjacency[b].append(c)
            adjacency[c].append(a); adjacency[c].append(b)
            i += 3
        }

        var visited = [Bool](repeating: false, count: vertexCount)
        var best: [Int] = []
        var componentCount = 0
        var stack: [Int] = []
        for start in 0..<vertexCount where !visited[start] {
            if adjacency[start].isEmpty {
                visited[start] = true
                continue
            }
            componentCount += 1
            stack.removeAll(keepingCapacity: true)
            stack.append(start)
            visited[start] = true
            var component: [Int] = []
            while let node = stack.popLast() {
                component.append(node)
                for neighbor in adjacency[node] where !visited[neighbor] {
                    visited[neighbor] = true
                    stack.append(neighbor)
                }
            }
            if component.count > best.count { best = component }
        }

        guard !best.isEmpty else { return (self, componentCount) }
        return (keepingVertices(Set(best)), componentCount)
    }

}
