import Foundation
import simd

/// TrueDepth の前処理済みフレームを TSDF へ融合し、円筒フュージョンへ渡す
/// 単一の実測表面を作る。ARKit の顔メッシュ頂点・トポロジは使用しない。
///
/// 多フレーム点群を直接円筒ビンへ入れると、同じ皮膚面が深度ノイズと残留姿勢差で
/// 複数の殻になり、頬・目元・口元に波打ちが残る。先に TSDF の zero crossing へ
/// 一本化することで、顔パーツの立体感を保ったまま後段の穴なしグリッドへ渡せる。
enum MeasuredTSDFSurfaceReconstructor {
    struct Configuration {
        var voxelSize: Float = 0.0015
        var truncation: Float = 0.0075
        /// Number of contributing frames, independent of depth pixel density.
        var minimumWeight: Float = 1
        var depthTruncation: Float = 1.5
        var boundsMargin: Float = 0.010
        /// ICP 補正済み実測点群から TSDF 領域を決める際の外れ値除外率。
        /// 0.1〜99.9 percentile の外側は margin で吸収し、単発飛び点で全軸の
        /// voxel 解像度が低下することを防ぐ。
        var boundsLowPercentile: Float = 0.001
        var boundsHighPercentile: Float = 0.999
        /// 最大連結面の前方基準から保持する実寸奥行き（メートル）。
        /// 入力 Z 幅に対する割合だと端末ごとの画角・観測範囲差がクロップ位置を変えるため、
        /// 顔表面として必要な物理深さを直接指定する。
        var frontDepthMeters: Float = 0.060
        /// 鼻先の孤立点にクロップ位置を引っ張られないための前方 Z パーセンタイル。
        var frontReferencePercentile: Float = 0.995
        var smoothingIterations: Int = 2
        var maximumDimension: Int = 256
    }

    struct FrontCropDiagnostics: Equatable {
        var inputMinimumZ: Float
        var inputMaximumZ: Float
        var frontReferenceZ: Float
        var thresholdZ: Float
        var inputVertexCount: Int
        var retainedVertexCount: Int
    }

    struct GridExtentDiagnostics: Equatable {
        var measuredMinimum: SIMD3<Float>
        var measuredMaximum: SIMD3<Float>
        var gridMinimum: SIMD3<Float>
        var gridMaximum: SIMD3<Float>
        var voxelSize: Float
        var surfacePointCount: Int
        var surfacePointsOutsideGrid: Int
    }

    struct Result {
        var mesh: FaceMesh
        var validVoxelCount: Int
        var rawVertexCount: Int
        var componentCount: Int
        var frontCropDiagnostics: FrontCropDiagnostics
        var gridExtentDiagnostics: GridExtentDiagnostics
    }

    static func reconstruct(
        keyframes: [FaceDepthKeyframe],
        deprojectedClouds: [[SIMD3<Float>]],
        corrections: [MeasuredFaceReconstructor.RigidCorrection],
        configuration: Configuration = Configuration()
    ) -> Result? {
        guard !keyframes.isEmpty,
              keyframes.count == deprojectedClouds.count,
              keyframes.count == corrections.count,
              let extent = gridExtent(
                deprojectedClouds: deprojectedClouds,
                corrections: corrections,
                configuration: configuration
              ) else {
            return nil
        }
        let origin = extent.origin
        let dims = extent.dimensions
        let voxel = extent.voxel

        let voxelCount = dims.x * dims.y * dims.z
        guard voxelCount > 8 else { return nil }
        let truncation = max(configuration.truncation, voxel)
        var tsdfSum = [Float](repeating: 0, count: voxelCount)
        var weights = [Float](repeating: 0, count: voxelCount)

        for index in keyframes.indices {
            integrate(
                keyframe: keyframes[index],
                correction: corrections[index],
                origin: origin,
                dims: dims,
                voxel: voxel,
                truncation: truncation,
                depthTruncation: configuration.depthTruncation,
                tsdfSum: &tsdfSum,
                weights: &weights
            )
        }

        var field = [Float](repeating: 1, count: voxelCount)
        var valid = [Bool](repeating: false, count: voxelCount)
        var validVoxelCount = 0
        for index in 0..<voxelCount where weights[index] >= configuration.minimumWeight {
            field[index] = tsdfSum[index] / weights[index]
            valid[index] = true
            validVoxelCount += 1
        }
        guard validVoxelCount > 0 else { return nil }

        let raw = MarchingCubes.surface(
            field: field,
            valid: valid,
            dims: dims,
            origin: origin,
            spacing: voxel,
            iso: 0
        )
        guard !raw.vertices.isEmpty, raw.triangles.count >= 3 else { return nil }

        var mesh = FaceMesh(
            vertices: raw.vertices,
            triangleIndices: raw.triangles
        )
        let rawVertexCount = mesh.vertexCount
        let firstLargest = mesh.largestConnectedComponent()
        mesh = firstLargest.mesh
        guard let cropped = croppedToMeasuredFront(
            mesh,
            depthMeters: configuration.frontDepthMeters,
            referencePercentile: configuration.frontReferencePercentile
        ) else {
            return nil
        }
        mesh = cropped.mesh.largestConnectedComponent().mesh
        if configuration.smoothingIterations > 0 {
            mesh = mesh.taubinSmoothed(iterations: configuration.smoothingIterations)
        }
        guard mesh.vertexCount > 2_000, mesh.triangleCount > 0 else { return nil }

        return Result(
            mesh: mesh,
            validVoxelCount: validVoxelCount,
            rawVertexCount: rawVertexCount,
            componentCount: firstLargest.count,
            frontCropDiagnostics: cropped.diagnostics,
            gridExtentDiagnostics: extent.diagnostics
        )
    }

    static func gridExtent(
        deprojectedClouds: [[SIMD3<Float>]],
        corrections: [MeasuredFaceReconstructor.RigidCorrection],
        configuration: Configuration
    ) -> (origin: SIMD3<Float>, dimensions: SIMD3<Int>, voxel: Float,
          diagnostics: GridExtentDiagnostics)? {
        guard !deprojectedClouds.isEmpty,
              deprojectedClouds.count == corrections.count,
              configuration.maximumDimension >= 2,
              configuration.boundsMargin.isFinite,
              configuration.boundsMargin >= configuration.truncation,
              configuration.boundsLowPercentile.isFinite,
              configuration.boundsHighPercentile.isFinite,
              configuration.boundsLowPercentile >= 0,
              configuration.boundsHighPercentile <= 1,
              configuration.boundsLowPercentile < configuration.boundsHighPercentile else {
            return nil
        }

        var correctedPoints: [SIMD3<Float>] = []
        correctedPoints.reserveCapacity(deprojectedClouds.reduce(0) { $0 + $1.count })
        for (points, correction) in zip(deprojectedClouds, corrections) {
            for point in points {
                let corrected = correction.apply(point)
                if corrected.x.isFinite, corrected.y.isFinite, corrected.z.isFinite {
                    correctedPoints.append(corrected)
                }
            }
        }
        guard correctedPoints.count > 100 else { return nil }

        var xs = correctedPoints.map(\.x)
        var ys = correctedPoints.map(\.y)
        var zs = correctedPoints.map(\.z)
        xs.sort()
        ys.sort()
        zs.sort()
        let lowerIndex = min(
            correctedPoints.count - 1,
            Int(Float(correctedPoints.count - 1) * configuration.boundsLowPercentile)
        )
        let upperIndex = min(
            correctedPoints.count - 1,
            Int(Float(correctedPoints.count - 1) * configuration.boundsHighPercentile)
        )
        let measuredMinimum = SIMD3<Float>(xs[lowerIndex], ys[lowerIndex], zs[lowerIndex])
        let measuredMaximum = SIMD3<Float>(xs[upperIndex], ys[upperIndex], zs[upperIndex])
        guard measuredMinimum.x < measuredMaximum.x,
              measuredMinimum.y < measuredMaximum.y,
              measuredMinimum.z < measuredMaximum.z else {
            return nil
        }

        let margin = SIMD3<Float>(repeating: configuration.boundsMargin)
        let lower = measuredMinimum - margin
        let requestedUpper = measuredMaximum + margin
        let span = requestedUpper - lower
        let maximumCells = Float(configuration.maximumDimension - 1)
        let voxel = max(
            configuration.voxelSize,
            span.x / maximumCells,
            span.y / maximumCells,
            span.z / maximumCells,
            0.0001
        )
        let dimensions = SIMD3<Int>(
            min(Int(ceil(span.x / voxel)) + 1, configuration.maximumDimension),
            min(Int(ceil(span.y / voxel)) + 1, configuration.maximumDimension),
            min(Int(ceil(span.z / voxel)) + 1, configuration.maximumDimension)
        )
        guard dimensions.x > 1, dimensions.y > 1, dimensions.z > 1 else { return nil }
        let gridMaximum = lower + SIMD3<Float>(
            Float(dimensions.x - 1),
            Float(dimensions.y - 1),
            Float(dimensions.z - 1)
        ) * voxel
        let outsideCount = correctedPoints.lazy.filter { point in
            point.x < lower.x || point.x > gridMaximum.x ||
                point.y < lower.y || point.y > gridMaximum.y ||
                point.z < lower.z || point.z > gridMaximum.z
        }.count
        return (
            lower,
            dimensions,
            voxel,
            GridExtentDiagnostics(
                measuredMinimum: measuredMinimum,
                measuredMaximum: measuredMaximum,
                gridMinimum: lower,
                gridMaximum: gridMaximum,
                voxelSize: voxel,
                surfacePointCount: correctedPoints.count,
                surfacePointsOutsideGrid: outsideCount
            )
        )
    }

    /// Evaluate the residual at the stored voxel position, once per frame.
    /// Pixel density and ray marching step count must not inflate observation support.
    static func integrate(
        keyframe: FaceDepthKeyframe,
        correction: MeasuredFaceReconstructor.RigidCorrection,
        origin: SIMD3<Float>,
        dims: SIMD3<Int>,
        voxel: Float,
        truncation: Float,
        depthTruncation: Float,
        tsdfSum: inout [Float],
        weights: inout [Float]
    ) {
        let width = keyframe.width
        let height = keyframe.height
        guard width > 1, height > 1,
              keyframe.depthValues.count == width * height else { return }
        let hasMask = keyframe.faceMask.count == width * height
        if let support = keyframe.measuredDepthSupport,
           support.count != width * height { return }
        let inverseRotation = correction.rotation.transpose
        for z in 0..<dims.z {
            for y in 0..<dims.y {
                for x in 0..<dims.x {
                    let position = origin + SIMD3<Float>(Float(x), Float(y), Float(z)) * voxel
                    let face = inverseRotation * (position - correction.translation)
                    let camera = keyframe.cameraFromFace * SIMD4<Float>(face.x, face.y, face.z, 1)
                    let optical = LensDistortionModel.arkitCameraToOptical(SIMD3<Float>(camera.x, camera.y, camera.z))
                    guard optical.z > 0, optical.z < depthTruncation,
                          let pixel = LensDistortionModel.projectToPixel(
                            optical, intrinsics: keyframe.intrinsics, lens: keyframe.lens
                          ), pixel.x.isFinite, pixel.y.isFinite,
                          pixel.x >= 0, pixel.y >= 0,
                          pixel.x < Float(width - 1), pixel.y < Float(height - 1) else { continue }
                    let px = Int(pixel.x), py = Int(pixel.y)
                    let i = py * width + px
                    if hasMask && (!keyframe.faceMask[i] || !keyframe.faceMask[i + 1]
                        || !keyframe.faceMask[i + width] || !keyframe.faceMask[i + width + 1]) {
                        continue
                    }
                    // Inpainted depth is useful for registration but is not a new observation.
                    if let support = keyframe.measuredDepthSupport,
                       !support[i] || !support[i + 1] || !support[i + width] || !support[i + width + 1] {
                        continue
                    }
                    let d00 = keyframe.depthValues[i], d10 = keyframe.depthValues[i + 1]
                    let d01 = keyframe.depthValues[i + width], d11 = keyframe.depthValues[i + width + 1]
                    guard d00.isFinite, d10.isFinite, d01.isFinite, d11.isFinite else { continue }
                    let minimum = min(d00, d10, d01, d11)
                    let maximum = max(d00, d10, d01, d11)
                    guard minimum > 0, maximum < depthTruncation,
                          maximum - minimum <= 0.010 else { continue }
                    let u = pixel.x - Float(px), v = pixel.y - Float(py)
                    let depth = (1 - v) * ((1 - u) * d00 + u * d10)
                        + v * ((1 - u) * d01 + u * d11)
                    let residual = depth - optical.z
                    guard abs(residual) <= truncation else { continue }
                    let index = (z * dims.y + y) * dims.x + x
                    tsdfSum[index] += residual / truncation
                    weights[index] += 1
                }
            }
        }
    }

    static func croppedToMeasuredFront(
        _ mesh: FaceMesh,
        depthMeters: Float,
        referencePercentile: Float
    ) -> (mesh: FaceMesh, diagnostics: FrontCropDiagnostics)? {
        guard !mesh.vertices.isEmpty,
              depthMeters.isFinite,
              depthMeters > 0,
              referencePercentile.isFinite,
              referencePercentile >= 0,
              referencePercentile <= 1 else {
            return nil
        }

        let sortedZ = mesh.vertices.map(\.z).filter(\.isFinite).sorted()
        guard sortedZ.count == mesh.vertices.count,
              let minimumZ = sortedZ.first,
              let maximumZ = sortedZ.last else {
            return nil
        }
        let referenceIndex = min(
            sortedZ.count - 1,
            Int(Float(sortedZ.count - 1) * referencePercentile)
        )
        let frontReferenceZ = sortedZ[referenceIndex]
        let thresholdZ = frontReferenceZ - depthMeters
        // Float の減算丸めで閾値上の頂点だけが落ちないよう、1µm の数値許容を持たせる。
        let thresholdTolerance: Float = 0.000_001
        let keep = Set(mesh.vertices.indices.filter {
            mesh.vertices[$0].z >= thresholdZ - thresholdTolerance
        })
        let croppedMesh = mesh.keepingVertices(keep)
        return (
            croppedMesh,
            FrontCropDiagnostics(
                inputMinimumZ: minimumZ,
                inputMaximumZ: maximumZ,
                frontReferenceZ: frontReferenceZ,
                thresholdZ: thresholdZ,
                inputVertexCount: mesh.vertexCount,
                retainedVertexCount: croppedMesh.vertexCount
            )
        )
    }
}
