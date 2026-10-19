import Foundation
import simd

/// Measured depth -> Metal preprocessing -> robust multiscale point-to-plane ICP
/// -> projective TSDF -> marching cubes -> cylindrical remeshing -> mesh cleanup.
/// Pose supplies rigid registration only. No tracking/template facial geometry is used.
enum MeasuredFaceReconstructor {
    struct Diagnostics: Equatable {
        var keyframesIngested: Int
        var cloudPointCount: Int
        var collapsedPointCount: Int
        var icpMedianCorrectionMeters: Float
        var gridColumns: Int
        var gridRows: Int
        var measuredBinCount: Int
        var tsdfFrontCropInputMinimumZ: Float
        var tsdfFrontCropInputMaximumZ: Float
        var tsdfFrontCropReferenceZ: Float
        var tsdfFrontCropThresholdZ: Float
        var tsdfFrontCropInputVertexCount: Int
        var tsdfFrontCropRetainedVertexCount: Int
        var tsdfMeasuredMinimum: SIMD3<Float>
        var tsdfMeasuredMaximum: SIMD3<Float>
        var tsdfGridMinimum: SIMD3<Float>
        var tsdfGridMaximum: SIMD3<Float>
        var tsdfVoxelSize: Float
        var tsdfSurfacePointCount: Int
        var tsdfSurfacePointsOutsideGrid: Int
        var finalVertexCount: Int
        var finalTriangleCount: Int
    }

    struct Result {
        var mesh: FaceMesh
        var diagnostics: Diagnostics
    }

    struct Configuration {
        /// 顔マスク erosion 回数（マスク縁の飛び画素対策）。
        var maskErosionIterations: Int = 2
        /// 無効画素インペイントの最大反復。
        var inpaintIterations: Int = 25
        /// バイラテラル深度平滑化の range σ（メートル）。
        var bilateralRangeSigma: Float = 0.004
        /// ICP 用フレーム点群の目標点数。
        var icpSamplePointCount: Int = 4000
        /// ICP 粗→細の対応距離（メートル）と反復数（早期終了あり）。
        var icpScales: [(corr: Float, iterations: Int)] = [(0.012, 20), (0.005, 20), (0.002, 30)]
        /// 全体 refinement ラウンド数（全フレームを平均モデルへ再整列）。
        var icpRefinementRounds: Int = 2
        /// 全体 refinement 用の細スケール対応距離（メートル）と反復数。
        var refinementScales: [(corr: Float, iterations: Int)] = [(0.004, 20), (0.0015, 30)]
        /// モデル点群のボクセル平均セル（メートル）。
        var modelVoxel: Float = 0.002
        /// 円筒グリッドの角度分解能（度）と縦分解能（メートル）。
        var binThetaDegrees: Float = CylindricalFaceFrame.defaultBinThetaDegrees
        var binHeight: Float = CylindricalFaceFrame.defaultBinHeight
        /// ビン内ロバスト集計: アンカー百分位と MAD バンド幅（メートル）。
        var binAnchorPercentile: Float = 0.5
        var binBandWidth: Float = 0.0015
        /// 拡散充填で許容する充填リング数（これより深い外挿ビンは捨てる）。
        var maximumFillDepth: Int = 12
        /// 半径グリッドのエッジ保存平滑化: 反復・空間半径（ビン）・空間σ・値域σ（メートル）。
        /// 値域σ 1.2mm が唇の段差を保護するため、空間半径を広げてもエッジは残る。
        var gridSmoothIterations: Int = 2
        var gridSmoothRadius: Int = 3
        var gridSmoothSigmaS: Float = 2.2
        var gridSmoothSigmaR: Float = 0.0012
        /// 有効ビンマスクの開処理（erode→dilate）反復。
        var maskOpenIterations: Int = 2

        var meshSmoothingIterations: Int = 3
        /// 上限頂点数。
        var maximumVertexCount: Int = 200_000

        init() {}

    }

    // MARK: - Entry point

    static func reconstruct(keyframes: [FaceDepthKeyframe],
                            configuration: Configuration = Configuration()) throws -> Result? {
        guard !keyframes.isEmpty else { return nil }
        // 1. 深度前処理
        var frames = keyframes
        for i in frames.indices {
            var mask = frames[i].faceMask
            if mask.count == frames[i].width * frames[i].height {
                mask = try MeasuredDepthPreprocessor.erodedMask(
                    mask,
                    width: frames[i].width,
                    height: frames[i].height,
                    iterations: configuration.maskErosionIterations
                )
                frames[i].faceMask = mask
            }
            if frames[i].measuredDepthSupport == nil {
                frames[i].measuredDepthSupport = frames[i].depthValues.enumerated().map { index, depth in
                    depth.isFinite && depth > 0 && depth < 1.5
                        && (mask.count != frames[i].depthValues.count || mask[index])
                }
            }
            frames[i].depthValues = try MeasuredDepthPreprocessor.preprocessedDepth(
                frames[i].depthValues, mask: frames[i].faceMask,
                width: frames[i].width, height: frames[i].height,
                configuration: configuration)
        }


        // 2. ロバスト多段 ICP（マニフェスト姿勢は初期値のみ）
        let clouds = try frames.map { try deprojectedCloud(keyframe: $0, stride: 1) }

        guard clouds.contains(where: { $0.count > 500 }) else { return nil }
        let (corrections, medianCorrection) = refinePoses(
            frames: frames, clouds: clouds, configuration: configuration)


        // 3. 前処理済み・姿勢補正済み深度を TSDF の zero crossing へ一本化する。
        // ARKit 顔メッシュの頂点や三角形は使用しない。後段の円筒グリッドは TSDF 表面だけを
        // 受け取るため、入力点群を直接集約したときの多重殻・頬の波打ちを持ち込まない。
        guard let tsdfSurface = MeasuredTSDFSurfaceReconstructor.reconstruct(
            keyframes: frames,
            deprojectedClouds: clouds,
            corrections: corrections
        ) else {
            return nil
        }

        let cloudPointCount = clouds.reduce(0) { $0 + $1.count }
        let collapsed = tsdfSurface.mesh.vertices
        guard collapsed.count > 2_000 else { return nil }

        // 4. 円筒フュージョン → グリッドメッシュ
        guard let fused = try cylindricalMesh(points: collapsed, configuration: configuration) else {
            return nil
        }

        var mesh = fused.mesh
        let (largest, _) = mesh.largestConnectedComponent()
        mesh = largest

        if configuration.meshSmoothingIterations > 0 {
            mesh = mesh.taubinSmoothed(iterations: configuration.meshSmoothingIterations)
        }

        if mesh.vertexCount > configuration.maximumVertexCount {
            mesh = mesh.decimated(targetVertexCount: configuration.maximumVertexCount)
        }

        guard mesh.triangleCount > 0 else { return nil }

        let diagnostics = Diagnostics(
            keyframesIngested: frames.count,
            cloudPointCount: cloudPointCount,
            collapsedPointCount: collapsed.count,
            icpMedianCorrectionMeters: medianCorrection,
            gridColumns: fused.columns,
            gridRows: fused.rows,
            measuredBinCount: fused.measuredBins,
            tsdfFrontCropInputMinimumZ: tsdfSurface.frontCropDiagnostics.inputMinimumZ,
            tsdfFrontCropInputMaximumZ: tsdfSurface.frontCropDiagnostics.inputMaximumZ,
            tsdfFrontCropReferenceZ: tsdfSurface.frontCropDiagnostics.frontReferenceZ,
            tsdfFrontCropThresholdZ: tsdfSurface.frontCropDiagnostics.thresholdZ,
            tsdfFrontCropInputVertexCount: tsdfSurface.frontCropDiagnostics.inputVertexCount,
            tsdfFrontCropRetainedVertexCount: tsdfSurface.frontCropDiagnostics.retainedVertexCount,
            tsdfMeasuredMinimum: tsdfSurface.gridExtentDiagnostics.measuredMinimum,
            tsdfMeasuredMaximum: tsdfSurface.gridExtentDiagnostics.measuredMaximum,
            tsdfGridMinimum: tsdfSurface.gridExtentDiagnostics.gridMinimum,
            tsdfGridMaximum: tsdfSurface.gridExtentDiagnostics.gridMaximum,
            tsdfVoxelSize: tsdfSurface.gridExtentDiagnostics.voxelSize,
            tsdfSurfacePointCount: tsdfSurface.gridExtentDiagnostics.surfacePointCount,
            tsdfSurfacePointsOutsideGrid: tsdfSurface.gridExtentDiagnostics.surfacePointsOutsideGrid,
            finalVertexCount: mesh.vertexCount,
            finalTriangleCount: mesh.triangleCount
        )
        return Result(mesh: mesh, diagnostics: diagnostics)
    }

    // MARK: - 1. 深度前処理

    // MARK: - 2. 逆投影と ICP

    /// マスク内の有効深度画素を顔ローカル空間の点へ逆投影する。
    static func deprojectedCloud(keyframe: FaceDepthKeyframe, stride strideStep: Int) throws -> [SIMD3<Float>] {
        try MeasuredDepthDeprojector.deprojectedCloud(keyframe: keyframe, stride: strideStep)
    }

    /// 剛体補正（回転 + 並進）。小角度更新の合成で構築する。
    struct RigidCorrection {
        var rotation: simd_float3x3
        var translation: SIMD3<Float>

        static let identity = RigidCorrection(rotation: matrix_identity_float3x3, translation: .zero)

        func rotate(_ p: SIMD3<Float>) -> SIMD3<Float> { rotation * p }

        func apply(_ p: SIMD3<Float>) -> SIMD3<Float> { rotation * p + translation }

        /// 小角度更新 (ω, t) を左から合成する。
        func updated(angular: SIMD3<Float>, translationDelta: SIMD3<Float>) -> RigidCorrection {
            let angle = simd_length(angular)
            let deltaR: simd_float3x3
            if angle > 1e-9 {
                deltaR = simd_float3x3(simd_quatf(angle: angle, axis: angular / angle))
            } else {
                deltaR = matrix_identity_float3x3
            }
            return RigidCorrection(rotation: deltaR * rotation,
                                   translation: deltaR * translation + translationDelta)
        }
    }

    /// 逐次モデル成長 + 全体 refinement の多段 point-to-plane ICP（Tukey 重み）。
    /// 返り値: フレームごとの補正と、補正並進量の中央値（診断用）。
    static func refinePoses(frames: [FaceDepthKeyframe],
                            clouds: [[SIMD3<Float>]],
                            configuration: Configuration) -> ([RigidCorrection], Float) {
        var corrections = [RigidCorrection](repeating: .identity, count: frames.count)
        let order = frames.indices.sorted { a, b in
            let pa = abs(frames[a].pose.yawDegrees) + abs(frames[a].pose.pitchDegrees)
            let pb = abs(frames[b].pose.yawDegrees) + abs(frames[b].pose.pitchDegrees)
            if pa != pb { return pa < pb }
            if frames[a].timestamp != frames[b].timestamp {
                return frames[a].timestamp < frames[b].timestamp
            }
            return a < b
        }
        let samples = clouds.map { subsampled($0, target: configuration.icpSamplePointCount) }
        // pass 1: 正面に近い順にモデルへ整列しながら成長させる。
        // モデルは ICP の対応先にしか使わないため、サブサンプル点群のボクセル平均で足りる
        // （全点で組むと 20 回のモデル再構築が支配的コストになる）。
        var accumulated: [SIMD3<Float>] = samples[order[0]]
        var model = IncrementalOrientedPointSet(
            voxelCell: configuration.modelVoxel,
            normalRadius: configuration.modelVoxel * 2.5
        )
        model.insert(samples[order[0]])

        for i in order.dropFirst() {

            let modelSnapshot = model.snapshot()


            corrections[i] = icpAlign(source: samples[i], model: modelSnapshot,
                                      initial: .identity, scales: configuration.icpScales)

            let aligned = samples[i].map { corrections[i].apply($0) }
            accumulated.append(contentsOf: aligned)

            model.insert(aligned)

        }

        // pass 2+: 全フレームを平均モデルへ再整列（微修正のみの細スケール）。
        for _ in 0..<max(0, configuration.icpRefinementRounds) {
            var aligned: [SIMD3<Float>] = []
            aligned.reserveCapacity(accumulated.count)
            for (i, points) in samples.enumerated() {
                aligned.append(contentsOf: points.map { corrections[i].apply($0) })
            }
            let refinedModel = OrientedPointSet(
                points: voxelAveraged(aligned, cell: configuration.modelVoxel),
                normalRadius: configuration.modelVoxel * 2.5)
            // モデルはラウンド内で固定なので、スケール別索引はフレーム間で共有する。
            let scaledModels = configuration.refinementScales.map { scale in
                (corr: scale.corr, iterations: scale.iterations,
                 model: refinedModel.regridded(cellSize: scale.corr))
            }
            for i in frames.indices {
                corrections[i] = icpAlign(source: samples[i], initial: corrections[i],
                                          scaledModels: scaledModels)
            }

        }

        var magnitudes = corrections.map { simd_length($0.translation) }
        magnitudes.sort()
        let median = magnitudes.isEmpty ? 0 : magnitudes[magnitudes.count / 2]
        return (corrections, median)
    }

    static func icpAlign(source: [SIMD3<Float>],
                         model: OrientedPointSet,
                         initial: RigidCorrection,
                         scales: [(corr: Float, iterations: Int)],
                         isCancelled: @escaping () -> Bool = { false }) -> RigidCorrection {
        icpAlign(source: source,
                 initial: initial,
                 scaledModels: scales.map { scale in
                     (corr: scale.corr, iterations: scale.iterations,
                      model: model.regridded(cellSize: scale.corr))
                 },
                 isCancelled: isCancelled)
    }

    /// 6x6 正規方程式のチャンク部分和（3x3 ブロック表現）。
    private struct IcpPartial {
        var aa = simd_double3x3()
        var an = simd_double3x3()
        var nn = simd_double3x3()
        var rhsAngular = SIMD3<Double>.zero
        var rhsTranslation = SIMD3<Double>.zero
        var matched = 0
    }

    @inline(__always)
    private static func outerProduct(_ u: SIMD3<Double>, _ v: SIMD3<Double>) -> simd_double3x3 {
        simd_double3x3(columns: (u * v.x, u * v.y, u * v.z))
    }

    /// 対応距離ごとに再グリッド済みのモデル（セル≈corr → 3x3x3 セル走査）で整列する。
    /// 対応点探索は固定長チャンクの並列リダクション。チャンク分割とマージ順は
    /// 点数のみで決まるため、結果は実行ごとに決定的（逐次版とは浮動小数点の
    /// 加算順のみ異なる）。
    private static func icpAlign(source: [SIMD3<Float>],
                                 initial: RigidCorrection,
                                 scaledModels: [(corr: Float, iterations: Int, model: OrientedPointSet)],
                                 isCancelled: @escaping () -> Bool = { false }) -> RigidCorrection {
        guard source.count >= 200 else { return initial }
        var correction = initial
        let chunkSize = 256
        let chunkCount = (source.count + chunkSize - 1) / chunkSize
        var partials = [IcpPartial](repeating: IcpPartial(), count: chunkCount)
        for (corr, iterations, model) in scaledModels {
            if isCancelled() { return correction }
            let tukeyK = max(corr / 2, 0.0015)
            // モデル配列・グリッドの借用をスケール全反復にわたって固定し、
            // ホットループから ARC を完全に排除する。
            model.withUnsafeView { modelView in
                for _ in 0..<iterations {
                    if isCancelled() { return }
                    let current = correction
                    source.withUnsafeBufferPointer { sourcePtr in
                        partials.withUnsafeMutableBufferPointer { partialsPtr in
                            DispatchQueue.concurrentPerform(iterations: chunkCount) { chunk in
                                guard !isCancelled() else { return }
                                var local = IcpPartial()
                                let lower = chunk * chunkSize
                                let upper = min(sourcePtr.count, lower + chunkSize)
                                for s in lower..<upper {
                                    if s & 63 == 0, isCancelled() { return }
                                    let q = current.apply(sourcePtr[s])
                                    guard let (target, normal) = modelView.nearestOriented(to: q, within: corr) else {
                                        continue
                                    }
                                    let residual = simd_dot(target - q, normal)
                                    let a = abs(residual)
                                    guard a < tukeyK * 3 else { continue }
                                    let ratio = min(a / tukeyK, 1)
                                    let base = 1 - ratio * ratio
                                    let weight = Double(base * base)
                                    let angularRow = simd_cross(q, normal)
                                    let u = SIMD3<Double>(Double(angularRow.x), Double(angularRow.y), Double(angularRow.z))
                                    let v = SIMD3<Double>(Double(normal.x), Double(normal.y), Double(normal.z))
                                    let b = Double(residual) * weight
                                    local.rhsAngular += u * b
                                    local.rhsTranslation += v * b
                                    local.aa += outerProduct(u, u * weight)
                                    local.an += outerProduct(u, v * weight)
                                    local.nn += outerProduct(v, v * weight)
                                    local.matched += 1
                                }
                                partialsPtr[chunk] = local
                            }
                        }
                    }
                    var normalMatrix = [Double](repeating: 0, count: 36)
                    var rhs = [Double](repeating: 0, count: 6)
                    var matched = 0
                    for partial in partials {
                        matched += partial.matched
                        for r in 0..<3 {
                            rhs[r] += partial.rhsAngular[r]
                            rhs[r + 3] += partial.rhsTranslation[r]
                            for c in 0..<3 {
                                // simd は列優先: M[col][row]。bottom-left は an の転置。
                                normalMatrix[r * 6 + c] += partial.aa[c][r]
                                normalMatrix[r * 6 + (c + 3)] += partial.an[c][r]
                                normalMatrix[(r + 3) * 6 + c] += partial.an[r][c]
                                normalMatrix[(r + 3) * 6 + (c + 3)] += partial.nn[c][r]
                            }
                        }
                    }
                    guard matched >= 100 else { break }
                    for d in 0..<6 { normalMatrix[d * 6 + d] += 1e-7 }
                    guard let solution = solve6x6(normalMatrix, rhs) else { break }
                    let angular = SIMD3<Float>(Float(solution[0]), Float(solution[1]), Float(solution[2]))
                    let delta = SIMD3<Float>(Float(solution[3]), Float(solution[4]), Float(solution[5]))
                    correction = correction.updated(angular: angular, translationDelta: delta)
                    if simd_length(angular) < 1e-5, simd_length(delta) < 1e-6 { break }
                }
            }
        }
        return correction
    }

    private static func solve6x6(_ matrix: [Double], _ rhs: [Double]) -> [Double]? {
        var a = matrix
        var b = rhs
        for column in 0..<6 {
            var pivotRow = column
            var pivotValue = abs(a[column * 6 + column])
            for row in (column + 1)..<6 {
                let value = abs(a[row * 6 + column])
                if value > pivotValue { pivotValue = value; pivotRow = row }
            }
            guard pivotValue > 1e-12 else { return nil }
            if pivotRow != column {
                for c in 0..<6 { a.swapAt(column * 6 + c, pivotRow * 6 + c) }
                b.swapAt(column, pivotRow)
            }
            let pivot = a[column * 6 + column]
            for c in 0..<6 { a[column * 6 + c] /= pivot }
            b[column] /= pivot
            for row in 0..<6 where row != column {
                let factor = a[row * 6 + column]
                guard factor != 0 else { continue }
                for c in 0..<6 { a[row * 6 + c] -= factor * a[column * 6 + c] }
                b[row] -= factor * b[column]
            }
        }
        return b.allSatisfy(\.isFinite) ? b : nil
    }

    // MARK: - 点群ユーティリティ（一様グリッド NN / ボクセル平均 / 法線）

    struct Cell: Hashable {
        var x: Int32
        var y: Int32
        var z: Int32

        init(_ p: SIMD3<Float>, cell: Float) {
            x = Int32((p.x / cell).rounded(.down))
            y = Int32((p.y / cell).rounded(.down))
            z = Int32((p.z / cell).rounded(.down))
        }

        init(x: Int32, y: Int32, z: Int32) {
            self.x = x; self.y = y; self.z = z
        }
    }

    /// 一様グリッド索引上の点集合。近傍探索と局所平面フィット法線を提供する。
    struct OrientedPointSet {
        let points: [SIMD3<Float>]
        let normals: [SIMD3<Float>]
        private let grid: CompactUniformGrid

        init(points: [SIMD3<Float>], normalRadius: Float) {
            self.points = points
            let grid = CompactUniformGrid(points: points, cellSize: max(normalRadius, 1e-4))
            self.grid = grid
            self.normals = Self.estimatedNormals(points: points, grid: grid, radius: normalRadius)
        }

        init(points: [SIMD3<Float>], normals: [SIMD3<Float>], cellSize: Float) {
            self.points = points
            self.normals = normals
            self.grid = CompactUniformGrid(points: points, cellSize: max(cellSize, 1e-4))
        }

        /// 探索半径に合わせたセルサイズで索引だけ組み直す（法線は再利用）。
        /// セル≈半径なら nearestOriented の走査が 3x3x3 セルで済む。
        func regridded(cellSize: Float) -> OrientedPointSet {
            OrientedPointSet(points: points, normals: normals, cellSize: cellSize)
        }

        /// ARC を伴わない借用ビュー（並列ホットループ用）。
        struct UnsafeView {
            let points: UnsafeBufferPointer<SIMD3<Float>>
            let normals: UnsafeBufferPointer<SIMD3<Float>>
            let grid: CompactUniformGrid.UnsafeView

            /// 中心セルを先に走査して bestDistance を締め、外周セルはセル AABB への
            /// 最小距離が best 以上なら中身を見ずに飛ばす（下界による正確な枝刈り。
            /// 粗スケールではセルあたり数百点になるため、この枝刈りが ICP の律速を決める）。
            func nearestOriented(to q: SIMD3<Float>, within radius: Float) -> (SIMD3<Float>, SIMD3<Float>)? {
                let cellSize = grid.cellSize
                let reach = Int32((radius / cellSize).rounded(.up))
                let center = grid.cellCoordinate(of: q)
                var bestIndex = -1
                var bestDistance = radius * radius
                let centerRange = grid.pointIndexRange(inCell: center)
                for k in centerRange {
                    let j = Int(grid.pointIndices[k])
                    let d2 = simd_length_squared(points[j] - q)
                    if d2 < bestDistance {
                        bestDistance = d2
                        bestIndex = j
                    }
                }
                var dz: Int32 = -reach
                while dz <= reach {
                    var dy: Int32 = -reach
                    while dy <= reach {
                        var dx: Int32 = -reach
                        while dx <= reach {
                            if dx == 0, dy == 0, dz == 0 {
                                dx += 1
                                continue
                            }
                            let cell = center &+ SIMD3<Int32>(dx, dy, dz)
                            let lo = SIMD3<Float>(Float(cell.x), Float(cell.y), Float(cell.z)) * cellSize
                            let gap = simd_max(simd_max(lo - q, q - lo - SIMD3<Float>(repeating: cellSize)),
                                               SIMD3<Float>.zero)
                            if simd_length_squared(gap) >= bestDistance {
                                dx += 1
                                continue
                            }
                            let range = grid.pointIndexRange(inCell: cell)
                            for k in range {
                                let j = Int(grid.pointIndices[k])
                                let d2 = simd_length_squared(points[j] - q)
                                if d2 < bestDistance {
                                    bestDistance = d2
                                    bestIndex = j
                                }
                            }
                            dx += 1
                        }
                        dy += 1
                    }
                    dz += 1
                }
                guard bestIndex >= 0 else { return nil }
                return (points[bestIndex], normals[bestIndex])
            }
        }

        /// 借用ビューの有効期間は `body` 内のみ（escape 禁止）。
        func withUnsafeView<R>(_ body: (UnsafeView) -> R) -> R {
            points.withUnsafeBufferPointer { pointsPtr in
                normals.withUnsafeBufferPointer { normalsPtr in
                    grid.withUnsafeView { gridView in
                        body(UnsafeView(points: pointsPtr, normals: normalsPtr, grid: gridView))
                    }
                }
            }
        }

        /// 法線推定の近傍はセルごとに上限を設けた層化サンプリングで集める。
        /// 「先頭 N 点で打ち切り」はセル走査順の空間バイアスで法線が歪み、
        /// MLS の収縮方向が壊れて表面がクラスト状になる（改善ループで実測）。
        /// 点ごとに独立なので固定長チャンクで並列化する（結果は決定的）。
        private static func estimatedNormals(points: [SIMD3<Float>],
                                             grid: CompactUniformGrid,
                                             radius: Float) -> [SIMD3<Float>] {
            var normals = [SIMD3<Float>](repeating: SIMD3<Float>(0, 0, 1), count: points.count)
            guard !points.isEmpty else { return normals }
            let r2 = radius * radius
            let perCellCap = 4
            let chunkSize = 2048
            let chunkCount = (points.count + chunkSize - 1) / chunkSize
            points.withUnsafeBufferPointer { pointsPtr in
                normals.withUnsafeMutableBufferPointer { out in
                    grid.withUnsafeView { gridView in
                        DispatchQueue.concurrentPerform(iterations: chunkCount) { chunk in
                            var neighbors: [SIMD3<Float>] = []
                            neighbors.reserveCapacity(108)
                            let lower = chunk * chunkSize
                            let upper = min(pointsPtr.count, lower + chunkSize)
                            for i in lower..<upper {
                                neighbors.removeAll(keepingCapacity: true)
                                let center = gridView.cellCoordinate(of: pointsPtr[i])
                                for dz in Int32(-1)...1 {
                                    for dy in Int32(-1)...1 {
                                        for dx in Int32(-1)...1 {
                                            let range = gridView.pointIndexRange(inCell: center &+ SIMD3<Int32>(dx, dy, dz))
                                            var taken = 0
                                            for k in range {
                                                let j = Int(gridView.pointIndices[k])
                                                if simd_length_squared(pointsPtr[j] - pointsPtr[i]) <= r2 {
                                                    neighbors.append(pointsPtr[j])
                                                    taken += 1
                                                    if taken >= perCellCap { break }
                                                }
                                            }
                                        }
                                    }
                                }
                                if neighbors.count >= 4, let normal = planeNormal(of: neighbors) {
                                    out[i] = normal.z < 0 ? -normal : normal
                                }
                            }
                        }
                    }
                }
            }
            return normals
        }

        /// 局所平面フィット法線（共分散の最小固有ベクトル）。
        /// 行クロス積近似は殻厚のあるノイズ雲で法線がジッタし MLS を壊すため、
        /// Jacobi 回転による厳密対角化を使う（対称 3x3、数スイープで収束）。
        static func planeNormal(of neighbors: [SIMD3<Float>]) -> SIMD3<Float>? {
            guard neighbors.count >= 3 else { return nil }
            var mean = SIMD3<Float>.zero
            for p in neighbors { mean += p }
            mean /= Float(neighbors.count)
            var a = [Double](repeating: 0, count: 9)  // 行優先 3x3 共分散
            for p in neighbors {
                let d = SIMD3<Double>(Double(p.x - mean.x), Double(p.y - mean.y), Double(p.z - mean.z))
                a[0] += d.x * d.x; a[1] += d.x * d.y; a[2] += d.x * d.z
                a[4] += d.y * d.y; a[5] += d.y * d.z; a[8] += d.z * d.z
            }
            a[3] = a[1]; a[6] = a[2]; a[7] = a[5]
            var v = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]  // 固有ベクトル（列）
            for _ in 0..<12 {
                // 最大の非対角成分を選んで回転で消す
                var p = 0, q = 1
                var maxOff = abs(a[1])
                if abs(a[2]) > maxOff { maxOff = abs(a[2]); p = 0; q = 2 }
                if abs(a[5]) > maxOff { maxOff = abs(a[5]); p = 1; q = 2 }
                if maxOff < 1e-15 { break }
                let app = a[p * 3 + p], aqq = a[q * 3 + q], apq = a[p * 3 + q]
                let theta = 0.5 * atan2(2 * apq, aqq - app)
                let c = cos(theta), s = sin(theta)
                for k in 0..<3 {
                    let akp = a[k * 3 + p], akq = a[k * 3 + q]
                    a[k * 3 + p] = c * akp - s * akq
                    a[k * 3 + q] = s * akp + c * akq
                }
                for k in 0..<3 {
                    let apk = a[p * 3 + k], aqk = a[q * 3 + k]
                    a[p * 3 + k] = c * apk - s * aqk
                    a[q * 3 + k] = s * apk + c * aqk
                }
                for k in 0..<3 {
                    let vkp = v[k * 3 + p], vkq = v[k * 3 + q]
                    v[k * 3 + p] = c * vkp - s * vkq
                    v[k * 3 + q] = s * vkp + c * vkq
                }
            }
            var minColumn = 0
            var minValue = a[0]
            if a[4] < minValue { minValue = a[4]; minColumn = 1 }
            if a[8] < minValue { minValue = a[8]; minColumn = 2 }
            let normal = SIMD3<Float>(Float(v[minColumn]), Float(v[3 + minColumn]), Float(v[6 + minColumn]))
            let length = simd_length(normal)
            guard length > 1e-9 else { return nil }
            return normal / length
        }
    }

    struct IncrementalOrientedPointSet {
        private struct VoxelAccumulator {
            var sum: SIMD3<Float>
            var count: Int
            var pointIndex: Int
        }

        private let voxelCell: Float
        private let normalRadius: Float
        private let bucketCell: Float
        private var voxels: [Cell: VoxelAccumulator] = [:]
        private var points: [SIMD3<Float>] = []
        private var normals: [SIMD3<Float>] = []
        private var buckets: [Cell: [Int32]] = [:]
        private var bucketByPoint: [Cell] = []
        private var pendingNormalIndices: Set<Int> = []

        init(voxelCell: Float, normalRadius: Float) {
            self.voxelCell = max(voxelCell, 1e-6)
            self.normalRadius = max(normalRadius, 1e-6)
            self.bucketCell = max(normalRadius, 1e-4)
        }

        /// 法線再計算はバッチで触れた点（新規＋ボクセル平均が動いた点）と、前回近傍不足で
        /// 計算できなかった点だけに限定する。近傍点の法線は stale を許容する:
        /// ボクセル平均の移動は深度ノイズ未満で、ICP は Tukey 重み付き point-to-plane
        /// なので影響せず、最終品質は refinement round のモデル完全再構築で決まる。
        /// （旧実装は半径内の全近傍を毎フレーム再計算し、フレーム数×モデル全点の O(F²) だった。）
        mutating func insert(_ newPoints: [SIMD3<Float>]) {
            guard !newPoints.isEmpty else { return }
            var affected = pendingNormalIndices
            for point in newPoints {
                let voxel = Cell(point, cell: voxelCell)
                if var accumulator = voxels[voxel] {
                    accumulator.sum += point
                    accumulator.count += 1
                    let updatedPoint = accumulator.sum / Float(accumulator.count)
                    points[accumulator.pointIndex] = updatedPoint
                    voxels[voxel] = accumulator
                    updateBucket(for: accumulator.pointIndex, to: updatedPoint)
                    affected.insert(accumulator.pointIndex)
                } else {
                    let index = points.count
                    points.append(point)
                    normals.append(SIMD3<Float>(0, 0, 1))
                    let bucket = Cell(point, cell: bucketCell)
                    bucketByPoint.append(bucket)
                    buckets[bucket, default: []].append(Int32(index))
                    voxels[voxel] = VoxelAccumulator(sum: point, count: 1, pointIndex: index)
                    affected.insert(index)
                }
            }
            recomputeNormals(indices: affected)
        }

        func snapshot() -> OrientedPointSet {
            OrientedPointSet(points: points, normals: normals, cellSize: bucketCell)
        }

        private mutating func updateBucket(for index: Int, to newPoint: SIMD3<Float>) {
            let oldBucket = bucketByPoint[index]
            let newBucket = Cell(newPoint, cell: bucketCell)
            guard oldBucket != newBucket else { return }
            if var indices = buckets[oldBucket] {
                indices.removeAll { $0 == Int32(index) }
                if indices.isEmpty {
                    buckets.removeValue(forKey: oldBucket)
                } else {
                    buckets[oldBucket] = indices
                }
            }
            buckets[newBucket, default: []].append(Int32(index))
            bucketByPoint[index] = newBucket
        }

        /// 近傍不足（< 4 点）で法線を確定できなかった点は `pendingNormalIndices` に残し、
        /// 次回 insert で再試行する（疎なフロンティア点が placeholder 法線 (0,0,1) の
        /// まま取り残されるのを防ぐ。密度が上がると集合は速やかに空になる）。
        private mutating func recomputeNormals(indices: Set<Int>) {
            guard !indices.isEmpty else { return }
            var neighbors: [SIMD3<Float>] = []
            neighbors.reserveCapacity(108)
            let perCellCap = 4
            let r2 = normalRadius * normalRadius
            for index in indices where index < points.count {
                neighbors.removeAll(keepingCapacity: true)
                let center = Cell(points[index], cell: bucketCell)
                for dz in -1...1 {
                    for dy in -1...1 {
                        for dx in -1...1 {
                            let key = Cell(x: center.x + Int32(dx), y: center.y + Int32(dy), z: center.z + Int32(dz))
                            guard let indices = buckets[key] else { continue }
                            var taken = 0
                            for rawIndex in indices where simd_length_squared(points[Int(rawIndex)] - points[index]) <= r2 {
                                neighbors.append(points[Int(rawIndex)])
                                taken += 1
                                if taken >= perCellCap { break }
                            }
                        }
                    }
                }
                if neighbors.count >= 4, let normal = OrientedPointSet.planeNormal(of: neighbors) {
                    normals[index] = normal.z < 0 ? -normal : normal
                    pendingNormalIndices.remove(index)
                } else {
                    pendingNormalIndices.insert(index)
                }
            }
        }
    }

    static func voxelAveraged(_ points: [SIMD3<Float>], cell: Float) -> [SIMD3<Float>] {
        guard cell > 0 else { return points }
        var sums: [Cell: (SIMD3<Float>, Int)] = [:]
        sums.reserveCapacity(points.count / 4 + 1)
        for p in points {
            let key = Cell(p, cell: cell)
            if var entry = sums[key] {
                entry.0 += p
                entry.1 += 1
                sums[key] = entry
            } else {
                sums[key] = (p, 1)
            }
        }
        // Dictionary の走査順を後段へ渡すと、プロセスごとのランダムなハッシュ seed で
        // MLS のセル内打ち切り・ICP の stride サンプルが変わり、同じスキャンでも判定値が
        // 揺れる。セル座標順へ固定し、点列まで入力データだけで決まるようにする。
        return sums.keys.sorted { lhs, rhs in
            if lhs.x != rhs.x { return lhs.x < rhs.x }
            if lhs.y != rhs.y { return lhs.y < rhs.y }
            return lhs.z < rhs.z
        }.map { key in
            let entry = sums[key]!
            return entry.0 / Float(entry.1)
        }
    }

    private static func subsampled(_ points: [SIMD3<Float>], target: Int) -> [SIMD3<Float>] {
        guard points.count > target, target > 0 else { return points }
        let step = max(1, points.count / target)
        var result: [SIMD3<Float>] = []
        result.reserveCapacity(target + 1)
        var i = 0
        while i < points.count {
            result.append(points[i])
            i += step
        }
        return result
    }

    // MARK: - 4. 円筒フュージョン

    struct CylindricalFusionResult {
        var mesh: FaceMesh
        var columns: Int
        var rows: Int
        var measuredBins: Int
    }

    static func cylindricalMesh(points: [SIMD3<Float>],
                                configuration: Configuration) throws -> CylindricalFusionResult? {
        guard points.count > 100 else { return nil }
        guard let cylindrical = CylindricalFaceFrame.samples(
            points: points,
            binThetaDegrees: configuration.binThetaDegrees,
            binHeight: configuration.binHeight,
            bounds: .percentile(low: 0.002, high: 0.998)
        ) else {
            return nil
        }
        let frame = cylindrical.frame
        let thetas = cylindrical.thetas
        let radii = cylindrical.radii
        let dY = frame.binHeight
        let columns = frame.columns
        let rows = frame.rows
        guard columns > 4, rows > 4, columns * rows < 4_000_000 else { return nil }

        // ビンへ集約
        var bins = [[Float]](repeating: [], count: columns * rows)
        for (i, p) in points.enumerated() {
            bins[frame.binIndex(theta: thetas[i], y: p.y)].append(radii[i])
        }

        var grid = [Float](repeating: .nan, count: columns * rows)
        var measuredBins = 0
        for i in bins.indices {
            var samples = bins[i]
            guard !samples.isEmpty else { continue }
            samples.sort()
            let anchor = samples[min(samples.count - 1,
                                     Int(Float(samples.count - 1) * configuration.binAnchorPercentile))]
            var sum: Float = 0
            var count = 0
            for r in samples where abs(r - anchor) <= configuration.binBandWidth {
                sum += r
                count += 1
            }
            guard count > 0 else { continue }
            grid[i] = sum / Float(count)
            measuredBins += 1
        }
        guard measuredBins > 500 else { return nil }

        // 拡散充填（充填深さを記録）
        let filled = try MeasuredCylindricalGridProcessor.diffusedFill(
            grid,
            columns: columns,
            rows: rows,
            maxIterations: 40
        )
        grid = filled.grid
        let fillDepth = filled.fillDepth

        // エッジ保存平滑化（残留テラスの除去）
        grid = try MeasuredCylindricalGridProcessor.smoothedGrid(
            grid,
            columns: columns,
            rows: rows,
            iterations: max(0, configuration.gridSmoothIterations),
            radius: configuration.gridSmoothRadius,
            sigmaS: configuration.gridSmoothSigmaS,
            sigmaR: configuration.gridSmoothSigmaR
        )

        // 有効ビンマスク: 充填深さ上限 → 開処理 → 最大連結領域
        var keep = [Bool](repeating: false, count: grid.count)
        for i in grid.indices {
            keep[i] = grid[i].isFinite && fillDepth[i] >= 0 && fillDepth[i] <= configuration.maximumFillDepth
        }
        keep = try openedMask2D(keep, columns: columns, rows: rows,
                                iterations: configuration.maskOpenIterations)
        keep = largestRegion2D(keep, columns: columns, rows: rows)
        keep = fillingInteriorHoles2D(keep, grid: grid, columns: columns, rows: rows)

        // グリッド三角形化
        var vertexIndex = [Int](repeating: -1, count: grid.count)
        var vertices: [SIMD3<Float>] = []
        for y in 0..<rows {
            for x in 0..<columns {
                let i = y * columns + x
                guard keep[i] else { continue }
                let theta = frame.thetaOrigin + Float(x) * frame.binTheta
                let radius = grid[i]
                vertexIndex[i] = vertices.count
                vertices.append(SIMD3<Float>(
                    frame.worldX(radius: radius, theta: theta),
                    frame.yOrigin + Float(y) * dY,
                    frame.worldZ(radius: radius, theta: theta)
                ))
            }
        }
        var indices: [UInt32] = []
        for y in 0..<(rows - 1) {
            for x in 0..<(columns - 1) {
                let a = vertexIndex[y * columns + x]
                let b = vertexIndex[y * columns + x + 1]
                let c = vertexIndex[(y + 1) * columns + x]
                let d = vertexIndex[(y + 1) * columns + x + 1]
                if a >= 0, b >= 0, c >= 0 {
                    indices.append(UInt32(a)); indices.append(UInt32(b)); indices.append(UInt32(c))
                }
                if b >= 0, c >= 0, d >= 0 {
                    indices.append(UInt32(b)); indices.append(UInt32(d)); indices.append(UInt32(c))
                }
            }
        }
        guard vertices.count > 100, indices.count >= 3 else { return nil }
        return CylindricalFusionResult(
            mesh: FaceMesh(
                vertices: vertices,
                triangleIndices: indices
            ),
            columns: columns,
            rows: rows,
            measuredBins: measuredBins
        )
    }

    static func openedMask2D(_ mask: [Bool], columns: Int, rows: Int, iterations: Int) throws -> [Bool] {
        try MeasuredCylindricalGridProcessor.openedMask2D(
            mask,
            columns: columns,
            rows: rows,
            iterations: iterations
        )
    }

    /// keep 領域の内部に閉じ込められた非 keep ビン（=メッシュの内部穴になる）を、
    /// 半径値が有限なら keep へ戻す。外周（グリッド縁に到達できる非 keep 領域）は残す。
    static func fillingInteriorHoles2D(_ mask: [Bool], grid: [Float],
                                       columns: Int, rows: Int) -> [Bool] {
        var outside = [Bool](repeating: false, count: mask.count)
        var stack: [Int] = []
        for x in 0..<columns {
            for y in [0, rows - 1] {
                let i = y * columns + x
                if !mask[i], !outside[i] { outside[i] = true; stack.append(i) }
            }
        }
        for y in 0..<rows {
            for x in [0, columns - 1] {
                let i = y * columns + x
                if !mask[i], !outside[i] { outside[i] = true; stack.append(i) }
            }
        }
        while let i = stack.popLast() {
            let y = i / columns
            let x = i % columns
            let candidates = [
                y > 0 ? i - columns : -1,
                y < rows - 1 ? i + columns : -1,
                x > 0 ? i - 1 : -1,
                x < columns - 1 ? i + 1 : -1
            ]
            for j in candidates where j >= 0 && !mask[j] && !outside[j] {
                outside[j] = true
                stack.append(j)
            }
        }
        var out = mask
        for i in mask.indices where !mask[i] && !outside[i] && grid[i].isFinite {
            out[i] = true
        }
        return out
    }

    static func largestRegion2D(_ mask: [Bool], columns: Int, rows: Int) -> [Bool] {
        var visited = [Bool](repeating: false, count: mask.count)
        var best: [Int] = []
        var stack: [Int] = []
        for start in mask.indices where mask[start] && !visited[start] {
            stack.removeAll(keepingCapacity: true)
            stack.append(start)
            visited[start] = true
            var component: [Int] = []
            while let i = stack.popLast() {
                component.append(i)
                let y = i / columns
                let x = i % columns
                let candidates = [
                    y > 0 ? i - columns : -1,
                    y < rows - 1 ? i + columns : -1,
                    x > 0 ? i - 1 : -1,
                    x < columns - 1 ? i + 1 : -1
                ]
                for j in candidates where j >= 0 && mask[j] && !visited[j] {
                    visited[j] = true
                    stack.append(j)
                }
            }
            if component.count > best.count { best = component }
        }
        var out = [Bool](repeating: false, count: mask.count)
        for i in best { out[i] = true }
        return out
    }
}
