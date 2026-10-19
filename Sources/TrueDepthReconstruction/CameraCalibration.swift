import Foundation
import simd

/// 深度マップの内部パラメータ（焦点距離・主点）。
///
/// `AVDepthData.cameraCalibrationData.intrinsicMatrix` は
/// `intrinsicMatrixReferenceDimensions`（多くは元映像の解像度）基準で定義されるため、
/// 実際の深度バッファ解像度へ **スケールしてから** 逆投影に使う。
/// `ARFrame.camera.intrinsics`（カラー解像度基準）とは混在させないこと（座標がズレる）。
public struct CameraIntrinsics: Codable, Equatable {
    public var fx: Float
    public var fy: Float
    public var cx: Float
    public var cy: Float

    public init(fx: Float, fy: Float, cx: Float, cy: Float) {
        self.fx = fx; self.fy = fy; self.cx = cx; self.cy = cy
    }

    /// 参照解像度の内部パラメータ行列を、対象（深度バッファ）解像度へスケールする。
    public static func scaled(matrix K: simd_float3x3,
                       referenceSize: SIMD2<Float>,
                       targetSize: SIMD2<Float>) -> CameraIntrinsics {
        let sx = targetSize.x / max(referenceSize.x, 1)
        let sy = targetSize.y / max(referenceSize.y, 1)
        return CameraIntrinsics(
            fx: K[0][0] * sx,
            fy: K[1][1] * sy,
            cx: K[2][0] * sx,
            cy: K[2][1] * sy
        )
    }
}

/// AVFoundation の半径方向レンズ歪み補正。
///
/// 方向は AVCameraCalibrationData.h の参照実装どおり:
/// `lensDistortionLookupTable` は rectified → observed（理想像へ歪みを再適用）、
/// `inverseLensDistortionLookupTable` は observed → rectified（観測画素の歪み除去）。
/// `usesInverseTable` は保持しているテーブルが後者であることを示す。
public struct LensDistortionModel: Equatable {
    public var center: SIMD2<Float>
    /// rectified -> observed（Apple の `lensDistortionLookupTable`）。
    public var forwardTable: [Float]?
    /// observed -> rectified（Apple の `inverseLensDistortionLookupTable`）。
    public var inverseTable: [Float]?
    /// LUT が定義される参照座標系での最大半径。
    public var maxRadius: Float
    /// 対象画像の画素差分を LUT 参照座標の差分へ変換する倍率。
    /// 旧 export は対象画像座標で LUT を評価していたため、欠落時は (1, 1)。
    public var targetToLookupScale: SIMD2<Float>
    /// 再現・診断用。AVFoundation が校正値を定義した参照画像寸法。
    public var referenceDimensions: SIMD2<Float>?
    /// 再現・診断用。実際に LUT を適用する深度画像寸法。
    public var imageDimensions: SIMD2<Float>?

    /// 既存データセット／Metal ABI 用の代表テーブル。逆投影のホットパス
    /// （observed → rectified）で直接適用できる inverse を優先し、
    /// 存在しない場合だけ forward を返す。
    public var table: [Float] { inverseTable ?? forwardTable ?? [] }
    public var usesInverseTable: Bool { inverseTable != nil }

    /// 歪みなし（座標不変）。
    public static let identity = LensDistortionModel(
        center: .zero,
        forwardTable: [],
        inverseTable: [],
        maxRadius: 1,
        targetToLookupScale: SIMD2<Float>(repeating: 1),
        referenceDimensions: nil,
        imageDimensions: nil
    )



    /// テスト・データセット復元用の1D半径テーブル。
    public init(center: SIMD2<Float>, table: [Float], maxRadius: Float, usesInverseTable: Bool = true) {
        self.center = center
        self.maxRadius = maxRadius
        targetToLookupScale = SIMD2<Float>(repeating: 1)
        referenceDimensions = nil
        imageDimensions = nil
        forwardTable = usesInverseTable ? nil : table
        inverseTable = usesInverseTable ? table : nil
    }

    /// 両方の Apple LUT を保持するテスト／復元用初期化子。
    public init(center: SIMD2<Float>,
         forwardTable: [Float]?,
         inverseTable: [Float]?,
         maxRadius: Float,
         targetToLookupScale: SIMD2<Float> = SIMD2<Float>(repeating: 1),
         referenceDimensions: SIMD2<Float>? = nil,
         imageDimensions: SIMD2<Float>? = nil) {
        self.center = center
        self.forwardTable = forwardTable
        self.inverseTable = inverseTable
        self.maxRadius = maxRadius
        self.targetToLookupScale = targetToLookupScale
        self.referenceDimensions = referenceDimensions
        self.imageDimensions = imageDimensions
    }

    /// 観測画素を歪み補正済み画素へ変換する。
    /// Apple の参照実装どおり inverse LUT を直接適用する（入力は観測画素なので
    /// 半径は常にテーブル定義域 [0, maxRadius] 内）。forward のみの入力は数値反転。
    public func rectifiedPixel(px: Float, py: Float) -> SIMD2<Float> {
        guard maxRadius > 0 else {
            return SIMD2(px, py)
        }

        let point = SIMD2(px, py)
        if let inverseTable, !inverseTable.isEmpty {
            return applyRadialMagnification(to: point, table: inverseTable)
        }
        if let forwardTable, !forwardTable.isEmpty {
            return invertedRadialMagnification(of: point, table: forwardTable)
        }
        return point
    }

    /// 歪み補正済み（rectified）画素を観測（distorted）画素へ変換する。
    /// `rectifiedPixel` の逆方向で、3D 点を実際の深度バッファ画素へ投影するときに使う。
    /// これを通さずに理想ピンホール画素で深度をサンプルすると、`unproject` 側で再度
    /// rectification が掛かり、画面端（頬・顎・輪郭）で系統的な数 mm のズレになる。
    ///
    /// forward LUT の定義域は rectified 半径 [0, maxRadius]。強い樽型歪みでは
    /// rectified 座標が maxRadius を超えるため、その領域は末尾値クランプではなく
    /// inverse LUT の二分反転で求める（`rectifiedPixel` と厳密に往復一致する）。
    public func distortedPixel(px: Float, py: Float) -> SIMD2<Float> {
        guard maxRadius > 0 else {
            return SIMD2(px, py)
        }

        let point = SIMD2(px, py)
        let hasInverse = !(inverseTable?.isEmpty ?? true)
        if let forwardTable, !forwardTable.isEmpty,
           lookupRadius(for: point - center) <= maxRadius || !hasInverse {
            return applyRadialMagnification(to: point, table: forwardTable)
        }
        if let inverseTable, hasInverse {
            return invertedRadialMagnification(of: point, table: inverseTable)
        }
        return point
    }

    /// 観測画素を rectified へ変換し、再び観測画素へ戻したときの誤差（px）。
    /// project / unproject が同じ座標系で round-trip できているかの検証に使う。
    public func roundTripPixelError(px: Float, py: Float) -> Float {
        let rectified = rectifiedPixel(px: px, py: py)
        let restored = distortedPixel(px: rectified.x, py: rectified.y)
        return simd_length(SIMD2(px, py) - restored)
    }

    /// 歪み補正済み画素と深度からカメラ光学空間の点へ逆投影する。
    public static func unproject(px: Float,
                          py: Float,
                          z: Float,
                          intrinsics k: CameraIntrinsics,
                          lens: LensDistortionModel?) -> SIMD3<Float> {
        let rectified: SIMD2<Float>
        if let lens {
            rectified = lens.rectifiedPixel(px: px, py: py)
        } else {
            rectified = SIMD2(px, py)
        }
        return unprojectRectified(px: rectified.x, py: rectified.y, z: z, intrinsics: k)
    }

    /// 補正済み画素 `(px, py)` と深度 `z`（メートル, 前方正）をカメラ光学空間の点へ逆投影する。
    @inline(__always)
    public static func unprojectRectified(px: Float, py: Float, z: Float, intrinsics k: CameraIntrinsics) -> SIMD3<Float> {
        let x = (px - k.cx) * z / k.fx
        let y = (py - k.cy) * z / k.fy
        return SIMD3<Float>(x, y, z)
    }

    /// 光学系の点を ARKit カメラ空間へ変換（x 軸まわり180°回転）。
    @inline(__always)
    public static func opticalToARKitCamera(_ p: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3<Float>(p.x, -p.y, -p.z)
    }

    /// ARKit カメラ空間の点を光学系へ変換。
    @inline(__always)
    public static func arkitCameraToOptical(_ p: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3<Float>(p.x, -p.y, -p.z)
    }

    /// 光学系の 3D 点を画素座標へ投影する（歪み補正前の観測画素）。
    public static func projectToPixel(_ optical: SIMD3<Float>, intrinsics k: CameraIntrinsics) -> SIMD2<Float>? {
        projectToPixel(optical, intrinsics: k, lens: nil)
    }

    /// 光学系の 3D 点を**観測（distorted）画素**へ投影する。
    /// ピンホールで rectified 画素を求めたあと、レンズの順方向歪みを適用して観測画素に戻す。
    /// 深度バッファは観測座標なので、サンプルはこの観測画素で行う必要がある。
    public static func projectToPixel(_ optical: SIMD3<Float>,
                               intrinsics k: CameraIntrinsics,
                               lens: LensDistortionModel?) -> SIMD2<Float>? {
        guard optical.z > 1e-6 else { return nil }
        let rectifiedPx = k.fx * optical.x / optical.z + k.cx
        let rectifiedPy = k.fy * optical.y / optical.z + k.cy
        guard let lens else { return SIMD2(rectifiedPx, rectifiedPy) }
        return lens.distortedPixel(px: rectifiedPx, py: rectifiedPy)
    }

    // MARK: - Private



    private static func interpolate(table: [Float], at normalized: Float) -> Float {
        guard !table.isEmpty else { return normalized }
        if table.count == 1 { return table[0] }
        let clamped = max(0, min(1, normalized))
        let scaled = clamped * Float(table.count - 1)
        let lower = Int(floor(scaled))
        let upper = min(lower + 1, table.count - 1)
        let fraction = scaled - Float(lower)
        return table[lower] * (1 - fraction) + table[upper] * fraction
    }

    private func applyRadialMagnification(to point: SIMD2<Float>, table: [Float]) -> SIMD2<Float> {
        let delta = point - center
        let radius = lookupRadius(for: delta)
        guard radius > 1e-6 else { return point }
        let normalized = min(radius / maxRadius, 1)
        let magnification = Self.interpolate(table: table, at: normalized)
        return center + delta * (1 + magnification)
    }

    /// 半径写像 `r' = r * (1 + LUT(r))` を二分探索で反転する。
    /// iPad の外周では magnification が +0.5 を超えるため、固定回数の不動点反復では
    /// 画素単位の誤差が残る。AVFoundation のテーブルが単調な半径写像であることを使い、
    /// 強い歪みでも決定的に収束させる。
    private func invertedRadialMagnification(of target: SIMD2<Float>, table: [Float]) -> SIMD2<Float> {
        let delta = target - center
        let targetRadius = lookupRadius(for: delta)
        guard targetRadius > 1e-6 else { return target }

        guard let lastMagnification = table.last,
              lastMagnification.isFinite,
              1 + lastMagnification > 0,
              maxRadius.isFinite,
              maxRadius > 0 else {
            return SIMD2<Float>(repeating: .nan)
        }

        func mappedRadius(_ radius: Float) -> Float {
            let normalized = min(radius / maxRadius, 1)
            return radius * (1 + Self.interpolate(table: table, at: normalized))
        }

        let mappedMaximumRadius = mappedRadius(maxRadius)
        if targetRadius > mappedMaximumRadius {
            // interpolate は範囲外で末尾値を保持するため、この領域は厳密な一次式。
            let sourceRadius = targetRadius / (1 + lastMagnification)
            return center + delta * (sourceRadius / targetRadius)
        }

        var lower: Float = 0
        var upper = maxRadius
        let lookupTolerance = max(
            1e-6,
            1e-4 * min(targetToLookupScale.x, targetToLookupScale.y)
        )
        var sourceRadius = (lower + upper) * 0.5
        var converged = false
        for _ in 0..<32 {
            let midpoint = (lower + upper) * 0.5
            let mapped = mappedRadius(midpoint)
            sourceRadius = midpoint
            if abs(mapped - targetRadius) <= lookupTolerance || upper - lower <= lookupTolerance {
                converged = true
                break
            }
            if mapped < targetRadius {
                lower = midpoint
            } else {
                upper = midpoint
            }
        }
        guard converged else { return SIMD2<Float>(repeating: .nan) }
        return center + delta * (sourceRadius / targetRadius)
    }

    @inline(__always)
    private func lookupRadius(for targetDelta: SIMD2<Float>) -> Float {
        simd_length(targetDelta * targetToLookupScale)
    }
}
