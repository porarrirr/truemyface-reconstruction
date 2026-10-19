import Foundation
import simd

/// 点群の一様グリッド索引（オープンアドレッシングのハッシュテーブル + counting sort）。
/// 旧 `[Cell: [Int32]]` Dictionary 索引の置き換え。
///
/// - 疎表現なので、クロップ前の点群に外れ値（マスク縁の背景画素など）が混ざって
///   AABB が数 m に広がってもメモリ使用量は点数にしか比例しない
///   （密グリッドだと最細スケールでセル数が爆発する）。
/// - セル内の点順序は入力順を保存する（安定 scatter）ため、perCellCap 打ち切りの
///   近傍集合・順序が旧実装の「挿入順 append」セマンティクスと一致する。
/// - 構築・探索とも決定的（点列とセルサイズだけで結果が決まる）。
struct CompactUniformGrid {
    let cellSize: Float
    /// セル順に整列した点インデックス（セル内は入力順）。
    let pointIndices: [UInt32]
    /// スロット → セルキー（空きスロットは `emptyKey`）。サイズは 2 の冪。
    private let keys: [UInt64]
    /// スロット → `pointIndices` の開始位置（要素数 = スロット数 + 1）。
    private let slotStart: [UInt32]
    private let slotMask: Int

    private static let emptyKey = UInt64.max

    /// 旧 `Cell` と同一の量子化（floor(p / cellSize)）。
    @inline(__always)
    static func cellCoordinate(of p: SIMD3<Float>, cellSize: Float) -> SIMD3<Int32> {
        SIMD3<Int32>(Int32((p.x / cellSize).rounded(.down)),
                     Int32((p.y / cellSize).rounded(.down)),
                     Int32((p.z / cellSize).rounded(.down)))
    }

    /// セル座標を 21bit x 3 に詰める（±2^20 セル ≒ セル 1mm でも ±1km をカバー）。
    @inline(__always)
    private static func packedKey(_ c: SIMD3<Int32>) -> UInt64 {
        let bias: Int64 = 1 << 20
        let x = UInt64(bitPattern: Int64(c.x) + bias) & 0x1F_FFFF
        let y = UInt64(bitPattern: Int64(c.y) + bias) & 0x1F_FFFF
        let z = UInt64(bitPattern: Int64(c.z) + bias) & 0x1F_FFFF
        return x | (y << 21) | (z << 42)
    }

    /// splitmix64 finalizer。
    @inline(__always)
    private static func hashed(_ key: UInt64) -> UInt64 {
        var x = key
        x ^= x >> 33
        x &*= 0xff51_afd7_ed55_8ccd
        x ^= x >> 33
        x &*= 0xc4ce_b9fe_1a85_ec53
        x ^= x >> 33
        return x
    }

    init(points: [SIMD3<Float>], cellSize: Float) {
        precondition(cellSize > 0, "cellSize must be positive")
        self.cellSize = cellSize
        guard !points.isEmpty else {
            pointIndices = []
            keys = []
            slotStart = [0]
            slotMask = 0
            return
        }

        // 負荷率 ≤ 0.5（ユニークセル数 ≤ 点数 ≤ テーブル/2）で線形探査を短く保つ。
        var tableSize = 64
        while tableSize < points.count * 2 {
            tableSize <<= 1
        }
        let mask = tableSize - 1
        var keys = [UInt64](repeating: Self.emptyKey, count: tableSize)
        var counts = [UInt32](repeating: 0, count: tableSize)
        var slotOfPoint = [Int32](repeating: 0, count: points.count)
        for i in points.indices {
            let key = Self.packedKey(Self.cellCoordinate(of: points[i], cellSize: cellSize))
            var slot = Int(Self.hashed(key) & UInt64(mask))
            while true {
                let existing = keys[slot]
                if existing == key { break }
                if existing == Self.emptyKey {
                    keys[slot] = key
                    break
                }
                slot = (slot + 1) & mask
            }
            slotOfPoint[i] = Int32(slot)
            counts[slot] += 1
        }
        var start = [UInt32](repeating: 0, count: tableSize + 1)
        for s in 0..<tableSize {
            start[s + 1] = start[s] + counts[s]
        }
        var cursor = start
        var indices = [UInt32](repeating: 0, count: points.count)
        for i in points.indices {
            let s = Int(slotOfPoint[i])
            indices[Int(cursor[s])] = UInt32(i)
            cursor[s] += 1
        }
        self.keys = keys
        self.slotStart = start
        self.pointIndices = indices
        self.slotMask = mask
    }

    /// ARC を伴わない借用ビュー。ホットループでは必ずこちらを使う。
    /// Array の要素アクセスやプロパティ参照は（特に -Onone で）アクセスごとに
    /// retain/release を発行し、並列時は参照カウントのアトミック競合で
    /// 直列実装より遅くなる（改善ループの sample で実測）。
    struct UnsafeView {
        let cellSize: Float
        fileprivate let slotMask: Int
        fileprivate let keys: UnsafeBufferPointer<UInt64>
        fileprivate let slotStart: UnsafeBufferPointer<UInt32>
        let pointIndices: UnsafeBufferPointer<UInt32>

        @inline(__always)
        func cellCoordinate(of p: SIMD3<Float>) -> SIMD3<Int32> {
            CompactUniformGrid.cellCoordinate(of: p, cellSize: cellSize)
        }

        /// セル座標に対応する `pointIndices` の範囲（存在しないセルは空 Range）。
        @inline(__always)
        func pointIndexRange(inCell c: SIMD3<Int32>) -> Range<Int> {
            guard !keys.isEmpty else { return 0..<0 }
            let key = CompactUniformGrid.packedKey(c)
            var slot = Int(CompactUniformGrid.hashed(key) & UInt64(slotMask))
            while true {
                let existing = keys[slot]
                if existing == key {
                    return Int(slotStart[slot])..<Int(slotStart[slot + 1])
                }
                if existing == CompactUniformGrid.emptyKey {
                    return 0..<0
                }
                slot = (slot + 1) & slotMask
            }
        }
    }

    /// 借用ビューの有効期間は `body` 内のみ（escape 禁止）。
    func withUnsafeView<R>(_ body: (UnsafeView) -> R) -> R {
        keys.withUnsafeBufferPointer { keysPtr in
            slotStart.withUnsafeBufferPointer { startPtr in
                pointIndices.withUnsafeBufferPointer { indicesPtr in
                    body(UnsafeView(cellSize: cellSize,
                                    slotMask: slotMask,
                                    keys: keysPtr,
                                    slotStart: startPtr,
                                    pointIndices: indicesPtr))
                }
            }
        }
    }
}
