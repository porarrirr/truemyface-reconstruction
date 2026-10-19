import Foundation
import simd

/// A two-dimensional image ROI limits connected-component depth segmentation.
/// Supplies no 3D template, symmetry constraint, or imputed facial surface.
enum DepthForegroundMask {
    /// Input coordinates are normalized, bottom-left Vision coordinates of a portrait image
    /// rotated 90 degrees clockwise from the raw camera buffer (EXIF right).
    static func rawPoint(portraitPoint p: SIMD2<Float>) -> SIMD2<Float> {
        SIMD2(1 - p.y, 1 - p.x)
    }

    static func make(depth: [Float], width: Int, height: Int,
                     portraitBounds: SIMD4<Float>) -> [Bool]? {
        let (count, overflow) = width.multipliedReportingOverflow(by: height)
        guard width > 0, height > 0, !overflow, depth.count == count,
              (0..<4).allSatisfy({ portraitBounds[$0].isFinite }),
              portraitBounds.x >= 0, portraitBounds.y >= 0,
              portraitBounds.z > 0, portraitBounds.w > 0,
              portraitBounds.x + portraitBounds.z <= 1.001,
              portraitBounds.y + portraitBounds.w <= 1.001 else { return nil }
        let center = SIMD2(portraitBounds.x + portraitBounds.z * 0.5,
                           portraitBounds.y + portraitBounds.w * 0.5)
        let rawCenter = rawPoint(portraitPoint: center)
        let cx = min(width - 1, max(0, Int(rawCenter.x * Float(width))))
        let cy = min(height - 1, max(0, Int(rawCenter.y * Float(height))))
        var central: [Float] = []
        for y in max(0, cy - 3)...min(height - 1, cy + 3) {
            for x in max(0, cx - 3)...min(width - 1, cx + 3) {
                let d = depth[y * width + x]
                if d.isFinite, d >= 0.15, d <= 0.65 { central.append(d) }
            }
        }
        guard central.count >= 4 else { return nil }
        central.sort()
        let reference = central[central.count / 2]
        // A 2D ellipse removes rectangular background corners; measured depth controls shape.
        var candidate = [Bool](repeating: false, count: count)
        for y in 0..<height {
            for x in 0..<width {
                let portrait = SIMD2<Float>(1 - (Float(y) + 0.5) / Float(height),
                                            1 - (Float(x) + 0.5) / Float(width))
                let delta = (portrait - center) / SIMD2(portraitBounds.z * 0.49, portraitBounds.w * 0.49)
                let index = y * width + x, d = depth[index]
                candidate[index] = simd_length_squared(delta) <= 1 && d.isFinite
                    && d >= 0.15 && d <= 0.65 && abs(d - reference) <= 0.085
            }
        }
        var seed: Int?
        var best = Int.max
        for y in max(0, cy - 3)...min(height - 1, cy + 3) {
            for x in max(0, cx - 3)...min(width - 1, cx + 3) where candidate[y * width + x] {
                let distance = (x - cx) * (x - cx) + (y - cy) * (y - cy)
                if distance < best { best = distance; seed = y * width + x }
            }
        }
        guard let seed else { return nil }
        var mask = [Bool](repeating: false, count: count)
        var stack = [seed]; mask[seed] = true
        while let index = stack.popLast() {
            let x = index % width, y = index / width
            for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
                guard nx >= 0, nx < width, ny >= 0, ny < height else { continue }
                let next = ny * width + nx
                if candidate[next], !mask[next], abs(depth[next] - depth[index]) <= 0.012 {
                    mask[next] = true; stack.append(next)
                }
            }
        }
        return mask.filter { $0 }.count >= 600 ? mask : nil
    }
}
