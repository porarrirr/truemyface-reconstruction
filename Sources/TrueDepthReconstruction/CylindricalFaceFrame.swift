import Foundation
import simd

/// Data-derived cylindrical coordinates for measured-surface remeshing.
struct CylindricalFaceFrame: Equatable {
    enum Bounds: Equatable {
        case fullExtent
        case percentile(low: Float, high: Float)
    }

    struct Samples {
        var frame: CylindricalFaceFrame
        var thetas: [Float]
        var radii: [Float]
    }

    static let defaultBinThetaDegrees: Float = 0.65
    static let defaultBinHeight: Float = 0.001

    let axisX: Float
    let axisZ: Float
    let thetaOrigin: Float
    let yOrigin: Float
    let binTheta: Float
    let binHeight: Float
    let columns: Int
    let rows: Int

    static func samples(points: [SIMD3<Float>],
                        binThetaDegrees: Float = Self.defaultBinThetaDegrees,
                        binHeight: Float = Self.defaultBinHeight,
                        bounds: Bounds) -> Samples? {
        guard !points.isEmpty, binThetaDegrees > 0, binHeight > 0 else { return nil }

        guard let axis = axis(for: points) else { return nil }
        let binTheta = binThetaDegrees * .pi / 180
        var thetas = [Float](repeating: 0, count: points.count)
        var radii = [Float](repeating: 0, count: points.count)
        var ys = [Float]()
        ys.reserveCapacity(points.count)

        for (index, point) in points.enumerated() {
            let theta = atan2f(point.x - axis.x, point.z - axis.z)
            thetas[index] = theta
            radii[index] = hypotf(point.x - axis.x, point.z - axis.z)
            ys.append(point.y)
        }

        let thetaRange = range(values: thetas, bounds: bounds)
        let yRange = range(values: ys, bounds: bounds)
        guard thetaRange.high >= thetaRange.low, yRange.high >= yRange.low else { return nil }

        let columns = Int(((thetaRange.high - thetaRange.low) / binTheta).rounded(.up)) + 1
        let rows = Int(((yRange.high - yRange.low) / binHeight).rounded(.up)) + 1
        guard columns > 0, rows > 0 else { return nil }

        return Samples(
            frame: CylindricalFaceFrame(
                axisX: axis.x,
                axisZ: axis.z,
                thetaOrigin: thetaRange.low,
                yOrigin: yRange.low,
                binTheta: binTheta,
                binHeight: binHeight,
                columns: columns,
                rows: rows
            ),
            thetas: thetas,
            radii: radii
        )
    }

    static func axis(for points: [SIMD3<Float>]) -> (x: Float, z: Float)? {
        guard !points.isEmpty else { return nil }
        var xs = points.map(\.x)
        var zs = points.map(\.z)
        xs.sort()
        zs.sort()
        let x0 = xs[xs.count / 2]
        let zIndex = min(zs.count - 1, max(0, Int(Float(zs.count) * 0.02)))
        return (x0, zs[zIndex] - 0.010)
    }

    func binIndex(theta: Float, y: Float) -> Int {
        row(for: y) * columns + column(for: theta)
    }

    func column(for theta: Float) -> Int {
        min(columns - 1, max(0, Int((theta - thetaOrigin) / binTheta)))
    }

    func row(for y: Float) -> Int {
        min(rows - 1, max(0, Int((y - yOrigin) / binHeight)))
    }

    func binTheta(_ bin: Int) -> Float {
        thetaOrigin + Float(bin % columns) * binTheta
    }

    func binY(_ bin: Int) -> Float {
        yOrigin + Float(bin / columns) * binHeight
    }

    func worldX(radius: Float, theta: Float) -> Float {
        axisX + radius * sinf(theta)
    }

    func worldZ(radius: Float, theta: Float) -> Float {
        axisZ + radius * cosf(theta)
    }

    private static func range(values: [Float], bounds: Bounds) -> (low: Float, high: Float) {
        var sorted = values
        sorted.sort()
        switch bounds {
        case .fullExtent:
            return (sorted[0], sorted[sorted.count - 1])
        case .percentile(let low, let high):
            return (percentile(sorted, low), percentile(sorted, high))
        }
    }

    private static func percentile(_ sorted: [Float], _ q: Float) -> Float {
        sorted[min(sorted.count - 1, max(0, Int(Float(sorted.count - 1) * q)))]
    }
}
