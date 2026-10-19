import CoreVideo
import Foundation

/// Copies row-padded Float32 depth buffers without retaining a camera frame.
enum DepthBufferReader {
    static func read(_ map: CVPixelBuffer) -> [Float]? {
        guard CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32,
              CVPixelBufferLockBaseAddress(map, .readOnly) == kCVReturnSuccess else { return nil }
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        let width = CVPixelBufferGetWidth(map), height = CVPixelBufferGetHeight(map)
        guard width > 0, height > 0, let base = CVPixelBufferGetBaseAddress(map) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(map)
        guard rowBytes >= width * MemoryLayout<Float>.stride else { return nil }
        var result = [Float](repeating: .nan, count: width * height)
        result.withUnsafeMutableBufferPointer { destination in
            for row in 0..<height {
                memcpy(destination.baseAddress! + row * width, base + row * rowBytes,
                       width * MemoryLayout<Float>.stride)
            }
        }
        for index in result.indices {
            let value = result[index]
            if !value.isFinite || value < 0.15 || value > 0.65 { result[index] = .nan }
        }
        return result
    }
}
