import CoreVideo
import Foundation
import XCTest
import simd
@testable import TrueDepthCapture

final class DepthCaptureInputTests: XCTestCase {
    func testFloatDepthCopyHonorsRowPaddingAndDropsInvalidDepth() throws {
        let rowBytes = 32, width = 3, height = 2
        let memory = UnsafeMutableRawPointer.allocate(byteCount: rowBytes * height, alignment: 16)
        defer { memory.deallocate() }
        memory.initializeMemory(as: UInt8.self, repeating: 0, count: rowBytes * height)
        let rows: [[Float]] = [[0.3, .nan, 0.2], [0.4, 0.8, -1]]
        for y in 0..<height {
            for x in 0..<width { memory.storeBytes(of: rows[y][x], toByteOffset: y * rowBytes + x * 4, as: Float.self) }
        }
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreateWithBytes(kCFAllocatorDefault, width, height,
            kCVPixelFormatType_DepthFloat32, memory, rowBytes, nil, nil, nil, &pixelBuffer)
        XCTAssertEqual(status, kCVReturnSuccess)
        let result = try XCTUnwrap(DepthBufferReader.read(try XCTUnwrap(pixelBuffer)))
        XCTAssertEqual(result.count, 6)
        XCTAssertEqual(result[0], 0.3); XCTAssertTrue(result[1].isNaN); XCTAssertEqual(result[2], 0.2)
        XCTAssertEqual(result[3], 0.4); XCTAssertTrue(result[4].isNaN); XCTAssertTrue(result[5].isNaN)
    }

    func testPortraitNormalizedCoordinatesMapBackToRawSensorSpace() {
        XCTAssertEqual(DepthForegroundMask.rawPoint(portraitPoint: SIMD2(0, 0)), SIMD2(1, 1))
        XCTAssertEqual(DepthForegroundMask.rawPoint(portraitPoint: SIMD2(1, 0)), SIMD2(1, 0))
        XCTAssertEqual(DepthForegroundMask.rawPoint(portraitPoint: SIMD2(0.25, 0.75)), SIMD2(0.25, 0.75))
    }

    func testForegroundUsesMeasuredDepthAndRejectsBackgroundAndDisconnectedIslands() throws {
        let width = 160, height = 200
        var depth = [Float](repeating: 0.3, count: width * height)
        // A full-depth discontinuity splits the right edge of the ROI from the seed.
        for y in 0..<height { for x in 96..<100 { depth[y * width + x] = 0.6 } }
        depth[100 * width + 80] = .nan
        let mask = try XCTUnwrap(DepthForegroundMask.make(depth: depth, width: width, height: height,
                                                        portraitBounds: SIMD4(0.1, 0.1, 0.8, 0.8)))
        XCTAssertTrue(mask[101 * width + 80])
        XCTAssertFalse(mask[100 * width + 80])
        XCTAssertFalse(mask[100 * width + 97])
        XCTAssertFalse(mask[100 * width + 110])
        XCTAssertFalse(mask[0])
        XCTAssertGreaterThan(mask.filter { $0 }.count, 600)
    }

    func testMaskRejectsMissingDepthInvalidROIAndSmallSupport() {
        XCTAssertNil(DepthForegroundMask.make(depth: [Float](repeating: .nan, count: 32_000), width: 160, height: 200,
                                             portraitBounds: SIMD4(0.1, 0.1, 0.8, 0.8)))
        XCTAssertNil(DepthForegroundMask.make(depth: [0.3], width: Int.max, height: 2,
                                             portraitBounds: SIMD4(0.1, 0.1, 0.8, 0.8)))
        XCTAssertNil(DepthForegroundMask.make(depth: [Float](repeating: 0.3, count: 32_000), width: 160, height: 200,
                                             portraitBounds: SIMD4(.nan, 0.1, 0.8, 0.8)))
        XCTAssertNil(DepthForegroundMask.make(depth: [Float](repeating: 0.3, count: 100), width: 10, height: 10,
                                             portraitBounds: SIMD4(0.1, 0.1, 0.8, 0.8)))
    }
}
