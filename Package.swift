// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "TrueDepthReconstruction",
    platforms: [.macOS(.v14), .iOS("18.0")],
    products: [
        .library(name: "TrueDepthReconstruction", targets: ["TrueDepthReconstruction"]),
        .library(name: "TrueDepthCapture", targets: ["TrueDepthCapture"]),
        .executable(name: "reconstruct-synthetic", targets: ["SyntheticDemo"])
    ],
    targets: [
        .target(name: "TrueDepthReconstruction", resources: [.copy("MeasuredDepthPreprocessor.metal")]),
        .target(name: "TrueDepthCapture", dependencies: ["TrueDepthReconstruction"]),
        .executableTarget(name: "SyntheticDemo", dependencies: ["TrueDepthReconstruction"]),
        .testTarget(name: "TrueDepthReconstructionTests", dependencies: ["TrueDepthReconstruction"]),
        .testTarget(name: "TrueDepthCaptureTests", dependencies: ["TrueDepthCapture", "TrueDepthReconstruction"])
    ]
)
