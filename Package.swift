// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "TrueDepthReconstruction",
    platforms: [.macOS(.v14), .iOS("18.0")],
    products: [
        .library(name: "TrueDepthReconstruction", targets: ["TrueDepthReconstruction"]),
        .executable(name: "reconstruct-synthetic", targets: ["SyntheticDemo"])
    ],
    targets: [
        .target(name: "TrueDepthReconstruction", resources: [.copy("MeasuredDepthPreprocessor.metal")]),
        .executableTarget(name: "SyntheticDemo", dependencies: ["TrueDepthReconstruction"]),
        .testTarget(name: "TrueDepthReconstructionTests", dependencies: ["TrueDepthReconstruction"])
    ]
)
