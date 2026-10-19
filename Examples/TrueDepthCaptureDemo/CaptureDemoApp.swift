import Foundation
import SwiftUI
import TrueDepthCapture
import TrueDepthReconstruction

// New, deliberately small harness. No UI or app infrastructure is copied from TrueMyFace.
@main
struct CaptureDemoApp: App {
    @StateObject private var model = CaptureDemoModel()
    @Environment(\.scenePhase) private var scenePhase
    var body: some Scene {
        WindowGroup {
            VStack(spacing: 18) {
                Text("TrueDepth → 3D").font(.title)
                Text("Hold your iPhone upright, about 20–50 cm away. Keep a neutral face. Start facing forward, then slowly look left, right, up and down.")
                Text(model.status).frame(maxWidth: .infinity, alignment: .leading)
                Text(model.coverage).font(.system(.body, design: .monospaced))
                if model.processing { ProgressView("Reconstructing measured depth…") }
                Button("Start camera scan") { model.start() }.disabled(model.busy)
                Button("Stop and reconstruct") { model.finish() }.disabled(!model.capturing)
                Button("Cancel capture") { model.cancel() }.disabled(!model.capturing)
                if let file = model.file { ShareLink("Share generated OBJ", item: file) }
                Text("Processing stays on this device. Sharing an OBJ is your explicit action.")
                    .font(.footnote)
            }
            .padding()
            .onChange(of: scenePhase) { _, phase in
                if phase == .background { model.cancel() }
            }
        }
    }
}

@MainActor
final class CaptureDemoModel: ObservableObject {
    @Published var status = "Ready. A physical TrueDepth iPhone is required."
    @Published var coverage = "front 0/2 · left 0/2 · right 0/2 · up 0/2 · down 0/2"
    @Published var busy = false
    @Published var capturing = false
    @Published var processing = false
    @Published var file: URL?
    private lazy var scanner = TrueDepthScanner { [weak self] event in
        Task { @MainActor [weak self] in self?.handle(event) }
    }

    func start() {
        if let file { try? FileManager.default.removeItem(at: file) }
        file = nil; busy = true; capturing = true; status = "Starting camera…"
        coverage = "front 0/2 · left 0/2 · right 0/2 · up 0/2 · down 0/2"
        scanner.start()
    }
    func finish() { scanner.finish() }
    func cancel() { if capturing { scanner.cancel() } }

    private func handle(_ event: TrueDepthCaptureEvent) {
        switch event {
        case .started: status = "Capturing. Slowly cover all five directions."
        case .progress(let count, let viewpoints):
            status = "Accepted \(count)/10 frames."
            coverage = CaptureViewpoint.allCases.map { "\($0.rawValue) \(viewpoints[$0, default: 0])/2" }.joined(separator: " · ")
        case .skipped(let message): status = message
        case .calibrationWarnings:
            status = "Calibration reports a projection-consistency warning. Stop and retry if capture looks inconsistent."
        case .reconstructing:
            capturing = false; processing = true; status = "Camera stopped. Reconstructing…"
        case .completed(let result):
            busy = false; capturing = false; processing = false
            do {
                let target = FileManager.default.temporaryDirectory.appendingPathComponent("measured-face-\(UUID().uuidString).obj")
                try Self.obj(result.mesh).write(to: target, atomically: true, encoding: .utf8)
                file = target
                status = "Generated \(result.mesh.vertexCount) vertices and \(result.mesh.triangleCount) triangles."
            } catch { status = "OBJ export failed: \(error.localizedDescription)" }
        case .failed(let error):
            busy = false; capturing = false; processing = false; status = error.localizedDescription
        case .cancelled:
            busy = false; capturing = false; processing = false; status = "Capture cancelled; frames discarded."
        }
    }

    private static func obj(_ mesh: FaceMesh) -> String {
        var output = "# Measured facial surface; vertex units: meters\no measured_face\n"
        for v in mesh.vertices { output += "v \(v.x) \(v.y) \(v.z)\n" }
        for i in stride(from: 0, to: mesh.triangleIndices.count, by: 3) {
            output += "f \(mesh.triangleIndices[i] + 1) \(mesh.triangleIndices[i + 1] + 1) \(mesh.triangleIndices[i + 2] + 1)\n"
        }
        return output
    }
}
