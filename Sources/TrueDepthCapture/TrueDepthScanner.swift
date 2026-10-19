#if os(iOS)
import ARKit
import AVFoundation
import Foundation
import ImageIO
import Vision
import simd
import TrueDepthReconstruction

public enum TrueDepthCaptureError: Error, LocalizedError {
    case unsupportedDevice, cameraPermissionDenied, missingCameraUsageDescription
    case timeout, trackingChanged, interrupted, insufficientCoverage, noSurface
    case processing(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedDevice: return "A physical iPhone with an available front TrueDepth format is required."
        case .cameraPermissionDenied: return "Allow camera access in Settings before scanning."
        case .missingCameraUsageDescription: return "The host app must provide NSCameraUsageDescription."
        case .timeout: return "Capture timed out. Keep a neutral face and slowly cover all five directions."
        case .trackingChanged: return "Face tracking changed identity. Start a new scan."
        case .interrupted: return "Capture was interrupted. Start a new scan."
        case .insufficientCoverage: return "Capture at least five frames across three directions including the front."
        case .noSurface: return "The measured depth did not support a reconstructable surface."
        case .processing(let detail): return detail
        }
    }
}

public enum TrueDepthCaptureEvent {
    case started
    case progress(frameCount: Int, viewpoints: [CaptureViewpoint: Int])
    case skipped(String)
    case calibrationWarnings([String])
    case reconstructing
    case completed(ReconstructionResult)
    case failed(TrueDepthCaptureError)
    case cancelled
}

/// Portrait-only capture adapter. ARKit provides session operation and rigid pose only.
/// Events arrive on the main queue. Retain this object for the duration of a scan.
public final class TrueDepthScanner: NSObject, ARSessionDelegate {
    private enum State { case idle, requestingPermission, capturing, reconstructing }
    private var captureSession: ARSession?
    private let ingress = CaptureFrameIngress()
    private let delegateQueue = DispatchQueue(label: "reconstruction.camera-ingress")
    private let queue = DispatchQueue(label: "reconstruction.capture", qos: .userInitiated)
    private let worker = DispatchQueue(label: "reconstruction.surface", qos: .userInitiated)
    private let onEvent: (TrueDepthCaptureEvent) -> Void
    private var state: State = .idle
    private var generation = UUID()
    private var selection = CaptureFrameSelection()
    private var anchorID: UUID?
    private var previousDepthTime: TimeInterval?
    private var previousAcceptedTime: TimeInterval?
    private var previousAttemptTime: TimeInterval = -.infinity
    private var previousNoticeTime: TimeInterval = -.infinity
    private var deadline: DispatchSourceTimer?

    public init(onEvent: @escaping (TrueDepthCaptureEvent) -> Void) {
        self.onEvent = onEvent
        super.init()
    }

    deinit { deadline?.cancel(); captureSession?.pause() }

    /// A front TrueDepth video format is checked in addition to face-tracking support.
    public static var isSupported: Bool {
        #if targetEnvironment(simulator)
        return false
        #else
        return ARFaceTrackingConfiguration.isSupported && preferredFormat() != nil
        #endif
    }

    public func start() {
        queue.async { [self] in
            guard state == .idle else { return }
            guard let purpose = Bundle.main.object(forInfoDictionaryKey: "NSCameraUsageDescription") as? String,
                  !purpose.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                emit(.failed(.missingCameraUsageDescription)); return
            }
            guard Self.isSupported else { emit(.failed(.unsupportedDevice)); return }
            generation = UUID()
            let token = generation
            state = .requestingPermission
            switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized: begin(token: token)
            case .notDetermined:
                AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                    guard let self else { return }
                    self.queue.async {
                        guard self.generation == token, self.state == .requestingPermission else { return }
                        if granted { self.begin(token: token) }
                        else { self.fail(.cameraPermissionDenied) }
                    }
                }
            default: fail(.cameraPermissionDenied)
            }
        }
    }

    /// Stops camera capture and reconstructs if partial coverage is sufficient.
    /// Full ten-frame/five-direction coverage triggers this automatically.
    public func finish() { queue.async { [self] in finishOnQueue() } }

    /// Cancels permission/capture work. Reconstruction already in progress runs to completion.
    public func cancel() {
        queue.async { [self] in
            guard state == .capturing || state == .requestingPermission else { return }
            reset(); emit(.cancelled)
        }
    }

    private static func preferredFormat() -> ARConfiguration.VideoFormat? {
        ARFaceTrackingConfiguration.supportedVideoFormats.first {
            $0.captureDeviceType == .builtInTrueDepthCamera && $0.captureDevicePosition == .front
        }
    }

    private func begin(token: UUID) {
        guard generation == token, state == .requestingPermission else { return }
        guard let format = Self.preferredFormat() else { fail(.unsupportedDevice); return }
        selection = CaptureFrameSelection(); anchorID = nil
        previousDepthTime = nil; previousAcceptedTime = nil
        previousAttemptTime = -.infinity; previousNoticeTime = -.infinity
        let configuration = ARFaceTrackingConfiguration()
        configuration.videoFormat = format
        configuration.worldAlignment = .camera
        configuration.maximumNumberOfTrackedFaces = 1
        configuration.isLightEstimationEnabled = false
        configuration.providesAudioData = false
        state = .capturing
        ingress.setEpoch(token)
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + 45)
        timer.setEventHandler { [weak self] in
            guard let self, self.state == .capturing else { return }
            self.fail(.timeout)
        }
        deadline = timer; timer.resume()
        let session = ARSession()
        session.delegate = self
        session.delegateQueue = delegateQueue
        captureSession = session
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        emit(.started)
    }

    public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard let token = ingress.acquire() else { return }
        queue.async { [self] in
            defer { ingress.release() }
            guard generation == token, state == .capturing, captureSession === session else { return }
            autoreleasepool { handleFrame(frame) }
        }
    }

    private func handleFrame(_ frame: ARFrame) {
        guard let face = frame.anchors.compactMap({ $0 as? ARFaceAnchor }).first, face.isTracked else {
            notice("Keep one face visible.", time: frame.timestamp); return
        }
        if let anchorID, anchorID != face.identifier { fail(.trackingChanged); return }
        guard let depth = frame.capturedDepthData else {
            notice("Waiting for measured TrueDepth data.", time: frame.timestamp); return
        }
        let depthTime = frame.capturedDepthDataTimestamp
        guard CaptureFrameTiming.accepts(imageTime: frame.timestamp, depthTime: depthTime,
                                         previousDepthTime: previousDepthTime,
                                         previousAcceptedTime: previousAcceptedTime),
              frame.timestamp - previousAttemptTime >= 0.25 else { return }
        previousDepthTime = depthTime; previousAttemptTime = frame.timestamp
        guard depth.depthDataAccuracy == .absolute else {
            notice("Waiting for metric depth.", time: frame.timestamp); return
        }
        do {
            guard let calibration = depth.cameraCalibrationData else {
                notice("Missing depth calibration.", time: frame.timestamp); return
            }
            let converted = depth.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
            let map = converted.depthDataMap
            let width = CVPixelBufferGetWidth(map), height = CVPixelBufferGetHeight(map)
            let projection = try TrueDepthCalibration.makeProjectionModel(
                calibration: calibration, depthImageSize: SIMD2(Float(width), Float(height)))
            let cameraFromFace = simd_inverse(frame.camera.transform) * face.transform
            guard (0..<4).allSatisfy({ column in (0..<4).allSatisfy { cameraFromFace[column][$0].isFinite } }),
                  abs(simd_determinant(cameraFromFace) - 1) < 0.001,
                  -cameraFromFace[3].z >= 0.15, -cameraFromFace[3].z <= 0.65 else {
                notice("Hold the phone about 20–50 cm from your face.", time: frame.timestamp); return
            }
            let pose = CapturePose.portrait(cameraFromFace)
            guard CaptureViewpoint.classify(pose) != nil else {
                notice("Turn more gently; keep the face mostly toward the camera.", time: frame.timestamp); return
            }
            let request = VNDetectFaceRectanglesRequest()
            try VNImageRequestHandler(cvPixelBuffer: frame.capturedImage, orientation: .right, options: [:]).perform([request])
            guard let observations = request.results, observations.count == 1, let observation = observations.first,
                  observation.confidence >= 0.8, let values = DepthBufferReader.read(map) else {
                notice("Waiting for one clearly visible face and valid depth.", time: frame.timestamp); return
            }
            let box = observation.boundingBox
            guard let mask = DepthForegroundMask.make(depth: values, width: width, height: height,
                portraitBounds: SIMD4(Float(box.minX), Float(box.minY), Float(box.width), Float(box.height))) else {
                notice("Insufficient connected foreground depth.", time: frame.timestamp); return
            }
            let keyframe = FaceDepthKeyframe(timestamp: depthTime, pose: pose, depthValues: values,
                width: width, height: height, intrinsics: projection.intrinsics, lens: projection.lens,
                cameraFromFace: cameraFromFace, faceMask: mask)
            guard selection.append(keyframe) else { return }
            anchorID = face.identifier; previousAcceptedTime = depthTime
            if !projection.validation.warnings.isEmpty { emit(.calibrationWarnings(projection.validation.warnings)) }
            emit(.progress(frameCount: selection.frames.count, viewpoints: selection.counts))
            if selection.isComplete { finishOnQueue() }
        } catch {
            fail(.processing("Depth capture/calibration failed: \(error.localizedDescription)"))
        }
    }

    public func session(_ session: ARSession, didFailWithError error: Error) {
        queue.async { [self] in
            if state == .capturing, captureSession === session { fail(.processing(error.localizedDescription)) }
        }
    }

    public func sessionWasInterrupted(_ session: ARSession) {
        queue.async { [self] in
            if state == .capturing, captureSession === session { fail(.interrupted) }
        }
    }

    private func finishOnQueue() {
        guard state == .capturing else { return }
        guard selection.canReconstruct else { fail(.insufficientCoverage); return }
        ingress.setEpoch(nil)
        captureSession?.pause(); captureSession = nil; deadline?.cancel(); deadline = nil
        state = .reconstructing
        let frames = selection.frames
        selection = CaptureFrameSelection()
        emit(.reconstructing)
        worker.async { [self] in
            let result = Result { try Reconstruction.reconstruct(frames: frames) }
            queue.async { [self] in
                reset()
                switch result {
                case .success(let surface):
                    if let surface { emit(.completed(surface)) }
                    else { emit(.failed(.noSurface)) }
                case .failure(let error): emit(.failed(.processing(error.localizedDescription)))
                }
            }
        }
    }

    private func fail(_ error: TrueDepthCaptureError) { reset(); emit(.failed(error)) }
    private func reset() {
        generation = UUID(); ingress.setEpoch(nil); captureSession?.pause(); captureSession = nil; deadline?.cancel(); deadline = nil
        selection = CaptureFrameSelection(); anchorID = nil; state = .idle
    }
    private func emit(_ event: TrueDepthCaptureEvent) {
        let callback = onEvent
        DispatchQueue.main.async { callback(event) }
    }
    private func notice(_ message: String, time: TimeInterval) {
        if time - previousNoticeTime >= 1 { previousNoticeTime = time; emit(.skipped(message)) }
    }
}
#endif
