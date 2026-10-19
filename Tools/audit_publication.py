#!/usr/bin/env python3
"""Check the exact public-source boundary before committing or publishing."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    '.gitignore', 'LICENSE', 'Package.swift', 'README.md', 'README.zh-CN.md',
    'Tools/audit_publication.py',
    'Sources/SyntheticDemo/main.swift',
    'Sources/TrueDepthReconstruction/CameraCalibration.swift',
    'Sources/TrueDepthReconstruction/TrueDepthCalibration.swift',
    'Sources/TrueDepthReconstruction/TrueDepthCalibration+AVFoundation.swift',
    'Sources/TrueDepthReconstruction/TrueDepthCalibrationValidator.swift',
    'Tests/TrueDepthReconstructionTests/TrueDepthCalibrationTests.swift',
    'Tests/TrueDepthReconstructionTests/TrueDepthCalibrationValidatorTests.swift',
    'Sources/TrueDepthReconstruction/CompactUniformGrid.swift',
    'Sources/TrueDepthReconstruction/CylindricalFaceFrame.swift',
    'Sources/TrueDepthReconstruction/FaceDepthKeyframe.swift',
    'Sources/TrueDepthReconstruction/FaceMesh.swift',
    'Sources/TrueDepthReconstruction/MarchingCubes.swift',
    'Sources/TrueDepthReconstruction/MarchingCubesTables.swift',
    'Sources/TrueDepthReconstruction/MeasuredCylindricalGridProcessor.swift',
    'Sources/TrueDepthReconstruction/MeasuredDepthDeprojector.swift',
    'Sources/TrueDepthReconstruction/MeasuredDepthPreprocessor.metal',
    'Sources/TrueDepthReconstruction/MeasuredDepthPreprocessor.swift',
    'Sources/TrueDepthReconstruction/MeasuredFaceReconstructor.swift',
    'Sources/TrueDepthReconstruction/MeasuredMetalSupport.swift',
    'Sources/TrueDepthReconstruction/MeasuredTSDFSurfaceReconstructor.swift',
    'Sources/TrueDepthReconstruction/Reconstruction.swift',
    'Tests/TrueDepthReconstructionTests/MarchingCubesTests.swift',
    'Tests/TrueDepthReconstructionTests/MeasuredMetalFailureTests.swift',
    'Tests/TrueDepthReconstructionTests/MeasuredTSDFSurfaceReconstructorTests.swift',
    'Tests/TrueDepthReconstructionTests/ReconstructionTests.swift',
}
# Deliberately no network, app, capture, rendering or persistence frameworks.
ALLOWED_IMPORTS = {'Foundation', 'simd', 'Metal', 'XCTest', 'PackageDescription', 'TrueDepthReconstruction'}
errors = []
if (ROOT / '.git').exists():
    raw = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
    paths = {p.decode() for p in raw.split(b'\0') if p}
else:
    paths = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*')
             if p.is_file() and not any(part in {'.build', '.swiftpm', '.git'} for part in p.relative_to(ROOT).parts)}
for name in sorted(paths - ALLOWED):
    errors.append(f'File outside publication allowlist: {name}')
for name in sorted(ALLOWED - paths):
    errors.append(f'Missing expected public file: {name}')
for name in sorted(paths & ALLOWED):
    p = ROOT / name
    if p.is_symlink():
        errors.append(f'Symlink not allowed: {name}')
        continue
    if p.stat().st_size > 256_000:
        errors.append(f'Unexpectedly large file: {name}')
    source = p.read_text(encoding='utf-8')
    # Run source checks only on production/test code, not explanatory docs or this scanner.
    if p.suffix not in {'.swift', '.metal'}:
        continue
    for module in re.findall(r'^\s*(?:@testable\s+)?import\s+(\w+)', source, re.M):
        if module not in ALLOWED_IMPORTS and not (
            module == 'AVFoundation' and name == 'Sources/TrueDepthReconstruction/TrueDepthCalibration+AVFoundation.swift'
        ):
            errors.append(f'Forbidden import {module}: {name}')
    checks = {
        'network endpoint': r'https?://',
        'personal filesystem path': r'/(?:Users|home)/',
        'network client': r'\b(?:URLSession|NSURLSession|URLRequest|NWConnection|CFHTTPMessage|WebSocket|WKWebView)\b',
        'app or capture dependency': r'\b(?:ARFaceGeometry|ARFaceProvider|SwiftUI|UIKit|SceneKit|SwiftData|AVCaptureSession|WorkSession|AppAttest|TelemetryClient|TelemetryServer|ScanTelemetry)\b',
        'credential': r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+|-----BEGIN [A-Z ]*PRIVATE KEY-----)',
        'real fixture reference': r'(?:RealCaptureFixture|Fixtures/RealCapture|truemyface-debug-\d)',
    }
    for label, pattern in checks.items():
        if re.search(pattern, source):
            errors.append(f'{label}: {name}')
if errors:
    print('\n'.join(errors), file=sys.stderr)
    sys.exit(1)
print(f'Publication audit passed: {len(paths)} allowlisted text files; algorithm-only sources and synthetic tests/demo.')
