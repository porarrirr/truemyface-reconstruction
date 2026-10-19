import XCTest

final class CaptureDemoUITests: XCTestCase {
    func testSimulatorShowsHardwareRequirementWithoutSyntheticFallback() throws {
        #if targetEnvironment(simulator)
        let app = XCUIApplication()
        app.launch()
        let start = app.buttons["Start camera scan"]
        XCTAssertTrue(start.waitForExistence(timeout: 10))
        start.tap()
        XCTAssertTrue(app.staticTexts["A physical iPhone with an available front TrueDepth format is required."].waitForExistence(timeout: 10))
        XCTAssertTrue(start.isEnabled)
        XCTAssertFalse(app.buttons["Stop and reconstruct"].isEnabled)
        XCTAssertFalse(app.buttons["Share generated OBJ"].exists)
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "Unsupported Simulator hardware"
        attachment.lifetime = .keepAlways
        add(attachment)
        #else
        throw XCTSkip("This test checks the Simulator-only hardware rejection path; it does not initiate physical capture.")
        #endif
    }
}
