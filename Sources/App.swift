import AppKit
import Metal
import QuartzCore

@main
enum AppMain {
    static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.regular)
        let delegate = AppDelegate()
        app.delegate = delegate
        app.run()
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private let window: NSWindow
    private let view: FieldView

    override init() {
        view = FieldView(frame: NSRect(x: 0, y: 0, width: 1080, height: 1080))
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1080, height: 1080),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.contentView = view
        window.minSize = NSSize(width: 480, height: 480)
        window.backgroundColor = .black
        window.title = "36 Points Metal"
        super.init()
        view.onTitle = { [weak window] title in
            window?.title = title
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        installMenu()
        window.collectionBehavior = .fullScreenPrimary
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        view.start()
        DispatchQueue.main.async { [weak window] in
            window?.toggleFullScreen(nil)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    private func installMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem()
        main.addItem(appItem)
        let appMenu = NSMenu()
        let quit = NSMenuItem(title: "Quit 36 Points Metal", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        quit.keyEquivalentModifierMask = .command
        appMenu.addItem(quit)
        appItem.submenu = appMenu

        let viewItem = NSMenuItem()
        main.addItem(viewItem)
        let viewMenu = NSMenu(title: "View")
        let fullScreen = NSMenuItem(
            title: "Enter Full Screen",
            action: #selector(NSWindow.toggleFullScreen(_:)),
            keyEquivalent: "f"
        )
        fullScreen.keyEquivalentModifierMask = [.command, .control]
        viewMenu.addItem(fullScreen)
        viewItem.submenu = viewMenu

        NSApp.mainMenu = main
    }
}

final class FieldView: NSView {
    var onTitle: ((String) -> Void)?
    private let metalLayer = CAMetalLayer()
    private var timer: Timer?
    private var sim: ThirtySixPoints?
    private var blit: MTLRenderPipelineState?
    private var queue: MTLCommandQueue?
    private var keyMonitor: Any?
    private var showingLightGround = false

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
    }

    required init?(coder: NSCoder) { nil }

    override func makeBackingLayer() -> CALayer { metalLayer }

    func start() {
        do {
            guard let device = MTLCreateSystemDefaultDevice(),
                  let queue = device.makeCommandQueue()
            else { throw ThirtySixPoints.PointsError.allocation }
            self.queue = queue
            metalLayer.device = device
            metalLayer.pixelFormat = .bgra8Unorm
            metalLayer.framebufferOnly = true
            metalLayer.isOpaque = true
            let sim = try ThirtySixPoints(device: device, queue: queue, displaySide: displaySide())
            self.sim = sim
            let library = try device.makeDefaultLibrary(bundle: .main)
            guard let vert = library.makeFunction(name: "blitVert"),
                  let frag = library.makeFunction(name: "blitFrag")
            else { throw ThirtySixPoints.PointsError.shader }
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = vert
            desc.fragmentFunction = frag
            desc.colorAttachments[0].pixelFormat = .bgra8Unorm
            blit = try device.makeRenderPipelineState(descriptor: desc)
            publishTitle()
        } catch {
            fputs("36 Points Metal failed: \(error)\n", stderr)
            return
        }
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self else { return event }
            return self.handle(event)
        }
        let timer = Timer(timeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            self?.tick()
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    private func tick() {
        guard let sim, let blit, let queue else { return }
        metalLayer.frame = bounds
        let scale = window?.backingScaleFactor ?? 1
        metalLayer.contentsScale = scale
        metalLayer.drawableSize = CGSize(width: bounds.width * scale, height: bounds.height * scale)
        sim.setDisplaySide(displaySide())
        sim.step()
        guard let drawable = metalLayer.nextDrawable(),
              let buffer = queue.makeCommandBuffer(),
              let pass = MTLRenderPassDescriptor() as MTLRenderPassDescriptor?
        else { return }
        pass.colorAttachments[0].texture = drawable.texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        let lightGround = sim.inverted
        let ground: Double = lightGround ? 1 : 0
        pass.colorAttachments[0].clearColor = MTLClearColor(red: ground, green: ground, blue: ground, alpha: 1)
        if lightGround != showingLightGround {
            showingLightGround = lightGround
            window?.backgroundColor = NSColor(white: ground, alpha: 1)
        }
        guard let encoder = buffer.makeRenderCommandEncoder(descriptor: pass) else { return }
        var blitArgs = BlitArgs(
            width: Float(drawable.texture.width),
            height: Float(drawable.texture.height),
            invert: sim.inverted ? 1 : 0
        )
        encoder.setRenderPipelineState(blit)
        encoder.setFragmentTexture(sim.displayTexture, index: 0)
        encoder.setFragmentBytes(&blitArgs, length: MemoryLayout<BlitArgs>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        buffer.present(drawable)
        buffer.commit()
    }

    private func handle(_ event: NSEvent) -> NSEvent? {
        guard let sim else { return event }
        switch event.keyCode {
        case 123:
            sim.select((sim.index - 1 + sim.count) % sim.count)
            publishTitle()
            return nil
        case 124:
            sim.select((sim.index + 1) % sim.count)
            publishTitle()
            return nil
        default:
            break
        }
        if event.charactersIgnoringModifiers?.lowercased() == "r" {
            sim.reset()
            return nil
        }
        return event
    }

    private func displaySide() -> Int {
        let scale = window?.screen?.backingScaleFactor ?? window?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 2
        let frame = window?.screen?.frame ?? NSScreen.main?.frame ?? bounds
        let screenSide = Int(max(frame.height * scale, frame.width * scale * 0.5).rounded(.up))
        let drawableSide = Int(max(bounds.height * scale, bounds.width * scale * 0.5).rounded(.up))
        return max(screenSide, drawableSide, 1080)
    }

    private func publishTitle() {
        guard let sim else { return }
        onTitle?(String(format: "36 Points Metal  —  %02d  %@", sim.index + 1, sim.name))
    }
}

private struct BlitArgs {
    var width: Float
    var height: Float
    var invert: Float
}
