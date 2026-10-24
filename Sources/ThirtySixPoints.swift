import Foundation
import Metal

// Metal port of the 36 Points simulation (2019–2022) by Sage Jenson.
// https://www.sagejenson.com/36points/
// Licensed CC BY-NC-SA 3.0: https://creativecommons.org/licenses/by-nc-sa/3.0/

struct Particle {
    var x: Float
    var y: Float
    var age: Float
    var theta: Float
}

final class ThirtySixPoints {
    private(set) var displayTexture: MTLTexture
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let particles: MTLBuffer
    private let trails: [MTLTexture]
    private let updatePipeline: MTLComputePipelineState
    private let paintPipeline: MTLRenderPipelineState
    private let blurPipeline: MTLRenderPipelineState
    private let fadePipeline: MTLRenderPipelineState
    private let copyPipeline: MTLRenderPipelineState
    private let particleCount: Int
    private let simSize = 512
    // The original paints 1px strokes into a 1080 canvas. Scale that weight with the display.
    private let strokeResolution: Float = 1080
    private var front = 0
    private var slot = 0
    private var lerpT: Float = 1
    private var past: [Float]
    private var target: [Float]
    private var current: [Float]

    var name: String { PointsCatalog.names[slot] }
    var index: Int { slot }
    var count: Int { PointsCatalog.count }

    var inverted: Bool {
        PointsCatalog.inverts(slot)
    }

    init(device: MTLDevice, queue: MTLCommandQueue, displaySide: Int = 1080) throws {
        self.device = device
        self.queue = queue
        particleCount = Int((Double(simSize * simSize) * 2.7).rounded(.down))
        let displaySide = max(displaySide, 1)
        guard let particles = device.makeBuffer(length: particleCount * MemoryLayout<Particle>.stride, options: .storageModeShared),
              let trailA = Self.texture(device, size: simSize),
              let trailB = Self.texture(device, size: simSize),
              let screen = Self.texture(device, size: displaySide)
        else { throw PointsError.allocation }
        self.particles = particles
        trails = [trailA, trailB]
        displayTexture = screen

        let library = try device.makeDefaultLibrary(bundle: .main)
        guard let update = library.makeFunction(name: "pointsUpdate"),
              let fill = library.makeFunction(name: "pointsFillVert"),
              let paintVert = library.makeFunction(name: "pointsPaintVert"),
              let paintFrag = library.makeFunction(name: "pointsPaintFrag"),
              let blurFrag = library.makeFunction(name: "pointsBlurFrag"),
              let fadeFrag = library.makeFunction(name: "pointsFadeFrag"),
              let copyFrag = library.makeFunction(name: "fieldCopyFrag"),
              let copyVert = library.makeFunction(name: "blitVert")
        else { throw PointsError.shader }

        updatePipeline = try device.makeComputePipelineState(function: update)
        paintPipeline = try Self.paintPipeline(device, vert: paintVert, frag: paintFrag)
        blurPipeline = try Self.coverPipeline(device, vert: fill, frag: blurFrag, blend: false)
        fadePipeline = try Self.coverPipeline(device, vert: fill, frag: fadeFrag, blend: true)
        copyPipeline = try Self.coverPipeline(device, vert: copyVert, frag: copyFrag, blend: false)

        let first = PointsCatalog.values[0]
        past = first
        target = first
        current = first
        reset()
    }

    func reset() {
        front = 0
        seedParticles()
        clear(trails[0])
        clear(trails[1])
        clear(displayTexture)
    }

    func select(_ index: Int) {
        let next = min(max(0, index), PointsCatalog.count - 1)
        guard next != slot else { return }
        past = current
        target = PointsCatalog.values[next]
        lerpT = 0
        slot = next
    }

    func setDisplaySide(_ side: Int) {
        let side = max(side, 1)
        guard side > displayTexture.width,
              let next = Self.texture(device, size: side),
              let buffer = queue.makeCommandBuffer(),
              let pass = Self.pass(next, clear: false),
              let encoder = buffer.makeRenderCommandEncoder(descriptor: pass)
        else { return }
        var size = SIMD2<Float>(Float(side), Float(side))
        encoder.setRenderPipelineState(copyPipeline)
        encoder.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(side), height: Double(side), znear: 0, zfar: 1))
        encoder.setFragmentTexture(displayTexture, index: 0)
        encoder.setFragmentBytes(&size, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        buffer.commit()
        displayTexture = next
    }

    func step() {
        lerpT = min(1, lerpT + max((1 - lerpT) * 0.15, 0.001))
        current = mix(past, target, lerpT)

        guard let buffer = queue.makeCommandBuffer() else { return }
        encodeUpdate(buffer)
        encodePaint(buffer, texture: trails[front], opacity: current[14], pointSize: 1, dotSize: current[19], viewport: simSize)
        let passes = min(16, max(0, Int(current[16].rounded())))
        for _ in 0..<passes {
            encodeBlur(buffer, from: trails[front], to: trails[1 - front], decay: current[15])
            front = 1 - front
        }
        encodeFade(buffer, amount: current[18])
        let stroke = Float(displayTexture.width) / strokeResolution
        encodePaint(
            buffer,
            texture: displayTexture,
            opacity: current[17],
            pointSize: stroke,
            dotSize: current[19] * stroke,
            viewport: displayTexture.width
        )
        buffer.commit()
    }

    private func seedParticles() {
        let raw = particles.contents().bindMemory(to: Particle.self, capacity: particleCount)
        for index in 0..<particleCount {
            let n = UInt32(index)
            raw[index] = Particle(
                x: unit(n, 1) * 2 - 1,
                y: unit(n, 2) * 2 - 1,
                age: Float(index) / Float(particleCount),
                theta: unit(n, 3) * 2 * .pi
            )
        }
    }

    private func encodeUpdate(_ buffer: MTLCommandBuffer) {
        guard let encoder = buffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(updatePipeline)
        encoder.setBuffer(particles, offset: 0, index: 0)
        current.withUnsafeBufferPointer { ptr in
            encoder.setBytes(ptr.baseAddress!, length: 20 * MemoryLayout<Float>.stride, index: 1)
        }
        var size = Float(simSize)
        var total = UInt32(particleCount)
        encoder.setBytes(&size, length: MemoryLayout<Float>.stride, index: 2)
        encoder.setBytes(&total, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setTexture(trails[front], index: 0)
        let width = updatePipeline.threadExecutionWidth
        let threads = MTLSize(width: width, height: 1, depth: 1)
        let groups = MTLSize(width: (particleCount + width - 1) / width, height: 1, depth: 1)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        encoder.endEncoding()
    }

    private func encodePaint(
        _ buffer: MTLCommandBuffer,
        texture: MTLTexture,
        opacity: Float,
        pointSize: Float,
        dotSize: Float,
        viewport: Int
    ) {
        guard let pass = Self.pass(texture, clear: false),
              let encoder = buffer.makeRenderCommandEncoder(descriptor: pass)
        else { return }
        var paint = PaintUniforms(opacity: opacity, pointSize: pointSize, dotSize: dotSize)
        encoder.setRenderPipelineState(paintPipeline)
        encoder.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(viewport), height: Double(viewport), znear: 0, zfar: 1))
        encoder.setVertexBuffer(particles, offset: 0, index: 0)
        encoder.setVertexBytes(&paint, length: MemoryLayout<PaintUniforms>.stride, index: 1)
        encoder.setFragmentBytes(&paint, length: MemoryLayout<PaintUniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: particleCount)
        encoder.endEncoding()
    }

    private func encodeBlur(_ buffer: MTLCommandBuffer, from: MTLTexture, to: MTLTexture, decay: Float) {
        guard let pass = Self.pass(to, clear: false),
              let encoder = buffer.makeRenderCommandEncoder(descriptor: pass)
        else { return }
        var blur = BlurUniforms(decay: decay, size: Float(simSize))
        encoder.setRenderPipelineState(blurPipeline)
        encoder.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(simSize), height: Double(simSize), znear: 0, zfar: 1))
        encoder.setFragmentTexture(from, index: 0)
        encoder.setFragmentBytes(&blur, length: MemoryLayout<BlurUniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    private func encodeFade(_ buffer: MTLCommandBuffer, amount: Float) {
        guard let pass = Self.pass(displayTexture, clear: false),
              let encoder = buffer.makeRenderCommandEncoder(descriptor: pass)
        else { return }
        var fade = amount
        encoder.setRenderPipelineState(fadePipeline)
        let side = Double(displayTexture.width)
        encoder.setViewport(MTLViewport(originX: 0, originY: 0, width: side, height: side, znear: 0, zfar: 1))
        encoder.setFragmentBytes(&fade, length: MemoryLayout<Float>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    private func clear(_ texture: MTLTexture) {
        guard let buffer = queue.makeCommandBuffer(),
              let pass = Self.pass(texture, clear: true),
              let encoder = buffer.makeRenderCommandEncoder(descriptor: pass)
        else { return }
        encoder.endEncoding()
        buffer.commit()
        buffer.waitUntilCompleted()
    }

    private func mix(_ a: [Float], _ b: [Float], _ t: Float) -> [Float] {
        var mixed = [Float](repeating: 0, count: 20)
        for index in 0..<20 {
            let amount = index == 19 ? pow(t, 10) : t
            let value: Float
            if index == 1 || index == 4 || index == 7 || index == 11 || index == 19 {
                value = pow(max(a[index], 0), 1 - amount) * pow(max(b[index], 0), amount)
            } else {
                value = a[index] + (b[index] - a[index]) * amount
            }
            mixed[index] = value.isFinite ? value : 0
        }
        return mixed
    }

    private func unit(_ n: UInt32, _ salt: UInt32) -> Float {
        var x = n &* 747796405 &+ salt &* 2891336453
        x ^= x >> 16
        x &*= 0x7feb352d
        x ^= x >> 15
        x &*= 0x846ca68b
        x ^= x >> 16
        return Float(x & 0x00ff_ffff) / Float(0x0100_0000)
    }

    private static func texture(_ device: MTLDevice, size: Int) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: size,
            height: size,
            mipmapped: false
        )
        desc.usage = [.renderTarget, .shaderRead]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }

    private static func pass(_ texture: MTLTexture, clear: Bool) -> MTLRenderPassDescriptor? {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = texture
        pass.colorAttachments[0].loadAction = clear ? .clear : .load
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        return pass
    }

    private static func blend(_ attachment: MTLRenderPipelineColorAttachmentDescriptor) {
        attachment.isBlendingEnabled = true
        attachment.sourceRGBBlendFactor = .sourceAlpha
        attachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
        attachment.sourceAlphaBlendFactor = .sourceAlpha
        attachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
    }

    private static func paintPipeline(
        _ device: MTLDevice,
        vert: MTLFunction,
        frag: MTLFunction
    ) throws -> MTLRenderPipelineState {
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vert
        desc.fragmentFunction = frag
        desc.colorAttachments[0].pixelFormat = .rgba16Float
        blend(desc.colorAttachments[0])
        return try device.makeRenderPipelineState(descriptor: desc)
    }

    private static func coverPipeline(
        _ device: MTLDevice,
        vert: MTLFunction,
        frag: MTLFunction,
        blend: Bool
    ) throws -> MTLRenderPipelineState {
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vert
        desc.fragmentFunction = frag
        desc.colorAttachments[0].pixelFormat = .rgba16Float
        if blend { self.blend(desc.colorAttachments[0]) }
        return try device.makeRenderPipelineState(descriptor: desc)
    }

    enum PointsError: Error {
        case allocation
        case shader
    }
}

private struct PaintUniforms {
    var opacity: Float
    var pointSize: Float
    var dotSize: Float
}

private struct BlurUniforms {
    var decay: Float
    var size: Float
}
