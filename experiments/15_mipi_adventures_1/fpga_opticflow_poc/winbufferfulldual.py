from amaranth import *

class FrameBuffer2x(Elaboratable):
    """
    Buffers two full grayscale frames, swapping every vsync.
    On each pixel clock, outputs pixel from current frame and matching pixel from previous frame.
    """
    def __init__(self, hres=640, vres=480):
        # Inputs
        self.pixel_in = Signal(8)
        self.valid_in = Signal()
        self.hsync = Signal()
        self.vsync = Signal()

        # Outputs
        self.curr_pixel = Signal(8)
        self.prev_pixel = Signal(8)
        self.valid_out = Signal()

        # Internal
        self.hres = hres
        self.vres = vres

    def elaborate(self, platform):
        m = Module()

        # Dual frame memories
        depth = self.hres * self.vres
        fb0 = Memory(width=8, depth=depth)
        fb1 = Memory(width=8, depth=depth)

        # Read and write ports
        rp0 = m.submodules.rp0 = fb0.read_port()
        wp0 = m.submodules.wp0 = fb0.write_port()
        rp1 = m.submodules.rp1 = fb1.read_port()
        wp1 = m.submodules.wp1 = fb1.write_port()

        # Pixel position
        addr = Signal(range(depth))
        y = Signal(range(self.vres))
        x = Signal(range(self.hres))

        # Frame selector
        frame_select = Signal()  # 0 or 1

        with m.If(self.valid_in):
            m.d.sync += [
                addr.eq(y * self.hres + x),

                # Update write ports
                wp0.addr.eq(addr),
                wp1.addr.eq(addr),
                wp0.data.eq(self.pixel_in),
                wp1.data.eq(self.pixel_in),

                # Update read ports from previous frame
                rp0.addr.eq(addr),
                rp1.addr.eq(addr),

                # Output assignment
                self.valid_out.eq(1),
                self.curr_pixel.eq(self.pixel_in),
                self.prev_pixel.eq(Mux(frame_select, rp0.data, rp1.data))
            ]

            # Coordinate update
            with m.If(x == self.hres - 1):
                m.d.sync += [
                    x.eq(0),
                    y.eq(y + 1)
                ]
            with m.Else():
                m.d.sync += x.eq(x + 1)

        with m.If(self.hsync):
            m.d.sync += x.eq(0)
        with m.If(self.vsync):
            m.d.sync += [
                y.eq(0),
                frame_select.eq(~frame_select)
            ]

        # Only one frame is written per cycle
        m.d.comb += [
            wp0.en.eq(self.valid_in & ~frame_select),
            wp1.en.eq(self.valid_in & frame_select),
        ]

        return m


