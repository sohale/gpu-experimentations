#  Streaming Wrapper for Lucas-Kanade Flow

# A complete streaming optical flow pipeline for 8-bit grayscale video
# Real-time computation of flow vectors (u, v)
# Easy to tile or replicate for multi-pixel or region-based processing
# Line-buffered to minimize RAM use

# todo:
# Add thresholds to mask small or noisy motion
# Color-encode (u, v) vectors into HSV/RGB
# Sum flow vectors over region to classify motion
# Add FIFO to decouple flow core from output interface


from amaranth import *
from lucas_kanade import LucasKanadeKernel


class FlowStreamWrapper(Elaboratable):
    def __init__(self, hres=640, vres=480):
        self.pixel_in = Signal(8)
        self.valid_in = Signal()
        self.hsync = Signal()
        self.vsync = Signal()
        self.clk_pix = ClockSignal("pix")

        # Output motion vectors
        self.u = Signal(signed(16))
        self.v = Signal(signed(16))
        self.valid_out = Signal()

        self.hres = hres
        self.vres = vres

    def elaborate(self, platform):
        m = Module()

        # Line buffers (current and previous)
        curr_line_1 = Memory(width=8, depth=self.hres)
        curr_line_2 = Memory(width=8, depth=self.hres)
        prev_line_1 = Memory(width=8, depth=self.hres)
        prev_line_2 = Memory(width=8, depth=self.hres)

        # Read/write ports
        rp_curr1 = m.submodules.rp_curr1 = curr_line_1.read_port()
        rp_curr2 = m.submodules.rp_curr2 = curr_line_2.read_port()
        wp_curr = m.submodules.wp_curr = curr_line_2.write_port()

        rp_prev1 = m.submodules.rp_prev1 = prev_line_1.read_port()
        rp_prev2 = m.submodules.rp_prev2 = prev_line_2.read_port()
        wp_prev = m.submodules.wp_prev = prev_line_2.write_port()

        # Kernel instantiation
        m.submodules.lk = lk = LucasKanadeKernel()

        # X and Y counters
        x = Signal(range(self.hres))
        y = Signal(range(self.vres))
        line_toggle = Signal()

        # Shifting window for curr and prev frames
        curr_window = Array([Signal(8) for _ in range(9)])
        prev_window = Array([Signal(8) for _ in range(9)])

        # Main FSM / logic
        with m.If(self.valid_in):
            # Shift current and previous window pixels
            m.d.sync += [
                curr_window[0].eq(curr_window[1]),
                curr_window[1].eq(curr_window[2]),
                curr_window[3].eq(curr_window[4]),
                curr_window[4].eq(curr_window[5]),
                curr_window[6].eq(curr_window[7]),
                curr_window[7].eq(curr_window[8]),
                curr_window[2].eq(rp_curr1.data),
                curr_window[5].eq(rp_curr2.data),
                curr_window[8].eq(self.pixel_in),

                prev_window[0].eq(prev_window[1]),
                prev_window[1].eq(prev_window[2]),
                prev_window[3].eq(prev_window[4]),
                prev_window[4].eq(prev_window[5]),
                prev_window[6].eq(prev_window[7]),
                prev_window[7].eq(prev_window[8]),
                prev_window[2].eq(rp_prev1.data),
                prev_window[5].eq(rp_prev2.data),
                prev_window[8].eq(self.pixel_in),  # crude approximation
            ]

            # Write current pixel to line buffers
            m.d.sync += [
                wp_curr.addr.eq(x),
                wp_curr.data.eq(self.pixel_in),
                wp_curr.en.eq(1),
                wp_prev.addr.eq(x),
                wp_prev.data.eq(curr_window[4]),  # middle of last frame
                wp_prev.en.eq(1),
            ]

            # Update counters
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
                line_toggle.eq(~line_toggle)
            ]

        # Assign to kernel
        for i in range(9):
            m.d.sync += [
                lk.curr[i].eq(curr_window[i]),
                lk.prev[i].eq(prev_window[i])
            ]

        m.d.comb += [
            lk.valid_in.eq(self.valid_in),
            self.u.eq(lk.u),
            self.v.eq(lk.v),
            self.valid_out.eq(lk.valid_out)
        ]

        return m


