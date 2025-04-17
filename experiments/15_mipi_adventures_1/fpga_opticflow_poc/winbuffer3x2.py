# AmaranthHDL Module: Line Buffer for 3x2 Window
#    Generates a 3x2 buffer
#    Assumes 1 pixel per clock cycle 
#    Feed this module with correctly synchronized pixel streams 
#    Use MIPI → DVP bridge or Lattice IP


from amaranth import *
from amaranth.sim import Simulator


class LineBufferedWindow(Elaboratable):
    def __init__(self, hres=640):
        self.pixel_in = Signal(8)
        self.valid_in = Signal()
        self.hsync = Signal()
        self.vsync = Signal()
        self.pix_clk = ClockSignal("pix")

        # Output: 3x2 neighborhood (3 pixels from 2 lines)
        self.window = Array([Signal(8, name=f"px{i}") for i in range(6)])
        self.valid_out = Signal()

        # Config
        self.hres = hres

    def elaborate(self, platform):
        m = Module()

        # Line buffers (shift registers / RAMs)
        linebuf_1 = Memory(width=8, depth=self.hres)
        linebuf_2 = Memory(width=8, depth=self.hres)

        rd_addr = Signal(range(self.hres))
        wr_addr = Signal(range(self.hres))

        rp1 = m.submodules.rp1 = linebuf_1.read_port(transparent=False)
        wp1 = m.submodules.wp1 = linebuf_1.write_port()
        rp2 = m.submodules.rp2 = linebuf_2.read_port(transparent=False)
        wp2 = m.submodules.wp2 = linebuf_2.write_port()

        line_select = Signal(reset=0)

        # Shift register for current line (3 pixels)
        curr_line = [Signal(8) for _ in range(3)]

        with m.If(self.valid_in):
            # Shift current line
            m.d.pix += [
                curr_line[2].eq(curr_line[1]),
                curr_line[1].eq(curr_line[0]),
                curr_line[0].eq(self.pixel_in)
            ]

            # Read from buffer(s)
            m.d.pix += [
                rp1.addr.eq(wr_addr),
                rp2.addr.eq(wr_addr)
            ]

            # Write to current line buffer
            with m.If(line_select == 0):
                m.d.pix += wp1.addr.eq(wr_addr)
                m.d.pix += wp1.data.eq(self.pixel_in)
                m.d.pix += wp1.en.eq(1)
            with m.Else():
                m.d.pix += wp2.addr.eq(wr_addr)
                m.d.pix += wp2.data.eq(self.pixel_in)
                m.d.pix += wp2.en.eq(1)

            # Assign window outputs
            with m.If(line_select == 0):
                m.d.pix += [
                    self.window[0].eq(rp2.data),  # top left
                    self.window[1].eq(rp2.data),  # top mid
                    self.window[2].eq(rp2.data),  # top right
                ]
                m.d.pix += [
                    self.window[3].eq(curr_line[2]),
                    self.window[4].eq(curr_line[1]),
                    self.window[5].eq(curr_line[0]),
                ]
            with m.Else():
                m.d.pix += [
                    self.window[0].eq(rp1.data),
                    self.window[1].eq(rp1.data),
                    self.window[2].eq(rp1.data),
                ]
                m.d.pix += [
                    self.window[3].eq(curr_line[2]),
                    self.window[4].eq(curr_line[1]),
                    self.window[5].eq(curr_line[0]),
                ]

            m.d.pix += [
                self.valid_out.eq(1),
                wr_addr.eq(wr_addr + 1)
            ]

        with m.If(self.hsync):
            m.d.pix += wr_addr.eq(0)

        with m.If(self.vsync):
            m.d.pix += line_select.eq(~line_select)

        return m


