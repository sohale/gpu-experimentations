from amaranth import *

class LucasKanadeKernel(Elaboratable):
    """
    Simplified Lucas-Kanade optical flow kernel operating on a 3x3 window.
    Inputs: current and previous 3x3 grayscale window pixels.
    Outputs: integer estimate of flow vector (u, v) at the center pixel.
    """
    def __init__(self):
        # 3x3 grayscale pixels, current frame
        self.curr = Array([Signal(8, name=f"curr_{i}") for i in range(9)])
        # 3x3 grayscale pixels, previous frame
        self.prev = Array([Signal(8, name=f"prev_{i}") for i in range(9)])

        # Output motion vector components
        self.u = Signal(signed(16))
        self.v = Signal(signed(16))

        # Control signals
        self.valid_in = Signal()
        self.valid_out = Signal()

    def elaborate(self, platform):
        m = Module()

        # Signals for gradients
        Ix = Signal(signed(16))
        Iy = Signal(signed(16))
        It = Signal(signed(16))

        Ix_sq = Signal(signed(32))
        Iy_sq = Signal(signed(32))
        denom = Signal(signed(32))

        u_tmp = Signal(signed(32))
        v_tmp = Signal(signed(32))

        with m.If(self.valid_in):
            # Approximate gradients
            m.d.sync += Ix.eq(self.curr[5] - self.curr[3])       # center-right - center-left
            m.d.sync += Iy.eq(self.curr[7] - self.curr[1])       # bottom-center - top-center
            m.d.sync += It.eq(self.curr[4] - self.prev[4])       # center pixel diff

            # Prepare denominator (Ix² + Iy² + 1)
            m.d.sync += Ix_sq.eq(Ix * Ix)
            m.d.sync += Iy_sq.eq(Iy * Iy)
            m.d.sync += denom.eq(Ix_sq + Iy_sq + 1)  # Avoid divide-by-zero

            # Compute flow (scaled integer approx)
            m.d.sync += u_tmp.eq(-It * Ix)
            m.d.sync += v_tmp.eq(-It * Iy)

            m.d.sync += self.u.eq(u_tmp // denom)
            m.d.sync += self.v.eq(v_tmp // denom)

            m.d.sync += self.valid_out.eq(1)
        with m.Else():
            m.d.sync += self.valid_out.eq(0)

        return m


