

from amaranth import *

class LucasKanadeKernel(Elaboratable):
    """
    A simplified Lucas-Kanade optical flow kernel operating on a 3x3 window.
    Approximates gradient components and computes u,v motion vectors assuming:
    - I_x = ∂I/∂x
    - I_y = ∂I/∂y
    - I_t = ∂I/∂t
    Assumes windowed input data from previous frame buffers and current frame.
    Outputs crude estimates of u,v vectors using integer math.
    """
    def __init__(self):
        # Inputs: pixel values from current and previous frame (3x3 windows)
        self.curr = Array([Signal(8, name=f"curr_{i}") for i in range(9)])
        self.prev = Array([Signal(8, name=f"prev_{i}") for i in range(9)])

        # Outputs: motion vector u, v (signed 16-bit)
        self.u = Signal(signed(16))
        self.v = Signal(signed(16))

        # Control
        self.valid_in = Signal()
        self.valid_out = Signal()

    def elaborate(self, platform):
        m = Module()

        # Gradient approximations
        Ix = Signal(signed(16))
        Iy = Signal(signed(16))
        It = Signal(signed(16))

        # Gradient and error terms for central pixel
        with m.If(self.valid_in):
            # Horizontal gradient (Ix): center-right - center-left
            m.d.sync += Ix.eq(self.curr[5] - self.curr[3])
            # Vertical gradient (Iy): bottom-center - top-center
            m.d.sync += Iy.eq(self.curr[7] - self.curr[1])
            # Temporal gradient (It): center - prev_center
            m.d.sync += It.eq(self.curr[4] - self.prev[4])

            # Very simplified Lucas-Kanade kernel (just one point)
            # Normally you'd sum over window: (A^T A)^-1 A^T b
            # We approximate u = -It * Ix / (Ix^2 + Iy^2)
            # and v = -It * Iy / (Ix^2 + Iy^2)
            Ix_sq = Signal(signed(32))
            Iy_sq = Signal(signed(32))
            denom = Signal(signed(32))
            u_tmp = Signal(signed(32))
            v_tmp = Signal(signed(32))

            m.d.sync += Ix_sq.eq(Ix * Ix)
            m.d.sync += Iy_sq.eq(Iy * Iy)
            m.d.sync += denom.eq(Ix_sq + Iy_sq + 1)  # add +1 to avoid div by 0

            m.d.sync += u_tmp.eq(-It * Ix)
            m.d.sync += v_tmp.eq(-It * Iy)

            m.d.sync += self.u.eq(u_tmp // denom)
            m.d.sync += self.v.eq(v_tmp // denom)

            m.d.sync += self.valid_out.eq(1)
        with m.Else():
            m.d.sync += self.valid_out.eq(0)

        return m


