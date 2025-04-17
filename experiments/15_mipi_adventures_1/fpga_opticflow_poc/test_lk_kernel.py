from amaranth.sim import Simulator, Settle
from amaranth import *

# Mock version of LucasKanadeKernel for testbench
class MockLK(Elaboratable):
    def __init__(self):
        self.curr = Array([Signal(8, name=f"curr_{i}") for i in range(9)])
        self.prev = Array([Signal(8, name=f"prev_{i}") for i in range(9)])
        self.u = Signal(signed(16))
        self.v = Signal(signed(16))
        self.valid_in = Signal()
        self.valid_out = Signal()

    def elaborate(self, platform):
        m = Module()

        # Simple test logic: u = curr[4] - prev[4], v = curr[1] - prev[1]
        with m.If(self.valid_in):
            m.d.sync += self.u.eq((self.curr[4] - self.prev[4]))
            m.d.sync += self.v.eq((self.curr[1] - self.prev[1]))
            m.d.sync += self.valid_out.eq(1)
        with m.Else():
            m.d.sync += self.valid_out.eq(0)

        return m

def test_lk_kernel():
    m = Module()
    lk = MockLK()
    m.submodules.lk = lk

    sim = Simulator(m)
    sim.add_clock(1e-6)

    def process():
        yield lk.curr[4].eq(100)
        yield lk.prev[4].eq(90)
        yield lk.curr[1].eq(120)
        yield lk.prev[1].eq(110)
        yield lk.valid_in.eq(1)
        yield
        yield Settle()
        u_val = yield lk.u
        v_val = yield lk.v
        valid = yield lk.valid_out
        print(f"u: {u_val}, v: {v_val}, valid_out: {valid}")
        assert u_val == 10
        assert v_val == 10

    sim.add_sync_process(process)
    sim.run()

test_lk_kernel()


