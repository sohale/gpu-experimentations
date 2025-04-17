from amaranth import *
from amaranth.sim import Simulator, Settle


class MockLK(Elaboratable):
    def __init__(self):
        self.curr = Array([Signal(8) for _ in range(9)])
        self.prev = Array([Signal(8) for _ in range(9)])
        self.u = Signal(signed(16))
        self.v = Signal(signed(16))
        self.valid_in = Signal()
        self.valid_out = Signal()

    def elaborate(self, platform):
        m = Module()
        with m.If(self.valid_in):
            m.d.sync += [
                self.u.eq(self.curr[4] - self.prev[4]),
                self.v.eq(self.curr[1] - self.prev[1]),
                self.valid_out.eq(1),
            ]
        with m.Else():
            m.d.sync += self.valid_out.eq(0)
        return m


def simulate():
    m = Module()
    dut = MockLK()
    m.submodules.dut = dut

    def proc():
        yield dut.curr[4].eq(100)
        yield dut.prev[4].eq(90)
        yield dut.curr[1].eq(120)
        yield dut.prev[1].eq(110)
        yield dut.valid_in.eq(1)
        yield
        yield Settle()
        u = yield dut.u
        v = yield dut.v
        valid = yield dut.valid_out
        print(f"u = {u}, v = {v}, valid_out = {valid}")

    sim = Simulator(m)
    sim.add_clock(1e-6)
    sim.add_sync_process(proc)
    sim.run()


if __name__ == "__main__":
    simulate()

