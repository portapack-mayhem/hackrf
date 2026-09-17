from amaranth.hdl import *
from amaranth.sim import Simulator

from dsp.mixer    import ComplexMultiplier
from util         import IQSample

from cmath        import exp, pi

import unittest


def cround(num):
    return complex(round(num.real), round(num.imag))

def make_exp_vector(bits, length):
    max_val = (1 << (bits-1)) - 1
    return [cround(exp(1j * 2 * pi * i / length) * max_val) for i in range(length)]

class MixerModel:
    def __init__(self, a_bits, b_bits, c_bits):
        self.a_bits = a_bits
        self.b_bits = b_bits
        self.c_bits = c_bits
    def __call__(self, a, b):
        res_width = self.a_bits + self.b_bits
        shift = res_width - self.c_bits - 1
        return cround((a * b) / (2 ** shift))

class TestMixer(unittest.TestCase):

    exp8       = make_exp_vector(8, 256)
    exp8_conj  = [ n.conjugate() for n in exp8 ]
    exp10      = make_exp_vector(10, 256)
    exp10_conj = [ n.conjugate() for n in exp10 ]

    def assertMixEqual(self, mixer, input_a, input_b, expected):

        async def testbench_write(ctx):
            for i in range(len(input_a)):
                ctx.set(mixer.a.p.i, int(input_a[i].real))
                ctx.set(mixer.a.p.q, int(input_a[i].imag))
                ctx.set(mixer.a.valid, 1)
                ctx.set(mixer.b.p.i, int(input_b[i].real))
                ctx.set(mixer.b.p.q, int(input_b[i].imag))
                ctx.set(mixer.b.valid, 1)
                await ctx.tick().until(mixer.a.ready & mixer.b.ready)
            ctx.set(mixer.a.valid, 0)
            ctx.set(mixer.b.valid, 0)
            await ctx.tick()

        async def testbench_read(ctx):
            ctx.set(mixer.c.ready, 1)
            for i, value in enumerate(expected):
                payload, = await ctx.tick().sample(mixer.c.p).until(mixer.c.valid)
                self.assertEqual(payload.i, value.real)
                self.assertEqual(payload.q, value.imag)

        sim = Simulator(mixer)
        sim.add_clock(1/100e6)
        sim.add_testbench(testbench_write)
        sim.add_testbench(testbench_read)
        sim.run()

    def assertEqualsModel(self, mixer, model, input_a, input_b):
        expected = [ model(a, b) for a, b in zip(input_a, input_b) ]
        self.assertMixEqual(mixer, input_a, input_b, expected)

    def test_constant(self):
        """Mix exponential with constant and obtain the same exponential."""
        exp8 = self.exp8
        dut = ComplexMultiplier(IQSample(8), IQSample(10), IQSample(8))
        self.assertMixEqual(dut, exp8, [511] * len(exp8), exp8)
        self.assertMixEqual(dut, exp8, [-512] * len(exp8), [-n for n in exp8])

    def test_compare_with_model(self):
        """Compare against reference mixer models."""
        exp8, exp8_conj = self.exp8, self.exp8_conj
        exp10, exp10_conj = self.exp10, self.exp10_conj

        # 8x8 -> 8
        dut = ComplexMultiplier(IQSample(8), IQSample(8), IQSample(8))
        mdl = MixerModel(8, 8, 8)
        self.assertEqualsModel(dut, mdl, exp8, exp8_conj)

        # 8x10 -> 8
        dut = ComplexMultiplier(IQSample(8), IQSample(10), IQSample(8))
        mdl = MixerModel(8, 10, 8)
        self.assertEqualsModel(dut, mdl, exp8, exp10_conj)

    def test_zero_multiply(self):
        """Multiplying by zero should yield zero."""
        dut = ComplexMultiplier(IQSample(8), IQSample(8), IQSample(8))
        zero_vec = [0] * len(self.exp8)
        self.assertMixEqual(dut, self.exp8, zero_vec, zero_vec)
        self.assertMixEqual(dut, zero_vec, self.exp8, zero_vec)

    def test_sign_inversion(self):
        """Multiply by -1 should invert sign of both components."""
        dut = ComplexMultiplier(IQSample(8), IQSample(8), IQSample(8))
        neg_one = [-128] * len(self.exp8)
        expected = [-n for n in self.exp8]
        self.assertMixEqual(dut, self.exp8, neg_one, expected)


if __name__ == "__main__":
    unittest.main()
