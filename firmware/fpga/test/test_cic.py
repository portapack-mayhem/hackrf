from collections     import namedtuple

from amaranth.hdl    import *
from amaranth.sim    import Simulator

from dsp.cic         import CICDecimator, CICInterpolator

import numpy as np
import unittest


class CICDecimatorModel:
    def __init__(self, cic_order, factor, M, width_in, width_out):
        # Build taps by convolving boxcar filter repeatedly.
        taps0 = [1 for _ in range(factor*M)]
        taps = [1]
        for i in range(cic_order):
            taps = np.convolve(taps, taps0)
        self.taps = taps
        self.factor = factor
        self.gain = (factor*M)**cic_order
        self.width_gain = 2**(width_out - width_in)
        self.width_out = width_out
    def __call__(self, samples):
        # Compute the expected result.
        #result = np.convolve(samples, self.taps)[:-len(self.taps)-1]
        result = np.convolve(samples, self.taps)
        result = result[::self.factor]                           # decimate
        result = np.round(result * self.width_gain / self.gain)  # scale
        result = np.clip(result, -2**(self.width_out-1), 2**(self.width_out-1)-1)
        result = result.astype(np .int32).tolist()          # convert to python list
        return result

class CICInterpolatorModel:
    def __init__(self, cic_order, factor, M, width_in, width_out):
        # Build taps by convolving boxcar filter repeatedly.
        taps0 = [1 for _ in range(factor*M)]
        taps = [1]
        for i in range(cic_order):
            taps = np.convolve(taps, taps0)
        self.taps = taps
        self.factor = factor
        self.gain = (factor*M)**cic_order // factor
        self.width_gain = 2**(width_out - width_in)
        self.width_out = width_out
    def __call__(self, samples):
        # Compute the expected result.
        result = np.zeros(self.factor * len(samples))
        result[::self.factor] = samples
        result = np.convolve(result, self.taps)
        result = np.round(result * self.width_gain / self.gain)         # scale
        result = np.clip(result, -2**(self.width_out-1), 2**(self.width_out-1)-1)
        result = result.astype(np.int32).tolist()  # convert to python list
        return result


class _TestFilter(unittest.TestCase):

    def _generate_samples(self, count, width):
        # Generate `count` random samples.
        rng = np.random.default_rng(0)
        samples = rng.normal(0, 1, count)

        # Convert to integer.
        samples = np.clip(samples, -1.0, 1.0)
        samples = np.round(samples * (2**(width-1) - 1)).astype(int)
        assert max(samples) < 2**(width-1) and min(samples) >= -2**(width-1)  # sanity check
        return samples

    def _filter(self, dut, samples, count, oob=[], outfile=None):

        async def input_process(ctx):
            if hasattr(dut, "enable"):
                ctx.set(dut.enable, 1)
            for name, value in oob.items():
                ctx.set(getattr(dut, name), value)
            await ctx.tick()
            await ctx.tick()

            ctx.set(dut.input.valid, 1)
            for sample in samples:
                ctx.set(dut.input.payload, [sample.item()])
                await ctx.tick().until(dut.input.ready)
            ctx.set(dut.input.valid, 0)

        filtered = []
        async def output_process(ctx):
            if not dut.output.signature.always_ready:
                ctx.set(dut.output.ready, 1)
            while len(filtered) < count:
                payload, = await ctx.tick().sample(dut.output.payload).until(dut.output.valid)
                filtered.append(payload[0])

        sim = Simulator(dut)
        sim.add_clock(1/100e6)
        sim.add_testbench(input_process)
        sim.add_testbench(output_process)
        if outfile is not None:
            with sim.write_vcd(outfile):
                sim.run()
        else:
            sim.run()
        
        return filtered


class TestCICDecimator(_TestFilter):

    def test_filter(self):
        num_samples = 1024
        test = namedtuple('CICDecimatorTest', ['M', 'order', 'rates', 'factor_log', 'width_in', 'width_out', 'outfile'], defaults=(None,)*7)
        cic_tests = []

        # for different CIC orders...
        for o in [1,2,3,4]:
            # test signal with no rate change
            cic_tests.append(test(M=1, order=o, rates=(1,), factor_log=0, width_in=8, width_out=8))
            cic_tests.append(test(M=2, order=o, rates=(1,), factor_log=0, width_in=8, width_out=8))
            cic_tests.append(test(M=2, order=o, rates=(1,), factor_log=0, width_in=8, width_out=12))

            # test decimation by 4 with different M values and minimum decimation factors
            cic_tests.append(test(M=1, order=o, rates=(1, 2, 4, 8, 16, 32), factor_log=2, width_in=8, width_out=8))
            cic_tests.append(test(M=2, order=o, rates=(1, 2, 4, 8, 16, 32), factor_log=2, width_in=8, width_out=8))
            cic_tests.append(test(M=1, order=o, rates=(2, 4, 8, 16, 32), factor_log=2, width_in=8, width_out=8))
            cic_tests.append(test(M=2, order=o, rates=(2, 4, 8, 16, 32), factor_log=2, width_in=8, width_out=8))
            cic_tests.append(test(M=1, order=o, rates=(4, 8, 16, 32), factor_log=2, width_in=8, width_out=8))

            # different bit widths
            cic_tests.append(test(M=1, order=o, rates=(1, 2, 4, 8, 16, 32), factor_log=2, width_in=8, width_out=9))
            cic_tests.append(test(M=1, order=o, rates=(1, 2, 4, 8, 16, 32), factor_log=2, width_in=8, width_out=10))
            cic_tests.append(test(M=1, order=o, rates=(1, 2, 4, 8, 16, 32), factor_log=0, width_in=8, width_out=12))
            cic_tests.append(test(M=1, order=o, rates=(1, 2, 4, 8, 16, 32), factor_log=1, width_in=8, width_out=12))
            cic_tests.append(test(M=1, order=o, rates=(1, 2, 4, 8, 16, 32), factor_log=2, width_in=8, width_out=12))
            
            # test fixed decimation by 32
            cic_tests.append(test(M=1, order=o, rates=(32,), factor_log=5, width_in=8, width_out=8))


        for t in cic_tests:
            with self.subTest(t):
                factor_log = t.factor_log
                factor = 1 << factor_log
                cic_order = t.order
                M = t.M
                input_samples = self._generate_samples(num_samples, t.width_in)

                model = CICDecimatorModel(cic_order, factor, M, t.width_in, t.width_out)
                expected = model(input_samples)

                # Simulate DUT
                dut = CICDecimator(M, cic_order, t.rates, t.width_in, t.width_out, always_ready=True)
                filtered = self._filter(dut, input_samples, len(input_samples)//factor, oob={"factor":factor_log}, outfile=t.outfile)

                # We allow some error due to internal truncation: expect some samples to differ at most by 1
                max_diff = np.max(np.abs(np.array(filtered) - np.array(expected[:len(filtered)])))
                
                self.assertLessEqual(max_diff, 1)
                #self.assertListEqual(expected[:len(filtered)], filtered)


    def test_overflow_does_not_happen(self):
        num_samples = 1024
        input_width = 8
        output_width = 8
        input_samples = np.array([127] * num_samples)

        factor_log = 2
        factor = 4
        cic_order = 3
        M = 2

        model = CICDecimatorModel(cic_order, factor, M, input_width, output_width)
        expected = model(input_samples)

        # Simulate DUT
        dut = CICDecimator(M, cic_order, (1,2,4,8,16,32), input_width, output_width, always_ready=True)
        filtered = self._filter(dut, input_samples, len(input_samples)//factor, oob={"factor":factor_log})

        # We allow some error due to internal truncation: expect some samples to differ at most by 1
        max_diff = np.max(np.abs(np.array(filtered) - np.array(expected[:len(filtered)])))
        
        self.assertLessEqual(max_diff, 1)


class TestCICInterpolator(_TestFilter):

    def test_filter(self):
        num_samples = 1024
        test = namedtuple('CICInterpolatorTest', ['M', 'order', 'rates', 'factor_log', 'width_in', 'width_out', 'outfile'], defaults=(None,)*7)
        cic_tests = []

        # for different CIC orders...
        for o in [1,2,3,4]:
            # test signal bypass
            cic_tests.append(test(M=1, order=o, rates=(1,), factor_log=0, width_in=8, width_out=8))
            cic_tests.append(test(M=1, order=o, rates=(1,), factor_log=0, width_in=12, width_out=8))

            # test interpolation by 4 with different M values and minimum interpolation factors
            cic_tests.append(test(M=1, order=o, rates=(1, 2, 4, 8, 16, 32), factor_log=2, width_in=8, width_out=8))
            cic_tests.append(test(M=2, order=o, rates=(1, 2, 4, 8, 16, 32), factor_log=2, width_in=8, width_out=8))
            cic_tests.append(test(M=1, order=o, rates=(2, 4, 8, 16, 32), factor_log=2, width_in=8, width_out=8))
            cic_tests.append(test(M=2, order=o, rates=(2, 4, 8, 16, 32), factor_log=2, width_in=8, width_out=8))
            cic_tests.append(test(M=1, order=o, rates=(4, 8, 16, 32), factor_log=2, width_in=8, width_out=8))

            # different bit widths
            cic_tests.append(test(M=1, order=o, rates=(1, 2, 4, 8, 16, 32), factor_log=2, width_in=16, width_out=8))
            cic_tests.append(test(M=2, order=o, rates=(1, 2, 4, 8, 16, 32), factor_log=2, width_in=16, width_out=8))
            cic_tests.append(test(M=1, order=o, rates=(2, 4, 8, 16, 32), factor_log=2, width_in=16, width_out=8))

            # test fixed interpolation by 32
            cic_tests.append(test(M=1, order=o, rates=(32,), factor_log=5, width_in=8, width_out=24))

            cic_tests.append(test(M=1, order=o, rates=(32,), factor_log=5, width_in=12, width_out=8))

        for t in cic_tests:
            with self.subTest(t):

                input_samples = self._generate_samples(num_samples, t.width_in)

                factor_log = t.factor_log
                factor = 1 << factor_log
                cic_order = t.order
                M = t.M

                model = CICInterpolatorModel(cic_order, factor, M, t.width_in, t.width_out)
                expected = model(input_samples)

                # Simulate DUT
                dut = CICInterpolator(M, cic_order, t.rates, t.width_in, t.width_out, always_ready=False)
                filtered = self._filter(dut, input_samples, len(input_samples)*factor, oob={"factor":factor_log}, outfile=t.outfile)
                
                self.assertListEqual(expected[:len(filtered)], filtered)


if __name__ == "__main__":
    unittest.main()