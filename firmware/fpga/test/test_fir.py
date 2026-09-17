from collections.abc import Iterable
from contextlib      import nullcontext

from amaranth.hdl    import *
from amaranth.lib    import data
from amaranth.sim    import Simulator

from amaranth_future import fixed
from dsp.fir         import FIRFilter, HalfBandDecimator, HalfBandInterpolator

import numpy as np
import unittest


class FIRModel:
    def __init__(self, taps, shape_out):
        self.taps = taps
        self.shape_out = shape_out
    def __call__(self, samples):
        shape_out = self.shape_out
        result = np.convolve(samples, self.taps)[:-(len(self.taps)-1)]
        result = np.round(result * 2**shape_out.f_bits) / 2**shape_out.f_bits
        result = np.clip(result, shape_out.min().as_float(), shape_out.max().as_float())
        return result.tolist()


class _TestFilter(unittest.TestCase):

    rng = np.random.default_rng(0)

    def _generate_samples(self, count, width, f_bits=0):
        # Generate `count` random samples.
        samples = self.rng.normal(0, 1, count)

        # Convert to integer.
        samples = np.clip(samples, -1.0, 1.0)
        samples = np.round(samples * (2**(width-1) - 1)).astype(int)
        assert max(samples) < 2**(width-1) and min(samples) >= -2**(width-1)  # sanity check

        return samples / 2**f_bits

    def _filter(self, dut, samples, count, outfile=None,
        empty_valid_cycles=0, empty_ready_cycles=0, deadline=None):

        write_done = Signal()
        read_done = Signal()

        filtered = []

        async def input_process(ctx):
            if hasattr(dut, "enable"):
                ctx.set(dut.enable, 1)
            await ctx.tick()
            for i, sample in enumerate(samples):
                if isinstance(dut.input.payload.shape(), data.ArrayLayout):
                    if not isinstance(sample, Iterable):
                        sample = (sample,)
                ctx.set(dut.input.payload, sample)
                ctx.set(dut.input.valid, 1)
                await ctx.tick().until(dut.input.ready)
                if empty_valid_cycles > 0:
                    ctx.set(dut.input.valid, 0)
                    await ctx.tick().repeat(empty_valid_cycles)
            ctx.set(dut.input.valid, 0)
            ctx.set(write_done, 1)
        
        async def output_process(ctx):
            while len(filtered) < count:
                payload, = (await ctx.tick()
                    .sample(dut.output.payload)
                    .until(dut.output.valid & dut.output.ready))
                if isinstance(payload.shape(), data.ArrayLayout):
                    if len(payload) > 1:
                        filtered.append([p.as_float() for p in payload])
                    else:
                        filtered.append(payload[0].as_float())
                else:
                    filtered.append(payload.as_float())
            ctx.set(read_done, 1)

        async def ready_process(ctx):
            ctx.set(dut.output.ready, 1)
            if empty_ready_cycles > 0:
                while not ctx.get(read_done):
                    await ctx.tick().until(dut.output.valid)
                    ctx.set(dut.output.ready, 0)
                    await ctx.tick().repeat(empty_ready_cycles)
                    ctx.set(dut.output.ready, 1)

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(input_process)
        sim.add_testbench(output_process)
        if not dut.output.signature.always_ready:
            sim.add_testbench(ready_process)
        context = sim.write_vcd(outfile) if outfile else nullcontext()
        with context:
            if deadline is None:
                sim.run()
            else:
                sim.run_until(1e-6 * deadline)
        
        return filtered


class TestFIRFilter(_TestFilter):

    def test_filter(self):
        taps = [-1, 0, 9, 16, 9, 0, -1]
        taps = [ tap / 32 for tap in taps ]

        num_samples = 1024
        input_width = 8
        input_samples = self._generate_samples(num_samples, input_width, f_bits=7)

        # Simulate DUT
        dut = FIRFilter(taps, shape=fixed.SQ(1, 7), always_ready=False)
        filtered = self._filter(dut, input_samples, len(input_samples), empty_ready_cycles=5)

        # Compare with FIR filter model.
        expected = FIRModel(taps, dut.shape_out)(input_samples)
        self.assertListEqual(filtered, expected)

    def test_filter_with_rounding_and_saturation(self):
        taps = [-1, 0, 9, 16, 9, 0, -1]
        taps = [ tap / 32 for tap in taps ]

        num_samples = 1024
        input_width = 8
        input_samples = self._generate_samples(num_samples, input_width, f_bits=7)

        # Simulate DUT
        dut = FIRFilter(taps, shape=fixed.SQ(1, 7), shape_out=fixed.SQ(1, 7), always_ready=False)
        filtered = self._filter(dut, input_samples, len(input_samples), empty_ready_cycles=5)

        # Compare with FIR filter model.
        expected = FIRModel(taps, dut.shape_out)(input_samples)
        self.assertListEqual(filtered, expected)


class TestHalfBandDecimator(_TestFilter):

    def test_filter(self):

        common_dut_options = dict(
            data_shape=fixed.SQ(1,7),
        )

        taps0 = (np.array([-1, 0, 9, 16, 9, 0, -1]) / 32).tolist()
        taps1 = (np.array([-2, 0, 7, 0, -18, 0, 41, 0, -92, 0, 320, 512, 320, 0, -92, 0, 41, 0, -18, 0, 7, 0, -2]) / 1024).tolist()

        inputs = {

            "test_filter_with_backpressure": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, always_ready=False, taps=taps0),
            },

            "test_filter_with_backpressure_and_rounding": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, shape_out=fixed.SQ(2, 7), always_ready=False, taps=taps0),
            },

            "test_filter_with_backpressure_and_empty_valid_cycles": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, always_ready=False, taps=taps0),
                "sim_opts": dict(empty_valid_cycles=3),
            },

            "test_filter_with_backpressure_taps1": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, always_ready=False, taps=taps1),
            },

            "test_filter_with_backpressure_taps1_rounding_and_saturation": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, shape_out=fixed.SQ(1, 7), always_ready=False, taps=taps1),
            },

            "test_filter_no_backpressure_and_empty_valid_cycles_taps1": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, always_ready=True, taps=taps0),
                "sim_opts": dict(empty_valid_cycles=6),
            },

            "test_filter_no_backpressure": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, always_ready=True, taps=taps1),
                "sim_opts": dict(empty_valid_cycles=3),
            },
        }
        
        for name, scenario in inputs.items():

            with self.subTest(name):
                taps        = scenario["dut_options"]["taps"]
                num_samples = scenario["num_samples"]

                input_width = 8
                samples_i_in = self._generate_samples(num_samples, input_width, f_bits=7)
                samples_q_in = self._generate_samples(num_samples, input_width, f_bits=7)

                # Simulate DUT
                dut = HalfBandDecimator(**scenario["dut_options"])
                filtered = self._filter(dut, zip(samples_i_in, samples_q_in), len(samples_i_in) // 2, **scenario.get("sim_opts",{}))
                filtered_i = [ x[0] for x in filtered ]
                filtered_q = [ x[1] for x in filtered ]

                # Compare with FIR filter model.
                firmodel = FIRModel(taps, dut.shape_out)
                expected_i = firmodel(samples_i_in)[1::2]
                expected_q = firmodel(samples_q_in)[1::2]
                self.assertListEqual(expected_i, filtered_i)
                self.assertListEqual(expected_q, filtered_q)


class TestHalfBandInterpolator(_TestFilter):

    def test_filter(self):

        common_dut_options = dict(
            data_shape=fixed.SQ(1,7),
            shape_out=fixed.SQ(1, 7),
        )

        taps0 = (np.array([-1, 0, 9, 16, 9, 0, -1]) / 32).tolist()
        taps1 = (np.array([-2, 0, 7, 0, -18, 0, 41, 0, -92, 0, 320, 512, 320, 0, -92, 0, 41, 0, -18, 0, 7, 0, -2]) / 1024).tolist()

        inputs = {

            "test_filter_with_backpressure": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, always_ready=False, taps=taps1),
            },

            "test_filter_with_backpressure_and_rounding": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, always_ready=False, taps=taps1),
            },

            "test_filter_with_backpressure_and_empty_valid_cycles": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, always_ready=False, taps=taps0),
                "sim_opts": dict(empty_ready_cycles=7, empty_valid_cycles=3),
            },

            "test_filter_with_backpressure_taps1": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, always_ready=False, taps=taps1),
                "sim_opts": dict(empty_ready_cycles=7),
            },

            "test_filter_no_backpressure_and_empty_valid_cycles_taps1": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, always_ready=True, taps=taps0),
                "sim_opts": dict(empty_valid_cycles=1, deadline=1024*2+10),
            },

            "test_filter_no_backpressure": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, always_ready=True, taps=taps1),
                "sim_opts": dict(empty_valid_cycles=1, deadline=1024*2+10),
            },

        }

        def upsample_x2(samples):
            samples_pad = np.zeros(2 * len(samples))
            samples_pad[0::2] = 2 * samples  # pad with zeros, adjust gain
            return samples_pad
    
        for name, scenario in inputs.items():
            with self.subTest(name):
                taps        = scenario["dut_options"]["taps"]
                num_samples = scenario["num_samples"]

                input_width = 8
                samples_i_in = self._generate_samples(num_samples, input_width, f_bits=7)
                samples_q_in = self._generate_samples(num_samples, input_width, f_bits=7)

                # Simulate DUT
                dut = HalfBandInterpolator(**scenario["dut_options"])
                filtered = self._filter(dut, zip(samples_i_in, samples_q_in), len(samples_i_in) * 2, **scenario.get("sim_opts",{}))
                filtered_i = [ x[0] for x in filtered ]
                filtered_q = [ x[1] for x in filtered ]

                # Compare with FIR filter model.
                firmodel = FIRModel(taps, dut.shape_out)
                expected_i = firmodel(upsample_x2(samples_i_in))
                expected_q = firmodel(upsample_x2(samples_q_in))
                self.assertListEqual(expected_i, filtered_i)
                self.assertListEqual(expected_q, filtered_q)


if __name__ == "__main__":
    unittest.main()
