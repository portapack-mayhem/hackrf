from amaranth.hdl    import *

from amaranth_future import fixed
from dsp.fir_mac16   import FIRFilterMAC16, HalfBandDecimatorMAC16, HalfBandInterpolatorMAC16

import numpy as np
import unittest

from test.test_fir   import _TestFilter, FIRModel


class TestFIRFilterMAC16(_TestFilter):

    def test_filter(self):
        taps = [-1, 0, 9, 16, 9, 0, -1]
        taps = [ tap / 32 for tap in taps ]

        num_samples = 1024
        input_width = 8
        input_samples = self._generate_samples(num_samples, input_width)

        # Simulate DUT
        dut = FIRFilterMAC16(taps, shape=fixed.SQ(8, 0), always_ready=False)
        filtered = self._filter(dut, input_samples, len(input_samples), empty_ready_cycles=5)

        # Compare with FIR filter model.
        expected = FIRModel(taps, dut.shape_out)(input_samples)
        self.assertListEqual(filtered, expected)


class TestHalfBandDecimatorMAC16(_TestFilter):

    def test_filter(self):

        common_dut_options = dict(
            data_shape=fixed.SQ(1,7),
            overclock_rate=4,
        )

        taps0 = (np.array([-1, 0, 9, 16, 9, 0, -1]) / 32).tolist()
        taps1 = (np.array([-2, 0, 7, 0, -18, 0, 41, 0, -92, 0, 320, 512, 320, 0, -92, 0, 41, 0, -18, 0, 7, 0, -2]) / 1024).tolist()

        inputs = {

            "test_filter_with_backpressure": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, always_ready=False, taps=taps0),
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
                dut = HalfBandDecimatorMAC16(**scenario["dut_options"])
                filtered = self._filter(dut, zip(samples_i_in, samples_q_in), len(samples_i_in) // 2, **scenario.get("sim_opts",{}))
                filtered_i = [ x[0] for x in filtered ]
                filtered_q = [ x[1] for x in filtered ]

                # Compare with FIR filter model.
                firmodel = FIRModel(taps, dut.shape_out)
                expected_i = firmodel(samples_i_in)[1::2]
                expected_q = firmodel(samples_q_in)[1::2]
                self.assertListEqual(expected_i, filtered_i)
                self.assertListEqual(expected_q, filtered_q)


class TestHalfBandInterpolatorMAC16(_TestFilter):

    def test_filter(self):

        common_dut_options = dict(
            data_shape=fixed.SQ(1,7),
            shape_out=fixed.SQ(1,7),
            overclock_rate=4,
        )

        taps0 = (np.array([-1, 0, 9, 16, 9, 0, -1]) / 32).tolist()
        taps1 = (np.array([-2, 0, 7, 0, -18, 0, 41, 0, -92, 0, 320, 512, 320, 0, -92, 0, 41, 0, -18, 0, 7, 0, -2]) / 1024).tolist()

        inputs = {

            "test_filter_with_backpressure": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, always_ready=False, num_channels=2, taps=taps1),
            },

            "test_filter_with_backpressure_and_empty_valid_cycles": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, num_channels=2, always_ready=False, taps=taps0),
                "sim_opts": dict(empty_ready_cycles=7, empty_valid_cycles=3),
            },

            "test_filter_with_backpressure_taps1": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, num_channels=2, always_ready=False, taps=taps1),
                "sim_opts": dict(empty_ready_cycles=7, empty_valid_cycles=0),
            },

            "test_filter_no_backpressure_and_empty_valid_cycles_taps1": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, num_channels=2, always_ready=True, taps=taps0),
                "sim_opts": dict(empty_valid_cycles=8, deadline=1024*10),
            },

            "test_filter_no_backpressure": {
                "num_samples": 1024,
                "dut_options": dict(**common_dut_options, num_channels=2, always_ready=True, taps=taps1),
                "sim_opts": dict(empty_valid_cycles=16),
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
                dut = HalfBandInterpolatorMAC16(**scenario["dut_options"])
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
