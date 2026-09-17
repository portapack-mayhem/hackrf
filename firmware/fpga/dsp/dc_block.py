#
# This file is part of HackRF.
#
# Copyright (c) 2025 Great Scott Gadgets <info@greatscottgadgets.com>
# SPDX-License-Identifier: BSD-3-Clause

from amaranth               import Module, Signal, Mux, signed, DomainRenamer, Cat, EnableInserter
from amaranth.lib           import wiring, stream, data
from amaranth.lib.wiring    import In, Out


class DCBlock(wiring.Component):
    """
    DC blocking filter with dithering

    Removes DC offset using a leaky integrator:
        y[n] = x[n] - avg[n-1]
        avg[n] = alpha * y[n] + avg[n-1]
    where alpha is the leakage coefficient (2**-ratio).
    """
    def __init__(self, width, ratio=12, num_channels=1, always_ready=True, domain="sync"):
        self.ratio = ratio
        self.width = width
        self.num_channels = num_channels
        self.domain = domain

        sig = stream.Signature(
            data.ArrayLayout(signed(width), num_channels),
            always_ready=always_ready
        )
        super().__init__({
            "input":  In(sig),
            "output": Out(sig),
            "enable": In(1),
        })

    def elaborate(self, platform):
        m = Module()

        # Resync control signaling.
        enable    = Signal()
        m.d.sync += enable.eq(self.enable)

        # Fixed-point configuration.
        ratio         = self.ratio
        input_shape   = signed(self.width)
        ext_precision = signed(self.width + ratio)
        
        # Shared PRNG for all channels.
        prng_en = Signal()
        m.submodules.prng = prng = EnableInserter(prng_en)(GaloisLFSR32())
        prng_bits = prng.output

        # Common signaling.
        with m.If(~self.output.valid | self.output.ready):
            if not self.input.signature.always_ready:
                m.d.comb += self.input.ready.eq(1)
            m.d.sync += self.output.valid.eq(self.input.valid)

        with m.If(self.input.ready & self.input.valid):
            m.d.comb += prng_en.eq(1)

        def saturating_sub(a, b):
            r = a - b
            r_sat = Cat((~r[-1]).replicate(self.width-1), r[-1])
            overflow = r[-1] ^ r[-2]  # sign bit of the result different from carry (top 2 bits)
            return Mux(overflow, r_sat, r[:self.width].as_signed())

        # Per-channel processing.
        for c in range(self.num_channels):

            # Signal declarations.
            sample_in = self.input.p[c]
            y         = Signal(input_shape)      # current output
            y_q       = Signal(input_shape)      # last output
            avg       = Signal(ext_precision)    # current average
            avg_q     = Signal(ext_precision)    # last average
            qavg      = Signal(input_shape)      # quantized avg
            qavg_q    = Signal(input_shape)      # last quantized avg
            error     = Signal(ratio)
            dither    = Signal(ratio)            # dither pattern

            # Generate unique dither pattern for each channel.
            m.d.sync += dither.eq(prng_bits.word_select(c, ratio))

            with m.If(self.input.valid & self.input.ready):

                m.d.sync += [
                    y_q         .eq(y),
                    avg_q       .eq(avg),
                    qavg_q      .eq(qavg),
                ]

                m.d.comb += [
                    y           .eq(saturating_sub(sample_in, qavg_q)),
                    avg         .eq(avg_q + y_q),  # update avg

                    # Truncate with dithering, discard quantization error.
                    Cat(error, qavg).eq(avg + dither),
                ]

                m.d.sync += self.output.p[c].eq(Mux(enable, y, self.input.p[c]))

        if self.domain != "sync":
            m = DomainRenamer(self.domain)(m)

        return m


class GaloisLFSR32(wiring.Component):
    output: Out(32)

    def __init__(self, poly=0x80200003, seed=1):
        assert poly != 0
        assert seed != 0
        self.poly = poly
        self.seed = seed
        super().__init__()

    def elaborate(self, platform):
        m = Module()

        state    = Signal(32, init=self.seed)
        feedback = state[0]

        next_state = Signal.like(state)
        m.d.comb += next_state.eq((state >> 1) ^ Mux(feedback, self.poly, 0))
        
        m.d.sync += state.eq(next_state)
        m.d.comb += self.output.eq(state)

        return m
