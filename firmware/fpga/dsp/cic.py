#
# This file is part of HackRF.
#
# Copyright (c) 2025 Great Scott Gadgets <info@greatscottgadgets.com>
# SPDX-License-Identifier: BSD-3-Clause

from math                   import floor, log2, ceil, comb

from amaranth               import Module, Signal, Const, Mux, signed, ResetInserter, DomainRenamer
from amaranth.utils         import bits_for, exact_log2

from amaranth.lib           import wiring, stream, data
from amaranth.lib.wiring    import In, Out, connect

from dsp.round              import convergent_round


class CICInterpolator(wiring.Component):
    def __init__(self, M, stages, rates, width_in, width_out=None, num_channels=1, always_ready=False, domain="sync"):
        assert all((r & (r-1)) == 0 for r in rates), "rates must be powers of 2"
        self.M         = M
        self.stages    = stages
        self.rates     = rates
        self.width_in  = width_in
        self.num_channels = num_channels
        if width_out is None:
            width_out  = width_in + self.bit_growths()[-1]
        self.width_out = width_out
        self._domain = domain
        super().__init__({
            "input":  In(stream.Signature(
                data.ArrayLayout(signed(width_in), num_channels),
                always_ready=always_ready
            )),
            "output": Out(stream.Signature(
                data.ArrayLayout(signed(width_out), num_channels),
                always_ready=always_ready
            )),
            "factor": In(range(bits_for(max(rates)))),
        })

    def bit_growths(self):
        bit_growths = cic_growth(N=self.stages, M=self.M, R=max(self.rates))
        return bit_growths

    def elaborate(self, platform):
        m = Module()

        always_ready = self.output.signature.always_ready

        # Detect interpolation factor changes to provide an internal reset signal.
        factor_last = Signal.like(self.factor)
        factor_change = Signal()
        m.d.sync += factor_last.eq(self.factor)
        m.d.sync += factor_change.eq(factor_last != self.factor)
        factor_reset = ResetInserter(factor_change)

        # Calculated bit growths only used below for integrator stages.
        bit_growths = iter(self.bit_growths())

        stages = []

        # When M=1, we can replace the inner CIC stage with an equivalent zero-order hold integrator.
        # When M=2, we can replace the inner CIC stage with a special upsampler that adds the last 2 samples.
        inner_zoh = list(self.rates) != [1] and self.M == 1
        m2_upsampler = list(self.rates) != [1] and self.M == 2

        # Comb stages.
        width = self.width_in
        for i in range(self.stages - int(inner_zoh + m2_upsampler)):
            next_width = self.width_in + next(bit_growths)
            stage = factor_reset(CombStage(self.M, width, width_out=next_width, num_channels=self.num_channels, always_ready=always_ready))
            m.submodules[f"comb{i}"] = stage
            stages += [ stage ]
            width = next_width
        
        # Upsampling.
        if list(self.rates) != [1]:
            if not m2_upsampler:
                if inner_zoh:
                    _ = next(bit_growths), next(bit_growths)  # drop comb and integrator growths
                stage = factor_reset(Upsampler(self.num_channels * width, max(self.rates), zero_order_hold=inner_zoh, variable=True, always_ready=always_ready))
            else:
                next_width = self.width_in + next(bit_growths)
                stage = factor_reset(UpsamplerM2(width, next_width, max(self.rates), variable=True, num_channels=self.num_channels, always_ready=always_ready))
                width = next_width
                _ = next(bit_growths)
            m.submodules["upsampler"] = stage
            m.d.sync += stage.factor.eq((1 << self.factor)-1)
            stages += [ stage ]

        # Integrator stages.
        for i in range(self.stages - int(inner_zoh + m2_upsampler)):
            width_out = self.width_in + next(bit_growths)
            stage = factor_reset(IntegratorStage(width, width_out, accum_width=width_out, num_channels=self.num_channels, always_ready=always_ready))
            m.submodules[f"integrator{i}"] = stage
            stages += [ stage ]
            width = width_out
    
        # Variable gain stage.
        shift_per_rate = { exact_log2(rate): (self.stages-1) * (exact_log2(max(self.rates)) - exact_log2(rate)) for rate in self.rates }
        stage = factor_reset(ProgrammableShift(width, width_out=self.width_out, num_channels=self.num_channels, shift_map=shift_per_rate, always_ready=always_ready))
        m.submodules["gain"] = stage
        if len(self.rates) > 1:
            m.d.sync += stage.factor.eq(self.factor)
        stages += [ stage ]
        width = self.width_out

        # Connect all stages to build the final filter.
        # For the upsampling CIC, we can only drop bits at the last stage.
        last = wiring.flipped(self.input)
        for stage in stages:
            connect(m, last, stage.input)
            last = stage.output
        connect(m, last, wiring.flipped(self.output))

        if self._domain != "sync":
            m = DomainRenamer(self._domain)(m)

        return m


class CICDecimator(wiring.Component):
    def __init__(self, M, stages, rates, width_in, width_out=None, num_channels=1, always_ready=False, domain="sync"):
        assert all((r & (r-1)) == 0 for r in rates), "rates must be powers of 2"
        self.M            = M
        self.stages       = stages
        self.rates        = rates
        self.width_in     = width_in
        self.num_channels = num_channels
        self._domain      = domain
        if width_out is None:
            width_out    = width_in + (stages * exact_log2(max(rates) * M))
        self.width_out    = width_out
        super().__init__({
            "input":  In(stream.Signature(
                data.ArrayLayout(signed(width_in), num_channels),
                always_ready=always_ready
            )),
            "output": Out(stream.Signature(
                data.ArrayLayout(signed(width_out), num_channels),
                always_ready=always_ready
            )),
            "factor": In(range(bits_for(max(rates)))),
        })

    def truncation_summary(self):
        rates = min(self.rates)
        return cic_truncation(N=self.stages, R=rates, M=self.M, 
                              Bin=self.width_in, Bout=self.width_out)

    def elaborate(self, platform):
        m = Module()

        stages = []

        always_ready = self.output.signature.always_ready

        full_width = self.width_in + (self.stages * exact_log2(max(self.rates) * self.M))
        stage_widths = ( full_width - n for n in self.truncation_summary() )

        # Sign extend stage
        last_width = self.width_in
        stage_width = next(stage_widths)
        stage = SignExtend(last_width, stage_width, num_channels=self.num_channels, always_ready=always_ready)
        m.submodules["signextend"] = stage
        stages += [ stage ]
        last_width = stage_width

        # Integrator stages
        for i in range(self.stages):
            stage_width = next(stage_widths)
            stage = IntegratorStage(last_width, stage_width, num_channels=self.num_channels, always_ready=always_ready)
            m.submodules[f"integrator{i}"] = stage
            stages += [ stage ]
            last_width = stage_width
        
        # Downsampling
        if list(self.rates) != [1]:
            stage = Downsampler(self.num_channels * last_width, max(self.rates), variable=True, always_ready=always_ready)
            m.submodules["downsampler"] = stage
            m.d.sync += stage.factor.eq((1 << self.factor)-1)
            stages += [ stage ]

        # Comb stages
        for i in range(self.stages):
            stage_width = next(stage_widths)
            stage = CombStage(self.M, last_width, stage_width, num_channels=self.num_channels, always_ready=always_ready)
            m.submodules[f"comb{i}"] = stage
            stages += [ stage ]
            last_width = stage_width

        # Gain stage

        # Ensure filter gain is at least the gain from width conversion.
        min_growth = self.stages * exact_log2(min(self.rates) * self.M)
        if min_growth < self.width_out - self.width_in:
            growth = self.width_out - self.width_in - min_growth
            stage = WidthConverter(last_width, last_width+growth, num_channels=self.num_channels, always_ready=always_ready)
            m.submodules["gain0"] = stage
            stages += [ stage ]
            last_width = last_width + growth

        shift_per_rate = { exact_log2(rate): self.stages * (exact_log2(max(self.rates)) - exact_log2(rate)) for rate in self.rates }
        # clip=False: we assume that rounding-induced overflow is not possible in decimator. Provide a test for that.
        stage = ProgrammableShift(last_width, width_out=self.width_out, num_channels=self.num_channels, shift_map=shift_per_rate, clip=False, always_ready=always_ready)
        m.submodules["gain"] = stage
        if len(self.rates) > 1:
            m.d.sync += stage.factor.eq(self.factor)
        stages += [stage]
        last_width = self.width_out

        # Connect stages, rounding/truncating where needed
        last = wiring.flipped(self.input)
        for stage in stages:
            connect(m, last, stage.input)
            last = stage.output
        connect(m, last, wiring.flipped(self.output))

        if self._domain != "sync":
            m = DomainRenamer(self._domain)(m)

        return m


class ProgrammableShift(wiring.Component):
    def __init__(self, width_in, shift_map, clip=True, width_out=None, num_channels=1, always_ready=False):
        self.num_channels = num_channels
        self.width_in = width_in
        self.width_out = width_out or width_in
        self.shift_map = shift_map
        self.clip = clip
        if len(self.shift_map) == 1:
            self.factor = Const(list(self.shift_map.keys())[0])
        super().__init__({
            "input":  In(stream.Signature(
                data.ArrayLayout(signed(self.width_in), num_channels),
                always_ready=always_ready
            )),
            "output": Out(stream.Signature(
                data.ArrayLayout(signed(self.width_out), num_channels),
                always_ready=always_ready
            )),
        } | ({"factor":  In(range(max(shift_map.keys())+1))} if len(shift_map)>1 else {}))

    def elaborate(self, platform):
        m = Module()

        # The input width is already prepared to fit the maximum gain. Other rates 
        # might have a smaller gain, which is compensated with the proper shift.
        value_scaled = Signal.like(self.input.p)
        with m.Switch(self.factor):
            for k, sh in self.shift_map.items():
                with m.Case(k):
                    for c in range(self.num_channels):
                        m.d.comb += value_scaled[c].eq(self.input.p[c] << sh)

        with m.If(~self.output.valid | self.output.ready):
            if not self.input.signature.always_ready:
                m.d.comb += self.input.ready.eq(1)
            m.d.sync += self.output.valid.eq(self.input.valid)
            with m.If(self.input.valid):
                for c in range(self.num_channels):
                    shift = self.width_in - self.width_out
                    if shift > 0:
                        # Convergent rounding / round to even.
                        m.d.sync += self.output.payload[c].eq(convergent_round(value_scaled[c], shift, clip=self.clip))
                        # Truncation. 
                        #m.d.sync += self.output.payload[c].eq(value_scaled[c][shift:])
                    else:
                        m.d.sync += self.output.payload[c][-shift:].eq(value_scaled[c])
        return m


class SignExtend(wiring.Component):
    def __init__(self, width_in, width_out, num_channels=1, always_ready=False):
        self.num_channels = num_channels
        self.always_ready = always_ready
        super().__init__({
            "input":  In(stream.Signature(
                data.ArrayLayout(signed(width_in), num_channels),
                always_ready=always_ready
            )),
            "output": Out(stream.Signature(
                data.ArrayLayout(signed(width_out), num_channels),
                always_ready=always_ready
            )),
        })

    def elaborate(self, platform):
        m = Module()
        for c in range(self.num_channels):
            m.d.comb += self.output.p[c].eq(self.input.p[c])
        m.d.comb += self.output.valid.eq(self.input.valid)
        if not self.always_ready:
            m.d.comb += self.input.ready.eq(self.output.ready)
        return m


class WidthConverter(wiring.Component):
    def __init__(self, width_in, width_out, num_channels=1, always_ready=False):
        self.width_in = width_in
        self.width_out = width_out
        self.num_channels = num_channels
        self.always_ready = always_ready
        super().__init__({
            "input":  In(stream.Signature(
                data.ArrayLayout(signed(width_in), num_channels),
                always_ready=always_ready
            )),
            "output": Out(stream.Signature(
                data.ArrayLayout(signed(width_out), num_channels),
                always_ready=always_ready
            )),
        })

    def elaborate(self, platform):
        m = Module()

        shift = self.width_out - self.width_in

        for c in range(self.num_channels):
            m.d.comb += self.output.p[c][shift:].eq(self.input.p[c])
        m.d.comb += self.output.valid.eq(self.input.valid)
        if not self.always_ready:
            m.d.comb += self.input.ready.eq(self.output.ready)
        return m


class CombStage(wiring.Component):
    def __init__(self, M, width_in, width_out=None, num_channels=1, always_ready=False):
        assert M in (1,2)
        self.M         = M
        self.width_in  = width_in
        self.width_out = width_out or width_in + 1
        self.num_channels = num_channels
        super().__init__({
            "input":  In(stream.Signature(
                data.ArrayLayout(signed(self.width_in), num_channels),
                always_ready=always_ready
            )),
            "output": Out(stream.Signature(
                data.ArrayLayout(signed(self.width_out), num_channels),
                always_ready=always_ready
            )),
        })
    
    def elaborate(self, platform):
        m = Module()

        shift = max(self.width_in - self.width_out, 0)
        delay = [ Signal.like(self.input.p) for _ in range(self.M) ]

        with m.If(~self.output.valid | self.output.ready):
            if not self.input.signature.always_ready:
                m.d.comb += self.input.ready.eq(1)
            m.d.sync += self.output.valid.eq(self.input.valid)
            with m.If(self.input.valid):
                m.d.sync += delay[0].eq(self.input.p)
                m.d.sync += [ delay[i].eq(delay[i-1]) for i in range(1, self.M) ]
                for c in range(self.num_channels):
                    diff = self.input.p[c] - delay[-1][c]
                    m.d.sync += self.output.p[c].eq(diff[shift:])

        return m


class IntegratorStage(wiring.Component):
    def __init__(self, width_in, width_out, accum_width=None, num_channels=1, always_ready=False):
        self.width_in = width_in
        self.width_out = width_out
        self.accum_width = accum_width or self.width_in
        self.num_channels = num_channels
        super().__init__({
            "input":  In(stream.Signature(
                data.ArrayLayout(signed(width_in), num_channels),
                always_ready=always_ready
            )),
            "output": Out(stream.Signature(
                data.ArrayLayout(signed(width_out), num_channels),
                always_ready=always_ready
            )),
        })

    def elaborate(self, platform):
        m = Module()

        shift = max(self.accum_width - self.width_out, 0)

        accumulator = Signal(data.ArrayLayout(signed(self.accum_width), self.num_channels))
        for c in range(self.num_channels):
            m.d.comb += self.output.payload[c].eq(accumulator[c][shift:].as_signed())

        with m.If(~self.output.valid | self.output.ready):
            if not self.input.signature.always_ready:
                m.d.comb += self.input.ready.eq(1)
            m.d.sync += self.output.valid.eq(self.input.valid)
            with m.If(self.input.valid):
                for c in range(self.num_channels):
                    m.d.sync += accumulator[c].eq(accumulator[c] + self.input.payload[c])

        return m


class Upsampler(wiring.Component):
    def __init__(self, width, factor, zero_order_hold=False, variable=False, always_ready=False):
        self.width = width
        self.zoh   = zero_order_hold
        signature = {
            "input":  In(stream.Signature(width, always_ready=always_ready)),
            "output": Out(stream.Signature(width, always_ready=always_ready)),
        }
        if variable:
            signature.update({"factor": In(range(factor))})
        else:
            self.factor = Const(factor)
        super().__init__(signature)

    def elaborate(self, platform):
        m = Module()

        counter = Signal.like(self.factor)

        with m.If(~self.output.valid | self.output.ready):
            with m.If(counter == 0):
                if not self.input.signature.always_ready:
                    m.d.comb += self.input.ready.eq(1)
                m.d.sync += self.output.payload.eq(self.input.payload)
                m.d.sync += self.output.valid.eq(self.input.valid)
                with m.If(self.input.valid):
                    m.d.sync += counter.eq(self.factor)
            with m.Else():
                if not self.zoh:
                    m.d.sync += self.output.payload.eq(0)
                m.d.sync += counter.eq(counter - 1)

        return m


class UpsamplerM2(wiring.Component):
    def __init__(self, width_in, width_out, factor, variable=False, num_channels=1, always_ready=False):
        self.width_in = width_in
        self.width_out = width_out
        self.num_channels = num_channels
        signature = {
            "input":  In(stream.Signature(data.ArrayLayout(signed(width_in), num_channels), always_ready=always_ready)),
            "output": Out(stream.Signature(data.ArrayLayout(signed(width_out), num_channels), always_ready=always_ready)),
        }
        if variable:
            signature.update({"factor": In(range(factor))})
        else:
            self.factor = Const(factor)
        super().__init__(signature)

    def elaborate(self, platform):
        m = Module()

        counter = Signal.like(self.factor)
        
        last_payload = Signal.like(self.input.p)

        with m.If(~self.output.valid | self.output.ready):
            with m.If(counter == 0):
                if not self.input.signature.always_ready:
                    m.d.comb += self.input.ready.eq(1)
                for c in range(self.num_channels):
                    m.d.sync += self.output.p[c].eq(self.input.p[c] + last_payload[c])
                m.d.sync += self.output.valid.eq(self.input.valid)
                with m.If(self.input.valid):
                    m.d.sync += last_payload.eq(self.input.payload)
                    m.d.sync += counter.eq(self.factor)
            with m.Else():
                m.d.sync += counter.eq(counter - 1)

        return m


class Downsampler(wiring.Component):
    def __init__(self, width, factor, variable=False, always_ready=False):
        signature = {
            "input":  In(stream.Signature(width, always_ready=always_ready)),
            "output": Out(stream.Signature(width, always_ready=always_ready)),
        }
        if variable:
            signature.update({"factor": In(range(factor))})
        else:
            self.factor = Const(factor)
        super().__init__(signature)

    def elaborate(self, platform):
        m = Module()

        counter = Signal.like(self.factor)

        with m.If(self.input.ready & self.input.valid):
            with m.If(counter == 0):
                m.d.sync += counter.eq(self.factor)
            with m.Else():
                m.d.sync += counter.eq(counter - 1)

        m.d.comb += [
            self.output.payload .eq(self.input.payload),
            self.output.valid   .eq(self.input.valid & (counter == 0)),
        ]
        if not self.input.signature.always_ready:
            m.d.comb += self.input.ready.eq(self.output.ready)

        return m


# Refs:
# [1]: Eugene Hogenauer, "An Economical Class of Digital Filters For Decimation and Interpolation,"
#      IEEE Trans. Acoust. Speech and Signal Proc., Vol. ASSP-29, April 1981, pp. 155-162.
# [2]: Rick Lyons, "Computing CIC filter register pruning using MATLAB"
#      https://www.dsprelated.com/showcode/269.php
# [3]: Peter Thorwartl, "Implementation of Filters", Part 3, lecture notes.
#      https://www.so-logic.net/documents/trainings/03_so_implementation_of_filters.pdf


# CIC downsamplers / decimators
# How much can we prune / truncate every stage output given a desired output width ?
# Calculate how many bits we can discard after each intermediate stage such that the quantization 
# error introduced is not greater than the one introduced by truncating/rounding at the filter 
# output.

def F_sq(N, R, M, i):
    assert i <= 2*N + 1
    if i == 2*N + 1: return 1  # eq. (16b) from [1]
    # h(k) and L (range of k), eq. (9b) from [1]
    if i <= N:
        # integrator stage
        L = N * (R * M - 1) + i - 1
        def h(k):
            return sum((-1)**l * comb(N, l) * comb(N-i+k-R*M*l, k-R*M*l)
                        for l in range(k//(R*M)+1))
    else:
        # comb stage
        L = 2*N + 1 - i
        def h(k):
            return (-1)**k * comb(2*N+1-i, k)
    # Compute standard deviation error gain from stage i to output
    F_i_sq = sum(h(k)**2 for k in range(L+1))
    return F_i_sq

def cic_truncation(N, R, M, Bin, Bout=None):
    full_width = Bin + ceil(N * log2(R * M))  # maximum width at output
    Bout = Bout or full_width                 # allow to specify full width
    B_last = full_width - Bout                # number of bits discarded at last stage
    t = log2(2**(2*B_last)/12) + log2(6 / N)  # Last two terms of (21) from [1]
    truncation = []
    for stage in range(2*N):
        ou = F_sq(N, R, M, stage+1)
        B_i = floor(0.5 * (-log2(ou) + t))    # Eq. (21) from [1]
        truncation.append(max(0, B_i))
    truncation.append(max(0, B_last))
    truncation[0] = 0  # [2]: fix case where input is truncated prior to any filtering
    return truncation

# CIC upsamplers / interpolators
# How much bit growth there is per intermediate stage?
# In the interpolator case, we cannot discard bits in intermediate stages: small errors in the 
# interpolator stages causes the variance of the error to grow without bound resulting in an 
# unstable filter.

def cic_growth(N, R, M):
    growths = []
    for i in range(1, 2*N+1):
        if i <= N:
            G_i = 2**i                             # comb stage
            # special case from [1] when differential delay is 1
            if M == 1 and i == N:
                G_i = 2**(N - 1)
        else:
            G_i = (2**(2*N-i) * (R*M)**(i-N)) / R  # integration stage
        growths.append(ceil(log2(G_i)))
    return growths
