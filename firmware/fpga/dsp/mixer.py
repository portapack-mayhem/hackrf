#
# This file is part of HackRF.
#
# Copyright (c) 2025 Great Scott Gadgets <info@greatscottgadgets.com>
# SPDX-License-Identifier: BSD-3-Clause

from amaranth               import Module, DomainRenamer, Signal, ClockSignal, Mux, Const, Cat
from amaranth.lib           import wiring, stream, data
from amaranth.lib.wiring    import In, Out

from dsp.round              import convergent_round
from dsp.sb_mac16           import SB_MAC16

from util                   import IQSample


class ComplexMultiplier(wiring.Component):

    def __init__(self, a_shape, b_shape, c_shape, always_ready=False, domain="sync"):
        super().__init__({
            "a": In(stream.Signature(a_shape, always_ready=always_ready)),
            "b": In(stream.Signature(b_shape, always_ready=always_ready)),
            "c": Out(stream.Signature(c_shape, always_ready=always_ready)),
        })
        self.always_ready = always_ready
        self.domain = domain

    @staticmethod
    def xform_stage(m, in_stream, xform):

        # Create output stream.
        out_shape = xform(in_stream.p.i).shape()
        out_stream = stream.Signature(
            IQSample(out_shape.width),
            always_ready=in_stream.signature.always_ready
        ).create()

        # Assign transformed stream.
        with m.If(~out_stream.valid | out_stream.ready):
            if not in_stream.signature.always_ready:
                m.d.comb += in_stream.ready.eq(1)
            m.d.sync += out_stream.valid.eq(in_stream.valid)
            m.d.sync += out_stream.p.i.eq(xform(in_stream.p.i))
            m.d.sync += out_stream.p.q.eq(xform(in_stream.p.q))

        return out_stream

    def elaborate(self, platform):
        m = Module()

        A = self.a.p.i
        B = self.a.p.q
        C = self.b.p.i
        D = self.b.p.q

        o_shape = (A * C  - B * D).shape()  # shape of the output
        o_width = o_shape.width

        mix_output = stream.Signature(
            IQSample(o_shape.width),
            always_ready=self.always_ready,
        ).create()

        ready_in  = Signal()
        valid_in  = Signal()
        m.d.comb += ready_in.eq(self.a.ready & self.b.ready)
        m.d.comb += valid_in.eq(self.a.valid & self.b.valid)

        def pipe(signal, length):
            pipe = [ signal ] + [ Signal.like(signal, name=f"{signal.name}_q{i}") for i in range(length) ]
            with m.If(ready_in):  # clock enable
                m.d.sync += [ pipe[i+1].eq(pipe[i]) for i in range(length) ]
            return pipe

        dsp_delay  = 3
        valid_pipe = pipe(valid_in, dsp_delay)

        if not self.always_ready:
            m.d.comb += self.a.ready.eq(~mix_output.valid | mix_output.ready)
            m.d.comb += self.b.ready.eq(~mix_output.valid | mix_output.ready)

        common_sb_mac16_params = dict(
            C_REG=0,
            A_REG=1,
            B_REG=1,
            D_REG=0,
            TOP_8x8_MULT_REG=0,
            BOT_8x8_MULT_REG=0,
            PIPELINE_16x16_MULT_REG1=0,
            PIPELINE_16x16_MULT_REG2=1,
            MODE_8x8=0,
            A_SIGNED=1,
            B_SIGNED=1,
        )

        # I = A*C - B*D  (select sub)

        # A*C
        m.submodules.mult_ac = mult_ac = SB_MAC16(
            **common_sb_mac16_params,
            TOPOUTPUT_SELECT=3,  # multiplier register
            BOTOUTPUT_SELECT=3,  # multiplier register
        )

        # B*D (and accumulator inputs connected to A*C result)
        m.submodules.mult_bd = mult_bd = SB_MAC16(
            **common_sb_mac16_params,
            TOPOUTPUT_SELECT=1,  # accumulator register
            TOPADDSUB_LOWERINPUT=2,
            TOPADDSUB_UPPERINPUT=1,
            TOPADDSUB_CARRYSELECT=2,
            BOTOUTPUT_SELECT=1,  # accumulator register
            BOTADDSUB_LOWERINPUT=2,
            BOTADDSUB_UPPERINPUT=1,
            BOTADDSUB_CARRYSELECT=0,
        )

        # Q = B*C + A*D

        # B*C
        m.submodules.mult_bc = mult_bc = SB_MAC16(
            **common_sb_mac16_params,
            TOPOUTPUT_SELECT=3,  # multiplier register
            BOTOUTPUT_SELECT=3,  # multiplier register
        )

        # A*D (C,D inputs connected to B*C result)
        m.submodules.mult_ad = mult_ad = SB_MAC16(
            **common_sb_mac16_params,
            TOPOUTPUT_SELECT=1,  # accumulator register
            TOPADDSUB_LOWERINPUT=2,
            TOPADDSUB_UPPERINPUT=1,
            TOPADDSUB_CARRYSELECT=2,
            BOTOUTPUT_SELECT=1,  # accumulator register
            BOTADDSUB_LOWERINPUT=2,
            BOTADDSUB_UPPERINPUT=1,
            BOTADDSUB_CARRYSELECT=0,
        )


        m.d.comb += [
            # mult AC.
            mult_ac.CLK             .eq(ClockSignal(self.domain)),
            mult_ac.CE              .eq(ready_in),
            mult_ac.A.as_signed()   .eq(A),
            mult_ac.B.as_signed()   .eq(C),
            mult_ac.AHOLD           .eq(~valid_pipe[0]),  # 0: load
            mult_ac.BHOLD           .eq(~valid_pipe[0]),

            # mult BD.
            mult_bd.CLK             .eq(ClockSignal(self.domain)),
            mult_bd.CE              .eq(ready_in),
            mult_bd.A.as_signed()   .eq(B),
            mult_bd.B.as_signed()   .eq(D),
            Cat(mult_bd.D, mult_bd.C).eq(mult_ac.O),
            mult_bd.AHOLD           .eq(~valid_pipe[0]),  # 0: load
            mult_bd.BHOLD           .eq(~valid_pipe[0]),
            mult_bd.CHOLD           .eq(0),
            mult_bd.DHOLD           .eq(0),
            mult_bd.OHOLDTOP        .eq(~valid_pipe[2]),
            mult_bd.OHOLDBOT        .eq(~valid_pipe[2]),
            mult_bd.ADDSUBTOP       .eq(1),  # subtract
            mult_bd.ADDSUBBOT       .eq(1),  # subtract
            mult_bd.OLOADTOP        .eq(0),
            mult_bd.OLOADBOT        .eq(0),

            # mult BC.
            mult_bc.CLK             .eq(ClockSignal(self.domain)),
            mult_bc.CE              .eq(ready_in),
            mult_bc.A.as_signed()   .eq(B),
            mult_bc.B.as_signed()   .eq(C),
            mult_bc.AHOLD           .eq(~valid_pipe[0]),  # 0: load
            mult_bc.BHOLD           .eq(~valid_pipe[0]),

            # mult AD.
            mult_ad.CLK             .eq(ClockSignal(self.domain)),
            mult_ad.CE              .eq(ready_in),
            mult_ad.A.as_signed()   .eq(A),
            mult_ad.B.as_signed()   .eq(D),
            Cat(mult_ad.D, mult_ad.C).eq(mult_bc.O),
            mult_ad.AHOLD           .eq(~valid_pipe[0]),  # 0: load
            mult_ad.BHOLD           .eq(~valid_pipe[0]),
            mult_ad.CHOLD           .eq(0),
            mult_ad.DHOLD           .eq(0),
            mult_ad.OHOLDTOP        .eq(~valid_pipe[2]),
            mult_ad.OHOLDBOT        .eq(~valid_pipe[2]),
            mult_ad.ADDSUBTOP       .eq(0),  # add
            mult_ad.ADDSUBBOT       .eq(0),  # add
            mult_ad.OLOADTOP        .eq(0),
            mult_ad.OLOADBOT        .eq(0),

            # Outputs.
            mix_output.p.i          .eq(mult_bd.O),
            mix_output.p.q          .eq(mult_ad.O),
            mix_output.valid        .eq(valid_pipe[dsp_delay]),
        ]

        last_output = mix_output
        
        # Add round and saturation stages to comply with output shape.
        last_output = self.xform_stage(m, last_output, 
            lambda t: convergent_round(t, o_width - 1 - self.c.p.i.shape().width - 1, clip=False))

        sat_shape = self.c.p.i.shape()
        lo = Const(-(1 << (sat_shape.width-1)), sat_shape)
        hi = Const((1 << (sat_shape.width-1)) - 1, sat_shape)
        last_output = self.xform_stage(m, last_output, lambda t: Mux(
            t > hi, hi, Mux(t < lo, lo, t[:sat_shape.width].as_signed())
        ))

        wiring.connect(m, last_output, wiring.flipped(self.c))

        if self.domain != "sync":
            m = DomainRenamer(self.domain)(m)
        
        return m

