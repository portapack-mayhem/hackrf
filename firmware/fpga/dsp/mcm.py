#
# This file is part of HackRF.
#
# Copyright (c) 2025 Great Scott Gadgets <info@greatscottgadgets.com>
# SPDX-License-Identifier: BSD-3-Clause

from collections            import defaultdict

from amaranth               import Module, Signal, signed
from amaranth.lib           import wiring, stream, data
from amaranth.lib.wiring    import In, Out
from amaranth.utils         import ceil_log2


class ShiftAddMCM(wiring.Component):
    def __init__(self, width, terms, num_channels=1, always_ready=False):
        self.terms = terms
        self.width = width
        self.num_channels = num_channels
        super().__init__({
            "input":  In(stream.Signature(
                data.ArrayLayout(signed(width), num_channels), 
                always_ready=always_ready)),
            "output": Out(stream.Signature(
                data.ArrayLayout(
                    data.StructLayout({
                        f"{i}": signed(1 + ceil_log2(2**(self.width-1) * abs(term))) for i, term in enumerate(terms)
                    }), num_channels), always_ready=always_ready)),
        })

    @property
    def delay(self):
        return 1

    def elaborate(self, platform):
        m = Module()

        unique_terms = set()              # unique, odd terms.
        term_outputs = defaultdict(list)  # outputs and shifts associated with each unique term.
        term_digits  = {}                 # shifts and signs for CSD representation.

        # Get unique terms and associate them to the outputs.
        for i, term in enumerate(self.terms):
            if term == 0:
                continue
            term_odd, out_shift = make_odd(term)
            term_outputs[term_odd].append((i, out_shift))
            unique_terms.add(term_odd)
        
        # Extract lists of CSD digits and their shifts.
        for term in unique_terms:
            digits = tuple((shift, digit) for shift, digit in enumerate(to_csd(term)) if digit != 0)
            term_digits[term] = digits

        # Stream control.
        advance = Signal()
        with m.If(~self.output.valid | self.output.ready):
            if not self.input.signature.always_ready:
                m.d.comb += self.input.ready.eq(1)
            m.d.sync += self.output.valid.eq(self.input.valid)
            m.d.comb += advance.eq(self.input.valid)

        # Compute multiplies.
        for c in range(self.num_channels):
            n = self.input.p[c]

            def get_leaf_node(shift, digit):
                base = n if digit > 0 else -n
                return base if shift == 0 else (base << shift)

            for term in unique_terms:
                result = None

                nodes = [ get_leaf_node(*digit_key) for digit_key in term_digits[term] ]
                result = balanced_sum(nodes)

                if result is None:
                    result = 0

                shape = signed(1 + ceil_log2(2**(self.width-1) * abs(term)))
                result_q = Signal(shape, name=f"mul_{term}_{c}")
                with m.If(advance):
                    m.d.sync += result_q.eq(result)

                for index, shift in term_outputs[term]:
                    m.d.comb += self.output.p[c][f"{index}"][shift:].eq(result_q)

        return m


def balanced_sum(nodes):
    if len(nodes) == 0:
        return 0
    while len(nodes) > 1:
        nodes = [
            nodes[i] + nodes[i + 1] if i + 1 < len(nodes) else nodes[i]
            for i in range(0, len(nodes), 2)
        ]
    return nodes[0]


def make_odd(n):
    """Convert number to odd fundamental by right-shifting. Returns (odd_part, shift_amount)"""
    if n == 0:
        return 0, 0
    
    shift = 0
    while n % 2 == 0:
        n = n >> 1
        shift += 1
    
    return n, shift


def to_csd(n):
    """ Convert integer to Canonical Signed Digit representation (LSB first). """
    if n == 0:
        return [0]
    
    sign = n < 0
    n = abs(n)
    binary = [ int(b) for b in f"{n:b}" ][::-1]

    # Apply CSD conversion algorithm.
    binary_padded = binary + [0]
    carry = 0
    csd = []
    for i, bit in enumerate(binary_padded):
        nextbit = binary_padded[i+1] if i+1 < len(binary_padded) else 0
        d = bit ^ carry
        ys = nextbit & d  # sign bit
        yd = ~nextbit & d  # data bit
        csd.append(yd - ys)
        carry = (bit & nextbit) | ((bit|nextbit)&carry)
    if sign:
        csd = [-1*c for c in csd]

    # Remove trailing zeros.
    while len(csd) > 1 and csd[-1] == 0:
        csd.pop()

    # Regular binary representation is preferred if the number
    # of additions was not improved.
    if sum(binary) <= sum(abs(d) for d in csd) - sign:
        if sign:
            return [ -d for d in binary ]
        return binary

    return csd
