#
# This file is part of HackRF.
#
# Copyright (c) 2025 Great Scott Gadgets <info@greatscottgadgets.com>
# SPDX-License-Identifier: BSD-3-Clause

from amaranth import Mux

def convergent_round(value, discarded_bits, clip=True):
    shape = value.shape()

    if discarded_bits == 0:
        return value
    if discarded_bits > shape.width - shape.signed:
        raise ValueError(f'cannot discard {discarded_bits} bits from a value with {shape.width - shape.signed} non-sign bits')
    if discarded_bits < 0:
        raise ValueError(f'cannot discard {discarded_bits} bits, only positive amounts')
    
    retained = value[discarded_bits:]
    discarded = value[:discarded_bits]
    msb_discarded = discarded[-1]
    rest_discarded = discarded[:-1]
    lsb_retained = retained[0] if len(retained) else 0
    # Round up:
    # - If discarded > 0.5
    # - If discarded == 0.5 and retained is odd
    round_up = msb_discarded & (rest_discarded.any() | lsb_retained)

    if shape.signed:
        retained = retained.as_signed()
        
    if len(retained) - shape.signed == 0:
        if not shape.signed:
            return round_up if not clip else 0
        result = (retained + round_up)[:-1] if not clip else (retained & ~round_up)
        return result.as_signed()
    
    result = retained + round_up
    if clip:
        # Perform saturation on maximum value to avoid extension of bit length.
        if shape.signed:
            result = result[:-1].as_signed()
            overflow = ~retained[-1] & result[-1]
            result = Mux(overflow, retained, result)
        else:
            result = Mux(result[-1], retained, result[:-1])
    return result.as_signed() if shape.signed else result
