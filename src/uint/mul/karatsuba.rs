//! Karatsuba multiplication
//!
//! This is a method which reduces the complexity of multiplication from O(n^2) to O(n^1.585).
//! For smaller numbers, it is best to stick to schoolbook multiplication, taking advantage
//! of better cache locality and avoiding recursion.
//!
//! In general, we consider the multiplication of two numbers of an equal size, `n` bits.
//! Setting b = 2^(n/2), then we can decompose the values:
//!   x•y = (x0 + x1•b)(y0 + y1•b)
//!
//! This equation is equivalent to a linear combination of three products of size `n/2`, which
//! may each be reduced by applying the same optimization.
//! Setting z0 = x0•y0, z1 = (x0 + x1)(y1 + y0), z2 = x1•y1:
//!   x•y = z0 + (z1 - z0 - z2)•b + z2•b^2
//!
//! Considering each sub-product as a tuple of integers `(lo, hi)`, the product is calculated as
//! follows (with appropriate carries):
//!   [z0.0, z0.1 + z1.0 - z0.0 - z2.0, z1.1 - z0.1 + z2.0 - z2.1, z2.1]
//!
//! Squaring uses a similar optimization, breaking the operation down into two half-size
//! squarings and a half-size multiplication:
//!
//!   x^2 = (x0 + x1•b)^2 = x0^2 + 2x0•x1•b + (x1•b)^2

use super::schoolbook;
use crate::{Limb, Uint, UintRef};

pub const MIN_STARTING_LIMBS: usize = 16;

#[inline]
const fn widening_mul_reduce<const LHS: usize, const RHS: usize, const HALF: usize>(
    lhs: &UintRef,
    rhs: &UintRef,
) -> (Uint<LHS>, Uint<RHS>) {
    assert!(LHS <= RHS && LHS == HALF * 2);
    let (x0, x1) = lhs.split_at(HALF);
    let (y0, y1) = rhs.split_at(HALF);

    // Calculate z0 = x0•y0
    let z0 = widening_mul_fixed(x0, y0);
    // Calculate z2 = x1•y1
    let z2 = widening_mul_fixed(x1, y1);

    // Calculate z1 = (x0 + x1)(y0 + y1)
    let (mut l0, mut l1) = (Uint::<HALF>::ZERO, Uint::<HALF>::ZERO);
    let (mut l0c, mut l1c) = (Limb::ZERO, Limb::ZERO);
    let mut i = 0;
    while i < HALF {
        (l0.limbs[i], l0c) = x0.0[i].carrying_add(x1.0[i], l0c);
        (l1.limbs[i], l1c) = y0.0[i].carrying_add(y1.0[i], l1c);
        i += 1;
    }
    let z1 = widening_mul_fixed(l0.as_uint_ref(), l1.as_uint_ref());

    // Middle terms of the result
    let (mut s0, mut s1) = (z0.1, z2.0);
    let (mut c, mut carry);

    // Add z1•b
    (s0, c) = s0.carrying_add(&z1.0, Limb::ZERO);
    (s1, c) = s1.carrying_add(&z1.1, c);
    carry = c;
    // Correct for overflowing terms in z1 by adding (l0c•l1 + l1c•l0)•b^2
    (s1, c) = s1.carrying_add(
        &Uint::select(&Uint::ZERO, &l0, l1c.is_nonzero()),
        Limb::ZERO,
    );
    carry = carry.wrapping_add(c);
    (s1, c) = s1.carrying_add(
        &Uint::select(&Uint::ZERO, &l1, l0c.is_nonzero()),
        Limb::ZERO,
    );
    carry = carry.wrapping_add(c);
    carry = carry.wrapping_add(l0c.bitand(l1c));

    // Subtract (z0 + z2)•b
    (s0, c) = s0.borrowing_sub(&z0.0, Limb::ZERO);
    (s1, c) = s1.borrowing_sub(&z0.1, c);
    carry = carry.wrapping_add(c);
    (s0, c) = s0.borrowing_sub(&z2.0, Limb::ZERO);
    (s1, c) = s1.borrowing_sub(&z2.1, c);
    carry = carry.wrapping_add(c);

    (
        concat(&z0.0, &s0),
        concat(&s1, &z2.1.overflowing_add_limb(carry).0),
    )
}

#[inline]
pub const fn widening_mul_fixed<const LHS: usize, const RHS: usize>(
    lhs: &UintRef,
    rhs: &UintRef,
) -> (Uint<LHS>, Uint<RHS>) {
    assert!(lhs.nlimbs() == LHS && rhs.nlimbs() == RHS);

    if LHS < MIN_STARTING_LIMBS || RHS < MIN_STARTING_LIMBS {
        let (mut lo, mut hi) = (Uint::ZERO, Uint::ZERO);
        schoolbook::mul_wide(
            lhs.as_slice(),
            rhs.as_slice(),
            lo.as_mut_limbs(),
            hi.as_mut_limbs(),
        );
        return (lo, hi);
    }

    if LHS <= RHS {
        let (y0, y1) = rhs.split_at(LHS);
        let (lo, mut hi) = match LHS {
            16 => widening_mul_reduce::<LHS, RHS, 8>(lhs, y0),
            32 => widening_mul_reduce::<LHS, RHS, 16>(lhs, y0),
            64 => widening_mul_reduce::<LHS, RHS, 32>(lhs, y0),
            128 => widening_mul_reduce::<LHS, RHS, 64>(lhs, y0),
            _ => {
                let mut lo_hi = [[Limb::ZERO; LHS]; 2];
                widening_mul(lhs, y0, UintRef::new_mut(lo_hi.as_flattened_mut()));
                (Uint::new(lo_hi[0]), Uint::new(lo_hi[1]).resize::<RHS>())
            }
        };
        if !y1.is_empty() {
            wrapping_mul_add(lhs, y1, hi.as_mut_uint_ref());
        }
        (lo, hi)
    } else {
        // LHS > RHS, swap arguments
        let (lo, hi) = widening_mul_fixed::<RHS, LHS>(rhs, lhs);
        // Need to repartition from (RHS, LHS) to (LHS, RHS)
        let mut lo = lo.resize::<LHS>();
        lo.as_mut_uint_ref()
            .trailing_mut(RHS)
            .copy_from(hi.as_uint_ref().leading(LHS - RHS));
        (
            lo,
            hi.wrapping_shr_by_limbs_vartime((LHS - RHS) as u32)
                .resize::<RHS>(),
        )
    }
}

#[inline]
pub const fn wrapping_mul_fixed<const LHS: usize, const RHS: usize>(
    lhs: &UintRef,
    rhs: &UintRef,
) -> Uint<LHS> {
    assert!(lhs.nlimbs() == LHS && rhs.nlimbs() == RHS);

    if LHS < MIN_STARTING_LIMBS || RHS < MIN_STARTING_LIMBS {
        let mut lo = Uint::ZERO;
        schoolbook::wrapping_mul(lhs.as_slice(), rhs.as_slice(), lo.as_mut_limbs());
        return lo;
    }

    #[inline]
    const fn reduce<const LHS: usize, const HALF: usize>(
        lhs: &UintRef,
        rhs: &UintRef,
    ) -> Uint<LHS> {
        assert!(LHS == HALF * 2);
        let (x0, x1) = lhs.split_at(HALF);
        let (y0, y1) = rhs.leading(LHS).split_at(HALF);

        // Calculate z0 = x0•y0
        let z0 = widening_mul_fixed::<HALF, HALF>(x0, y0);
        // Calculate z1 = x0•y1
        let z1 = wrapping_mul_fixed::<HALF, HALF>(x0, y1);
        // Calculate z2 = x1•y0
        let z2 = wrapping_mul_fixed::<HALF, HALF>(x1, y0);

        concat(&z0.0, &z0.1.wrapping_add(&z1).wrapping_add(&z2))
    }

    if LHS <= RHS {
        match LHS {
            16 => {
                return reduce::<LHS, 8>(lhs, rhs);
            }
            32 => {
                return reduce::<LHS, 16>(lhs, rhs);
            }
            64 => {
                return reduce::<LHS, 32>(lhs, rhs);
            }
            128 => {
                return reduce::<LHS, 64>(lhs, rhs);
            }
            _ => {}
        }
    }

    // LHS > RHS or less optimized size
    let mut lo = Uint::ZERO;
    wrapping_mul(lhs, rhs, lo.as_mut_uint_ref());
    lo
}

#[inline]
pub const fn widening_square_fixed<const LIMBS: usize>(
    uint: &UintRef,
) -> (Uint<LIMBS>, Uint<LIMBS>) {
    assert!(
        uint.nlimbs() == LIMBS,
        "invalid arguments to widening_square_fixed"
    );

    if LIMBS < MIN_STARTING_LIMBS {
        let (mut lo, mut hi) = (Uint::ZERO, Uint::ZERO);
        schoolbook::square_wide(uint.as_slice(), lo.as_mut_limbs(), hi.as_mut_limbs());
        return (lo, hi);
    }

    #[inline]
    const fn reduce<const LIMBS: usize, const HALF: usize>(
        uint: &UintRef,
    ) -> (Uint<LIMBS>, Uint<LIMBS>) {
        assert!(LIMBS == HALF * 2);
        let (x0, x1) = uint.split_at(HALF);

        // Calculate z0 = x0^2
        let z0 = widening_square_fixed::<HALF>(x0);
        // Calculate z1 = x0•x1
        let mut z1 = widening_mul_fixed::<HALF, HALF>(x0, x1);
        // Calculate z2 = x1^2
        let z2 = widening_square_fixed::<HALF>(x1);

        let (mut c, mut carry);
        // Double z1
        (z1.0, c) = z1.0.overflowing_shl1();
        (z1.1, carry) = z1.1.carrying_shl1(c);
        // Add z0.1, z2.0 to z1
        (z1.0, c) = z1.0.carrying_add(&z0.1, Limb::ZERO);
        (z1.1, c) = z1.1.carrying_add(&z2.0, c);
        carry = carry.wrapping_add(c);

        (
            concat(&z0.0, &z1.0),
            concat(&z1.1, &z2.1.overflowing_add_limb(carry).0),
        )
    }

    match LIMBS {
        16 => reduce::<LIMBS, 8>(uint),
        32 => reduce::<LIMBS, 16>(uint),
        64 => reduce::<LIMBS, 32>(uint),
        128 => reduce::<LIMBS, 64>(uint),
        _ => {
            let mut lo_hi = [[Limb::ZERO; LIMBS]; 2];
            widening_square(uint, UintRef::new_mut(lo_hi.as_flattened_mut()));
            (Uint::new(lo_hi[0]), Uint::new(lo_hi[1]))
        }
    }
}

#[inline]
pub const fn wrapping_square_fixed<const LIMBS: usize>(uint: &UintRef) -> Uint<LIMBS> {
    assert!(uint.nlimbs() == LIMBS);

    if LIMBS < MIN_STARTING_LIMBS {
        let mut lo = Uint::ZERO;
        schoolbook::wrapping_square(uint.as_slice(), lo.as_mut_limbs());
        return lo;
    }

    #[inline]
    const fn reduce<const LIMBS: usize, const HALF: usize>(uint: &UintRef) -> Uint<LIMBS> {
        assert!(LIMBS == HALF * 2);
        let (x0, x1) = uint.split_at(HALF);

        // Calculate z0 = x0^2
        let z0 = widening_square_fixed(x0);
        // Calculate z1 = x0•x13
        let z1 = wrapping_mul_fixed::<HALF, HALF>(x0, x1);

        concat(&z0.0, &z0.1.wrapping_add(&z1.overflowing_shl1().0))
    }

    match LIMBS {
        16 => reduce::<LIMBS, 8>(uint),
        32 => reduce::<LIMBS, 16>(uint),
        64 => reduce::<LIMBS, 32>(uint),
        128 => reduce::<LIMBS, 64>(uint),
        _ => {
            let mut lo = Uint::ZERO;
            wrapping_square(uint, lo.as_mut_uint_ref());
            lo
        }
    }
}

#[inline]
const fn reduce_mul(x: &UintRef, y: &UintRef, out: &mut UintRef, split: usize, add: bool) -> Limb {
    let (x0, x1) = x.split_at(split);
    let (y0, y1) = y.split_at(split);

    const fn reduce<const LIMBS: usize>(
        x: &UintRef,
        y: &UintRef,
        out: &mut UintRef,
        add: bool,
    ) -> Limb {
        if out.nlimbs() == LIMBS {
            let res = wrapping_mul_fixed::<LIMBS, LIMBS>(x, y);
            if add {
                out.carrying_add_assign(res.as_uint_ref(), Limb::ZERO)
            } else {
                out.copy_from(res.as_uint_ref());
                Limb::ZERO
            }
        } else {
            assert!(out.nlimbs() > LIMBS);
            let res = widening_mul_fixed::<LIMBS, LIMBS>(x, y);
            let (assign, tail) = out.split_at_mut(if out.nlimbs() > LIMBS * 2 {
                LIMBS * 2
            } else {
                out.nlimbs()
            });
            let (lo, hi) = assign.split_at_mut(LIMBS);
            if add {
                let carry = lo.carrying_add_assign(res.0.as_uint_ref(), Limb::ZERO);
                hi.carrying_add_assign(res.1.as_uint_ref().leading(hi.nlimbs()), carry);
                tail.add_assign_limb(carry)
            } else {
                lo.copy_from(res.0.as_uint_ref());
                hi.copy_from(res.1.as_uint_ref().leading(hi.nlimbs()));
                Limb::ZERO
            }
        }
    }

    let mut carry = match split {
        16 => reduce::<16>(x0, y0, out, add),
        32 => reduce::<32>(x0, y0, out, add),
        64 => reduce::<64>(x0, y0, out, add),
        _ => reduce::<128>(x0, y0, out, add),
    };

    // Handle trailing limbs
    if out.nlimbs() > split {
        if !x1.is_empty() {
            let c = wrapping_mul_add(x1, y, out.trailing_mut(split));
            carry = carry.wrapping_add(c);
        }
        if !y1.is_empty() {
            let tail_len = out.nlimbs() - split;
            let assign_len = if y.nlimbs() < tail_len {
                y.nlimbs()
            } else {
                tail_len
            };
            let (assign, tail) = out.trailing_mut(split).split_at_mut(assign_len);
            let c = wrapping_mul_add(y1, x0, assign);
            let c = tail.add_assign_limb(c);
            carry = carry.wrapping_add(c);
        }
    }

    carry
}

/// Multiply two limb slices, placing the result in `out`.
///
/// `lhs` and `rhs` may have different lengths.
/// `out` is assumed to be zeroed.
pub const fn widening_mul(lhs: &UintRef, rhs: &UintRef, out: &mut UintRef) {
    assert!(
        lhs.nlimbs() + rhs.nlimbs() == out.nlimbs(),
        "invalid arguments to widening_mul"
    );

    let overlap = if lhs.nlimbs() < rhs.nlimbs() {
        lhs.nlimbs()
    } else {
        rhs.nlimbs()
    };
    let split = previous_power_of_2(overlap);

    if split < MIN_STARTING_LIMBS {
        // let (lo, hi) = out.as_mut_slice().split_at_mut(lhs.nlimbs());
        // schoolbook::mul_wide(lhs.as_slice(), rhs.as_slice(), lo, hi);
        schoolbook::wrapping_mul(lhs.as_slice(), rhs.as_slice(), out.as_mut_slice());
        return;
    }

    reduce_mul(lhs, rhs, out, split, false);
}

/// Multiply two limb slices, placing the potentially-truncated result in `out`.
///
/// `lhs` and `rhs` may have different lengths.
/// `out` is assumed to be zeroed.
pub const fn wrapping_mul(lhs: &UintRef, rhs: &UintRef, out: &mut UintRef) {
    assert!(
        lhs.nlimbs() + rhs.nlimbs() >= out.nlimbs(),
        "invalid arguments to wrapping_mul"
    );

    let overlap = if lhs.nlimbs() < rhs.nlimbs() {
        lhs.nlimbs()
    } else {
        rhs.nlimbs()
    };
    let overlap = if out.nlimbs() < overlap {
        out.nlimbs()
    } else {
        overlap
    };
    let split = previous_power_of_2(overlap);

    if split < MIN_STARTING_LIMBS {
        schoolbook::wrapping_mul(lhs.as_slice(), rhs.as_slice(), out.as_mut_slice());
        return;
    }

    reduce_mul(lhs, rhs, out, split, false);
}

/// Multiply two limb slices, adding the result to `out`.
///
/// `lhs` and `rhs` may have different lengths.
#[inline(never)]
#[track_caller]
pub const fn wrapping_mul_add(lhs: &UintRef, rhs: &UintRef, out: &mut UintRef) -> Limb {
    assert!(
        lhs.nlimbs() + rhs.nlimbs() >= out.nlimbs(),
        "invalid arguments to wrapping_mul_add"
    );

    let overlap = if lhs.nlimbs() < rhs.nlimbs() {
        lhs.nlimbs()
    } else {
        rhs.nlimbs()
    };
    let overlap = if out.nlimbs() < overlap {
        out.nlimbs()
    } else {
        overlap
    };
    let split = previous_power_of_2(overlap);

    if split < MIN_STARTING_LIMBS {
        return schoolbook::wrapping_mul_add(lhs.as_slice(), rhs.as_slice(), out.as_mut_slice());
    }

    reduce_mul(lhs, rhs, out, split, true)
}

pub(crate) const fn widening_square(limbs: &UintRef, out: &mut UintRef) {
    assert!(
        limbs.nlimbs() * 2 == out.nlimbs(),
        "invalid arguments to widening_square"
    );

    if limbs.nlimbs() < MIN_STARTING_LIMBS {
        let (lo, hi) = out.split_at_mut(limbs.nlimbs());
        schoolbook::square_wide(limbs.as_slice(), lo.as_mut_slice(), hi.as_mut_slice());
        return;
    }

    wrapping_square(limbs, out);
}

pub(crate) const fn wrapping_square(limbs: &UintRef, out: &mut UintRef) {
    assert!(
        limbs.nlimbs() >= out.nlimbs() / 2,
        "invalid arguments to wrapping_square"
    );

    let split = previous_power_of_2(limbs.nlimbs());
    if split < MIN_STARTING_LIMBS {
        schoolbook::wrapping_square(limbs.as_slice(), out.as_mut_slice());
        return;
    }

    let (x0, x1) = limbs.split_at(split);
    // (x0 + x1b)^2 = x0^2 + x0x1b + (x0 + x1b)x1b

    const fn reduce<const LIMBS: usize>(x: &UintRef, out: &mut UintRef) {
        if out.nlimbs() == LIMBS {
            let res = wrapping_square_fixed::<LIMBS>(x);
            out.copy_from(res.as_uint_ref());
        } else {
            assert!(out.nlimbs() > LIMBS && out.nlimbs() <= LIMBS * 2);
            let res = widening_square_fixed::<LIMBS>(x);
            let (lo, hi) = out.split_at_mut(LIMBS);
            lo.copy_from(res.0.as_uint_ref());
            hi.copy_from(res.1.as_uint_ref().leading(hi.nlimbs()));
        }
    }

    match split {
        16 => reduce::<16>(x0, out),
        32 => reduce::<32>(x0, out),
        64 => reduce::<64>(x0, out),
        _ => reduce::<128>(x0, out),
    }

    if out.nlimbs() > split {
        let out_len = if limbs.nlimbs() < out.nlimbs() {
            limbs.nlimbs()
        } else {
            out.nlimbs()
        };
        let assign = out.trailing_mut(split);
        // Add x•x1b
        wrapping_mul_add(limbs, x1, assign);
        // Add x0•x1b
        let (assign, tail) = assign.split_at_mut(out_len - split);
        let carry = wrapping_mul_add(x0, x1, assign);
        tail.add_assign_limb(carry);
    }
}

#[inline]
const fn previous_power_of_2(val: usize) -> usize {
    1usize << val.ilog2()
}

#[inline]
const fn concat<const LIMBS: usize, const HALF: usize>(
    lo: &Uint<HALF>,
    hi: &Uint<HALF>,
) -> Uint<LIMBS> {
    assert!(LIMBS >= HALF * 2);
    let mut res = Uint::<LIMBS>::ZERO;
    let (lo_mut, hi_mut) = res
        .as_mut_uint_ref()
        .leading_mut(HALF * 2)
        .split_at_mut(HALF);
    lo_mut.copy_from_slice(lo.as_limbs());
    hi_mut.copy_from_slice(hi.as_limbs());
    res
}
