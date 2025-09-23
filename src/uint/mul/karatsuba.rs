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

pub const fn widening_mul_fixed<const LHS: usize, const RHS: usize>(
    lhs: &UintRef,
    rhs: &UintRef,
) -> (Uint<LHS>, Uint<RHS>) {
    debug_assert!(lhs.nlimbs() == LHS && rhs.nlimbs() == RHS);

    #[inline]
    const fn reduce<const LHS: usize, const RHS: usize, const HALF: usize>(
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

    if LHS < MIN_STARTING_LIMBS || RHS < MIN_STARTING_LIMBS {
        let (mut lo, mut hi) = (Uint::ZERO, Uint::ZERO);
        schoolbook::mul_wide(
            lhs.as_slice(),
            rhs.as_slice(),
            lo.as_mut_limbs(),
            hi.as_mut_limbs(),
        );
        (lo, hi)
    } else if LHS == RHS {
        match LHS {
            16 => reduce::<LHS, RHS, 8>(lhs, rhs),
            32 => reduce::<LHS, RHS, 16>(lhs, rhs),
            64 => reduce::<LHS, RHS, 32>(lhs, rhs),
            128 => reduce::<LHS, RHS, 64>(lhs, rhs),
            _ => {
                let mut lo_hi = [[Limb::ZERO; LHS]; 2];
                wrapping_mul(lhs, rhs, UintRef::new_flattened_mut(&mut lo_hi), false);
                (Uint::new(lo_hi[0]), Uint::new(lo_hi[1]).resize::<RHS>())
            }
        }
    } else if LHS < RHS {
        let (y0, y1) = rhs.split_at(LHS);
        let (lo, mut hi) = cast(widening_mul_fixed::<LHS, LHS>(lhs, y0));
        wrapping_mul(lhs, y1, hi.as_mut_uint_ref(), true);
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
pub const fn wrapping_mul_fixed<const LHS: usize>(
    lhs: &UintRef,
    rhs: &UintRef,
) -> (Uint<LHS>, Limb) {
    debug_assert!(lhs.nlimbs() == LHS);

    #[inline]
    const fn reduce<const LHS: usize, const HALF: usize>(
        lhs: &UintRef,
        rhs: &UintRef,
    ) -> (Uint<LHS>, Limb) {
        assert!(LHS == HALF * 2);
        let (x0, x1) = lhs.split_at(HALF);
        let (y0, y1) = rhs.leading(LHS).split_at(HALF);

        // Calculate z0 = x0•y0
        let z0 = widening_mul_fixed::<HALF, HALF>(x0, y0);
        // Calculate z1 = x0•y1
        let (z1, z1c) = wrapping_mul_fixed::<HALF>(x0, y1);
        // Calculate z2 = x1•y0
        let (z2, z2c) = wrapping_mul_fixed::<HALF>(x1, y0);

        let (hi, c1) = z0.1.carrying_add(&z1, Limb::ZERO);
        let (hi, c2) = hi.carrying_add(&z2, Limb::ZERO);
        let carry = z1c.wrapping_add(z2c).wrapping_add(c1).wrapping_add(c2);

        (concat(&z0.0, &hi), carry)
    }

    if LHS < MIN_STARTING_LIMBS || rhs.nlimbs() < MIN_STARTING_LIMBS {
        let mut lo = Uint::ZERO;
        let carry = schoolbook::wrapping_mul_add(lhs.as_slice(), rhs.as_slice(), lo.as_mut_limbs());
        return (lo, carry);
    } else if LHS <= rhs.nlimbs() {
        match LHS {
            16 => return reduce::<LHS, 8>(lhs, rhs),
            32 => return reduce::<LHS, 16>(lhs, rhs),
            64 => return reduce::<LHS, 32>(lhs, rhs),
            128 => return reduce::<LHS, 64>(lhs, rhs),
            _ => {}
        }
    }

    // LHS > RHS or less optimized size
    let mut lo = Uint::ZERO;
    let carry = wrapping_mul(lhs, rhs, lo.as_mut_uint_ref(), false);
    (lo, carry)
}

pub const fn widening_square_fixed<const LIMBS: usize>(
    uint: &UintRef,
) -> (Uint<LIMBS>, Uint<LIMBS>) {
    assert!(
        uint.nlimbs() == LIMBS,
        "invalid arguments to widening_square_fixed"
    );

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

    if LIMBS < MIN_STARTING_LIMBS {
        let (mut lo, mut hi) = (Uint::ZERO, Uint::ZERO);
        schoolbook::square_wide(uint.as_slice(), lo.as_mut_limbs(), hi.as_mut_limbs());
        return (lo, hi);
    }

    match LIMBS {
        16 => reduce::<LIMBS, 8>(uint),
        32 => reduce::<LIMBS, 16>(uint),
        64 => reduce::<LIMBS, 32>(uint),
        128 => reduce::<LIMBS, 64>(uint),
        _ => {
            let mut lo_hi = [[Limb::ZERO; LIMBS]; 2];
            wrapping_square(uint, UintRef::new_flattened_mut(&mut lo_hi));
            (Uint::new(lo_hi[0]), Uint::new(lo_hi[1]))
        }
    }
}

#[inline]
pub const fn wrapping_square_fixed<const LIMBS: usize>(uint: &UintRef) -> Uint<LIMBS> {
    let mut lo = Uint::ZERO;
    wrapping_square(uint, lo.as_mut_uint_ref());
    lo
}

/// Multiply two limb slices, placing the potentially-truncated result in `out`.
///
/// `lhs` and `rhs` may have different lengths.
/// `out` is assumed to be zeroed.
#[inline]
pub const fn wrapping_mul(lhs: &UintRef, rhs: &UintRef, out: &mut UintRef, add: bool) -> Limb {
    assert!(
        lhs.nlimbs() + rhs.nlimbs() >= out.nlimbs(),
        "invalid arguments to wrapping_mul"
    );

    const fn reduce<const LIMBS: usize>(
        x: &UintRef,
        y: &UintRef,
        out: &mut UintRef,
        add: bool,
    ) -> Limb {
        let out_len = out.nlimbs();
        if out_len <= x.nlimbs() {
            let (x0, x1) = x.leading(out_len).split_at(out_len - LIMBS);
            let y0 = y.leading(LIMBS);

            let (res, mut carry) = wrapping_mul_fixed::<LIMBS>(x1, y0);
            let assign = out.trailing_mut(out_len - LIMBS);
            if add {
                let c = assign.carrying_add_assign(res.as_uint_ref(), Limb::ZERO);
                carry = carry.wrapping_add(c);
            } else {
                assign.copy_from(res.as_uint_ref());
            }
            // Handle trailing limbs
            if !x0.is_empty() {
                let c = wrapping_mul(x0, y, out, true);
                carry = carry.wrapping_add(c)
            }
            carry
        } else {
            let (x0, x1) = x.split_at(LIMBS);
            let y_len = if y.nlimbs() < out_len {
                y.nlimbs()
            } else {
                out_len
            };
            let (y0, y1) = y.leading(y_len).split_at(LIMBS);
            let res = widening_mul_fixed::<LIMBS, LIMBS>(x0, y0);
            let (assign, tail) = out.split_at_mut(if out.nlimbs() < LIMBS * 2 {
                out.nlimbs()
            } else {
                LIMBS * 2
            });
            let (lo, hi) = assign.split_at_mut(LIMBS);
            let mut carry = if add {
                let mut carry = lo.carrying_add_assign(res.0.as_uint_ref(), Limb::ZERO);
                carry = hi.carrying_add_assign(res.1.as_uint_ref().leading(hi.nlimbs()), carry);
                tail.add_assign_limb(carry)
            } else {
                lo.copy_from(res.0.as_uint_ref());
                hi.copy_from(res.1.as_uint_ref().leading(hi.nlimbs()));
                Limb::ZERO
            };
            // Handle trailing limbs
            if !x1.is_empty() {
                let c = wrapping_mul(x1, y, out.trailing_mut(LIMBS), true);
                carry = carry.wrapping_add(c);
            }
            if !y1.is_empty() {
                let tail_len = out_len - LIMBS;
                let assign_len = if y_len < tail_len { y_len } else { tail_len };
                let (assign, tail) = out.trailing_mut(LIMBS).split_at_mut(assign_len);
                let c = wrapping_mul(y1, x0, assign, true);
                let c = tail.add_assign_limb(c);
                carry = carry.wrapping_add(c);
            }
            carry
        }
    }

    let overlap = if lhs.nlimbs() < rhs.nlimbs() {
        lhs.nlimbs()
    } else {
        rhs.nlimbs()
    };
    let overlap = if overlap < out.nlimbs() {
        overlap
    } else {
        out.nlimbs()
    };
    let split = previous_power_of_2(overlap);

    if split < MIN_STARTING_LIMBS {
        return schoolbook::wrapping_mul_add(lhs.as_slice(), rhs.as_slice(), out.as_mut_slice());
    }

    match split {
        16 => reduce::<16>(lhs, rhs, out, add),
        32 => reduce::<32>(lhs, rhs, out, add),
        64 => reduce::<64>(lhs, rhs, out, add),
        _ => reduce::<128>(lhs, rhs, out, add),
    }
}

#[inline]
pub(crate) const fn wrapping_square(uint: &UintRef, out: &mut UintRef) {
    assert!(
        out.nlimbs() <= uint.nlimbs() * 2,
        "invalid arguments to wrapping_square"
    );

    const fn reduce<const LIMBS: usize>(x: &UintRef, out: &mut UintRef) {
        let (x0, x1) = x.split_at(LIMBS);
        let (lo, hi) = out.split_at_mut(LIMBS);

        // Add z0 = x0^2
        let z0 = widening_square_fixed::<LIMBS>(x0);
        lo.copy_from(z0.0.as_uint_ref());

        // Add z1 = 2x0•x1•b
        if hi.nlimbs() <= LIMBS {
            let (z1, _carry) = wrapping_mul_fixed::<LIMBS>(x0, x1);
            let z1 = z1.overflowing_shl1().0;
            hi.copy_from(z0.1.wrapping_add(&z1).as_uint_ref().leading(hi.nlimbs()));
        } else {
            let (z01, z2) = hi.split_at_mut(LIMBS);
            z01.copy_from(z0.1.as_uint_ref());
            wrapping_square(x1, z2);
            let mut dx0 = Uint::<LIMBS>::ZERO;
            dx0.as_mut_uint_ref().copy_from(x0);
            let (dx0, dx0_hi) = dx0.overflowing_shl1();
            let z2_len = if z2.nlimbs() < x1.nlimbs() {
                z2.nlimbs()
            } else {
                x1.nlimbs()
            };
            let mut carry = z2.leading_mut(z2_len).conditional_add_assign(
                x1.leading(z2_len),
                Limb::ZERO,
                dx0_hi.is_nonzero(),
            );
            let (z1, z1tail) = hi.split_at_mut(LIMBS + z2_len);
            let c = wrapping_mul(dx0.as_uint_ref(), x1, z1, true);
            carry = carry.wrapping_add(c);
            z1tail.add_assign_limb(carry);
        }
    }

    let x = if uint.nlimbs() >= out.nlimbs() {
        uint.leading(out.nlimbs())
    } else {
        uint
    };
    if x.nlimbs() <= MIN_STARTING_LIMBS {
        schoolbook::wrapping_square(x.as_slice(), out.as_mut_slice());
        return;
    }

    // calc split based on output given x.len * 2 >= out.len
    let mut split = previous_power_of_2(out.nlimbs());
    if split > x.nlimbs() || 2 * split >= out.nlimbs() + MIN_STARTING_LIMBS {
        split /= 2;
    }

    match split {
        16 => reduce::<16>(x, out),
        32 => reduce::<32>(x, out),
        64 => reduce::<64>(x, out),
        _ => reduce::<128>(x, out),
    }
}

#[inline]
const fn previous_power_of_2(val: usize) -> usize {
    if val == 0 { 0 } else { 1usize << val.ilog2() }
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

#[inline(always)]
const fn cast<const LIMBS: usize, const LHS: usize, const RHS: usize>(
    (lo, hi): (Uint<LIMBS>, Uint<LIMBS>),
) -> (Uint<LHS>, Uint<RHS>) {
    assert!(LHS == LIMBS && RHS >= LIMBS);
    (lo.resize(), hi.resize())
}

#[cfg(feature = "rand_core")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Limb, Uint};
    use rand_core::{RngCore, SeedableRng};

    #[test]
    fn wrapping_mul_sizes() {
        const SIZE: usize = 200;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        for n in 0..100 {
            use crate::Random;

            let a = Uint::<SIZE>::random(&mut rng);
            let b = Uint::<SIZE>::random(&mut rng);
            let size_a = rng.next_u32() as usize % SIZE;
            let size_b = rng.next_u32() as usize % SIZE;
            let a = a.as_uint_ref().leading(size_a);
            let b = b.as_uint_ref().leading(size_b);
            let mut wide = [Limb::ZERO; SIZE * 2];
            wrapping_mul(a, b, UintRef::new_mut(&mut wide[..size_a + size_b]), false);
            for size in 1..size_a + size_b {
                let mut check = [Limb::ZERO; SIZE * 2];
                let wrapped = UintRef::new_mut(&mut check[..size]);
                wrapping_mul(b, a, wrapped, false);
                assert_eq!(
                    wrapped,
                    UintRef::new(&wide[..size]),
                    "comparison failed n={n}, a={a}, b={b}"
                );
            }
        }
    }

    #[test]
    fn wrapping_square_sizes() {
        const SIZE: usize = 200;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        for n in 0..100 {
            use crate::Random;

            let a = Uint::<SIZE>::random(&mut rng);
            let size_a = rng.next_u32() as usize % SIZE;
            let a = a.as_uint_ref().leading(size_a);
            let mut wide = [Limb::ZERO; SIZE * 2];
            wrapping_mul(a, a, UintRef::new_mut(&mut wide[..size_a * 2]), false);

            for size in 1..=size_a * 2 {
                let mut check = [Limb::ZERO; SIZE * 2];
                let wrapped = UintRef::new_mut(&mut check[..size]);
                println!("n={n} x={size_a} out={size}");
                wrapping_square(a, wrapped);
                assert_eq!(
                    wrapped,
                    UintRef::new(&wide[..size]),
                    "comparison failed n={n}, a={a}"
                );
            }
        }
    }
}
