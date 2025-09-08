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

pub const MIN_STARTING_LIMBS: usize = 32;
pub const MAX_REDUCE_LIMBS: usize = 24;

#[inline(always)]
pub(crate) const fn uint_widening_mul<const LHS: usize, const RHS: usize>(
    lhs: &Uint<LHS>,
    rhs: &Uint<RHS>,
) -> (Uint<LHS>, Uint<RHS>) {
    if LHS < MIN_STARTING_LIMBS || RHS < MIN_STARTING_LIMBS {
        let (mut lo, mut hi) = (Uint::<LHS>::ZERO, Uint::<RHS>::ZERO);
        schoolbook::mul_wide(
            lhs.as_limbs(),
            rhs.as_limbs(),
            lo.as_mut_limbs(),
            hi.as_mut_limbs(),
        );
        return (lo, hi);
    }

    let size = {
        let overlap = if LHS < RHS { LHS } else { RHS };
        overlap - (overlap & 1)
    };
    let half = size / 2;

    let (x, x_tail) = lhs.as_limbs().split_at(size);
    let (y, y_tail) = rhs.as_limbs().split_at(size);
    let (x0, x1) = UintRef::new(x).split_at(half);
    let (y0, y1) = UintRef::new(y).split_at(half);

    let (mut lo, mut mid, mut hi) = (Uint::<LHS>::ZERO, Uint::<LHS>::ZERO, Uint::<RHS>::ZERO);
    let ((lo_mut, lo_tail), mid_mut, hi_mut) = (
        lo.as_mut_uint_ref().split_at_mut(size),
        mid.as_mut_uint_ref().leading_mut(size),
        hi.as_mut_uint_ref().leading_mut(size),
    );
    let mut scratch = (Uint::<LHS>::ZERO, Uint::<LHS>::ZERO);
    let scratch = (scratch.0.as_mut_uint_ref(), scratch.1.as_mut_uint_ref());

    // Calculate z0 = x0•y0 into lo
    widening_mul(x0.as_slice(), y0.as_slice(), lo_mut, scratch.0);

    // Calculate z2 = x1•y1 into hi
    widening_mul(x1.as_slice(), y1.as_slice(), hi_mut, scratch.0);

    // Calculate z1 into mid
    let mut carry_b3 = compute_z1_mul((x0, x1), (y0, y1), mid_mut, scratch);

    // Subtract z0+z2 from mid
    let mut c = mid_mut.borrowing_sub_assign(lo_mut, Limb::ZERO);
    carry_b3 = carry_b3.wrapping_add(c);
    c = mid_mut.borrowing_sub_assign(hi_mut, Limb::ZERO);
    carry_b3 = carry_b3.wrapping_add(c);

    combine_overlapping(lo_mut, mid_mut, hi_mut, half, carry_b3);

    if LHS > RHS || LHS & 1 == 1 {
        // Need to shift hi backward into lo

        // Handle trailing limbs
        // if !x_tail.is_empty() {
        //     carrying_add_mul_limbs(x_tail, rhs, out.trailing_mut(size).as_mut());
        // }
        // if !y_tail.is_empty() {
        //     let (assign, tail) = out.trailing_mut(size).split_at_mut(size + y_tail.len());
        //     let carry = carrying_add_mul_limbs(y_tail, x, assign.as_mut());
        //     tail.add_assign_limb(carry);
        // }
    }

    (lo, hi)
}

#[inline(always)]
pub(crate) const fn uint_wrapping_mul<const LHS: usize, const RHS: usize>(
    lhs: &Uint<LHS>,
    rhs: &Uint<RHS>,
) -> Uint<LHS> {
    if LHS < MIN_STARTING_LIMBS || RHS < MIN_STARTING_LIMBS {
        let mut lo = Uint::<LHS>::ZERO;
        schoolbook::wrapping_mul(lhs.as_limbs(), rhs.as_limbs(), lo.as_mut_limbs());
        return lo;
    }

    let mut out = Uint::ZERO;
    let mut scratch = Uint::<LHS>::ZERO;
    if LHS == RHS && LHS & 1 == 0 {
        wrapping_mul_even(
            lhs.as_limbs(),
            rhs.as_limbs(),
            out.as_mut_uint_ref(),
            scratch.as_mut_uint_ref(),
        );
    } else {
        wrapping_mul(
            lhs.as_limbs(),
            rhs.as_limbs(),
            out.as_mut_uint_ref(),
            scratch.as_mut_uint_ref(),
        );
    }
    out
}

#[inline(always)]
pub(crate) const fn uint_widening_square<const LIMBS: usize>(
    uint: &Uint<LIMBS>,
) -> (Uint<LIMBS>, Uint<LIMBS>) {
    if LIMBS < MIN_STARTING_LIMBS * 2 {
        let (mut lo, mut hi) = (Uint::<LIMBS>::ZERO, Uint::<LIMBS>::ZERO);
        schoolbook::square_wide(uint.as_limbs(), lo.as_mut_limbs(), hi.as_mut_limbs());
        return (lo, hi);
    }

    let size = LIMBS - (LIMBS & 1);
    let half = size / 2;

    let (x, x_tail) = uint.as_limbs().split_at(size);
    let (x0, x1) = UintRef::new(x).split_at(half);

    let (mut lo, mut mid, mut hi) = (
        Uint::<LIMBS>::ZERO,
        Uint::<LIMBS>::ZERO,
        Uint::<LIMBS>::ZERO,
    );
    let ((lo_mut, lo_tail), mid_mut, hi_mut) = (
        lo.as_mut_uint_ref().split_at_mut(size),
        mid.as_mut_uint_ref().leading_mut(size),
        hi.as_mut_uint_ref().leading_mut(size),
    );
    let mut scratch = Uint::<LIMBS>::ZERO;
    let scratch = scratch.as_mut_uint_ref();

    // Calculate z0 = x0^2 into lo
    widening_square(x0.as_slice(), lo_mut, scratch);

    // Calculate z2 = x1^2 into hi
    widening_square(x1.as_slice(), hi_mut, scratch);

    // Calculate z1 = x0•x1 into mid
    widening_mul(x0.as_slice(), x1.as_slice(), mid_mut, scratch);
    let carry_b3 = Limb::select(Limb::ZERO, Limb::ONE, mid_mut.shl1_assign());

    combine_overlapping(lo_mut, mid_mut, hi_mut, half, carry_b3);

    // if LHS > RHS || LHS & 1 == 1 {
    // Need to shift hi backward into lo

    // Handle trailing limbs
    // if !x_tail.is_empty() {
    //     carrying_add_mul_limbs(x_tail, rhs, out.trailing_mut(size).as_mut());
    // }
    // if !y_tail.is_empty() {
    //     let (assign, tail) = out.trailing_mut(size).split_at_mut(size + y_tail.len());
    //     let carry = carrying_add_mul_limbs(y_tail, x, assign.as_mut());
    //     tail.add_assign_limb(carry);
    // }
    // }

    (lo, hi)
}

#[inline(always)]
pub(crate) const fn uint_wrapping_square<const LIMBS: usize>(uint: &Uint<LIMBS>) -> Uint<LIMBS> {
    if LIMBS < MIN_STARTING_LIMBS * 4 {
        let mut lo = Uint::<LIMBS>::ZERO;
        schoolbook::wrapping_square(uint.as_limbs(), lo.as_mut_limbs());
        return lo;
    }

    let size = LIMBS - (LIMBS & 1);
    let half = size / 2;

    let (x, x_tail) = uint.as_limbs().split_at(size);
    let (x0, x1) = UintRef::new(x).split_at(half);

    let (mut lo, mut mid, mut hi) = (
        Uint::<LIMBS>::ZERO,
        Uint::<LIMBS>::ZERO,
        Uint::<LIMBS>::ZERO,
    );
    let ((lo_mut, lo_tail), mid_mut, hi_mut) = (
        lo.as_mut_uint_ref().split_at_mut(size),
        mid.as_mut_uint_ref().leading_mut(size),
        hi.as_mut_uint_ref().leading_mut(size),
    );
    let mut scratch = Uint::<LIMBS>::ZERO;
    let scratch = scratch.as_mut_uint_ref();

    // Calculate z0 = x0^2 into lo
    widening_square(x0.as_slice(), lo_mut, scratch);

    // Calculate z2 = x1^2 into hi
    widening_square(x1.as_slice(), hi_mut, scratch);

    // Calculate z1 = x0•x1 into mid
    widening_mul(x0.as_slice(), x1.as_slice(), mid_mut, scratch);
    let carry_b3 = Limb::select(Limb::ZERO, Limb::ONE, mid_mut.shl1_assign());

    combine_overlapping(lo_mut, mid_mut, hi_mut, half, carry_b3);

    // if LHS > RHS || LHS & 1 == 1 {
    // Need to shift hi backward into lo

    // Handle trailing limbs
    // if !x_tail.is_empty() {
    //     carrying_add_mul_limbs(x_tail, rhs, out.trailing_mut(size).as_mut());
    // }
    // if !y_tail.is_empty() {
    //     let (assign, tail) = out.trailing_mut(size).split_at_mut(size + y_tail.len());
    //     let carry = carrying_add_mul_limbs(y_tail, x, assign.as_mut());
    //     tail.add_assign_limb(carry);
    // }
    // }

    lo
}

#[inline(always)]
const fn combine_overlapping(
    lo: &mut UintRef,
    mid: &UintRef,
    hi: &mut UintRef,
    half_size: usize,
    mut carry_b3: Limb,
) {
    // Add mid.0•b into lo
    let mut c = lo
        .trailing_mut(half_size)
        .carrying_add_assign(mid.leading(half_size), Limb::ZERO);

    // Add mid.1•b^2 into hi
    c = hi
        .leading_mut(half_size)
        .carrying_add_assign(mid.trailing(half_size), c);
    carry_b3 = carry_b3.wrapping_add(c);

    hi.trailing_mut(half_size).add_assign_limb(carry_b3);
}

/// Multiply two limb slices, adding the result to `out`.
///
/// `lhs` and `rhs` may have different lengths.
#[inline(never)]
#[track_caller]
pub const fn widening_mul(lhs: &[Limb], rhs: &[Limb], out: &mut UintRef, scratch: &mut UintRef) {
    assert!(
        lhs.len() + rhs.len() == out.len(),
        "invalid arguments to widening_mul"
    );
    let size = {
        let overlap = if lhs.len() < rhs.len() {
            lhs.len()
        } else {
            rhs.len()
        };
        overlap - (overlap & 1)
    };
    if size <= MAX_REDUCE_LIMBS {
        schoolbook::carrying_add_mul(lhs, rhs, out.as_mut_slice(), Limb::ZERO);
        return;
    }
    let (x, x_tail) = lhs.split_at(size);
    let (y, y_tail) = rhs.split_at(size);

    // Multiply the maximal number of matched limbs
    widening_mul_even(x, y, out.leading_mut(2 * size), scratch);

    // Handle trailing limbs
    if !x_tail.is_empty() {
        schoolbook::carrying_add_mul(
            x_tail,
            rhs,
            out.trailing_mut(size).as_mut_slice(),
            Limb::ZERO,
        );
    }
    if !y_tail.is_empty() {
        let (assign, tail) = out.trailing_mut(size).split_at_mut(size + y_tail.len());
        let carry = schoolbook::carrying_add_mul(y_tail, x, assign.as_mut_slice(), Limb::ZERO);
        tail.add_assign_limb(carry);
    }
}

/// Multiply two limb slices, adding the result to `out`.
///
/// `lhs` and `rhs` must have the same length and an even number of limbs.
pub const fn widening_mul_even(
    lhs: &[Limb],
    rhs: &[Limb],
    out: &mut UintRef,
    scratch: &mut UintRef,
) {
    let size = lhs.len();
    assert!(
        size & 1 == 0 && rhs.len() == size && out.len() == size * 2 && scratch.len() >= size * 2,
        "invalid arguments to widening_mul_even"
    );
    let half = size / 2;

    let (x0, x1) = UintRef::new(lhs).split_at(half);
    let (y0, y1) = UintRef::new(rhs).split_at(half);

    // Calculate z1 = (x0+x1)•(y0+y1) into the middle half of output
    let mut carry_b3 = compute_z1_mul(
        (x0, x1),
        (y0, y1),
        out.range_mut(half..size + half),
        scratch.split_at_mut(size),
    );

    // Calculate z0 = x0•y0 into scratch
    let (z0, scratch) = scratch.split_at_mut(size);
    z0.fill(Limb::ZERO);
    widening_mul(x0.as_slice(), y0.as_slice(), z0, scratch);

    // Add z0 to output
    let carry_b2 = out.leading_mut(size).carrying_add_assign(z0, Limb::ZERO);

    // Subtract z0•b from the output
    let mut c = out
        .range_mut(half..size + half)
        .borrowing_sub_assign(z0, Limb::ZERO);
    carry_b3 = carry_b3.wrapping_add(c);

    // Calculate z2 = x1•y1 into scratch
    let z2 = z0;
    z2.fill(Limb::ZERO);
    widening_mul(x1.as_slice(), y1.as_slice(), z2, scratch);

    // Subtract z2•b from the output
    c = out
        .range_mut(half..size + half)
        .borrowing_sub_assign(z2, Limb::ZERO);
    carry_b3 = carry_b3.wrapping_add(c);

    // Add z2.0•b^2 to the output
    c = out
        .range_mut(size..size + half)
        .carrying_add_assign(z2.leading(half), carry_b2);
    carry_b3 = carry_b3.wrapping_add(c);

    // Add z2.1•b^3 to the output and complete the carries
    out.trailing_mut(size + half)
        .carrying_add_assign(z2.trailing(half), carry_b3);
}

/// A helper function to compute `z1 = (x0+x1)(y0+y1)`
#[inline]
const fn compute_z1_mul(
    (x0, x1): (&UintRef, &UintRef),
    (y0, y1): (&UintRef, &UintRef),
    out: &mut UintRef,
    scratch: (&mut UintRef, &mut UintRef),
) -> Limb {
    debug_assert!(scratch.0.len() == out.len() && scratch.1.len() >= out.len());
    let half = out.len() / 2;
    let (s0, s1) = scratch.0.leading_mut(out.len()).split_at_mut(half);

    // Compute s0 = (x0 + x1) + s0c•b
    s0.copy_from(x0);
    let s0c = s0.carrying_add_assign(x1, Limb::ZERO);

    // Compute s1 = (y0 + y1) + s1c•b
    s1.copy_from(y0);
    let s1c = s1.carrying_add_assign(y1, Limb::ZERO);

    // Compute z1 = (x0 + x1)(y0 + y1), except for the high bit of each sum
    widening_mul(s0.as_slice(), s1.as_slice(), out, scratch.1);

    // Correct for missing high bits in multiplication
    // Add (s0•s1c)b to output
    let mut carry =
        out.trailing_mut(half)
            .conditional_carrying_add_assign(s0, Limb::ZERO, s1c.is_nonzero());
    // Add (s1•s0c)b to output
    let c =
        out.trailing_mut(half)
            .conditional_carrying_add_assign(s1, Limb::ZERO, s0c.is_nonzero());
    carry = carry.wrapping_add(c);

    // Add (s0c•s1c•b^3) to output, which will be addressed by the carry
    carry.wrapping_add(s0c.bitand(s1c))
}

/// Multiply two limb slices, computing only the lower limbs of the product, placing the result in `out`.
///
/// `lhs` and `rhs` may have different lengths.
/// `out` is assumed to be zeroed.
#[inline(never)]
pub const fn wrapping_mul(lhs: &[Limb], rhs: &[Limb], out: &mut UintRef, scratch: &mut UintRef) {
    assert!(lhs.len() == out.len(), "invalid arguments to wrapping_mul");
    let size = {
        let overlap = if lhs.len() < rhs.len() {
            lhs.len()
        } else {
            rhs.len()
        };
        overlap - (overlap & 1)
    };
    if size <= MAX_REDUCE_LIMBS * 2 {
        schoolbook::wrapping_mul(lhs, rhs, out.as_mut_slice());
        return;
    }
    let (x_head, x) = lhs.split_at(lhs.len() - size);
    let rhs = if rhs.len() > lhs.len() {
        rhs.split_at(lhs.len()).0
    } else {
        rhs
    };
    let y = rhs.split_at(rhs.len() - size).1;

    // Multiply the maximal number of matched limbs
    wrapping_mul_even(x, y, out.trailing_mut(out.len() - size), scratch);

    if !x_head.is_empty() {
        let end_pos = x_head.len() + y.len();
        let carry = schoolbook::carrying_add_mul(
            x_head,
            y,
            out.leading_mut(end_pos).as_mut_slice(),
            Limb::ZERO,
        );
        out.trailing_mut(end_pos).add_assign_limb(carry);
    }
}

/// Multiply two limb slices, adding only the lower limbs of the result to `out`.
///
/// `lhs` and `rhs` must have the same length and an even number of limbs.
#[inline(always)]
pub const fn wrapping_mul_even(
    lhs: &[Limb],
    rhs: &[Limb],
    out: &mut UintRef,
    scratch: &mut UintRef,
) {
    let size = lhs.len();
    assert!(
        size & 1 == 0 && rhs.len() == size && out.len() == size && scratch.len() >= size,
        "invalid arguments to wrapping_mul_even"
    );

    let half = size / 2;
    let even = half & 1 == 0;
    let (x0, x1) = UintRef::new(lhs).split_at(half);
    let (y0, y1) = UintRef::new(rhs).split_at(half);

    // Calculate z0 = x0•y0 into output
    widening_mul(x0.as_slice(), y0.as_slice(), out, scratch);

    // Add z1 = x0•y1 to second half of output
    wrapping_mul(
        x0.as_slice(),
        y1.as_slice(),
        out.trailing_mut(half),
        scratch,
    );

    // Add z2 = x0•y1 to second half of output
    wrapping_mul(
        x1.as_slice(),
        y0.as_slice(),
        out.trailing_mut(half),
        scratch,
    );
}

#[inline(never)]
pub(crate) const fn widening_square(limbs: &[Limb], out: &mut UintRef, scratch: &mut UintRef) {
    let size = limbs.len();
    assert!(
        out.len() == 2 * size && scratch.len() >= 2 * size,
        "invalid arguments to widening_square"
    );
    if size <= MAX_REDUCE_LIMBS * 2 || (size & 1) == 1 {
        let (lo, hi) = out.as_mut_slice().split_at_mut(size);
        schoolbook::square_wide(limbs, lo, hi);
        return;
    }

    let half = size / 2;
    let (x0, x1) = limbs.split_at(half);

    // Calculate z0 = x0^2 into first half of output
    widening_square(x0, out.leading_mut(size), scratch);

    // Calculate z2 = x1^2 into second half of output (z2•b^2)
    widening_square(x1, out.trailing_mut(size), scratch);

    // Calculate z1 = x0•x1 into scratch
    let (z1, scratch) = scratch.split_at_mut(size);
    z1.fill(Limb::ZERO);
    widening_mul(x0, x1, z1, scratch);

    // Multiply z1 by 2
    let carry1 = z1.shl1_assign();

    // Add 2z1•b to the output
    let carry2 = out
        .range_mut(half..size + half)
        .carrying_add_assign(z1, Limb::ZERO);

    // Apply the carries
    out.trailing_mut(size + half)
        .add_assign_limb(Limb::select(Limb::ZERO, Limb::ONE, carry1).wrapping_add(carry2));
}

#[inline(never)]
pub(crate) fn wrapping_square(limbs: &[Limb], out: &mut UintRef, scratch: &mut UintRef) {
    let size = limbs.len();
    assert!(
        out.len() == size && scratch.len() >= size,
        "invalid arguments to wrapping_square"
    );
    if size <= MAX_REDUCE_LIMBS * 2 || (size & 1) == 1 {
        schoolbook::wrapping_square(limbs, out.as_mut_slice());
        return;
    }

    let half = size / 2;
    let (x0, x1) = limbs.split_at(half);

    // Calculate z0 = x0^2 into the output
    widening_square(x0, out, scratch);

    // Calculate z1 = x0•x1 into scratch
    let (z1, z1_scratch) = scratch.split_at_mut(half);
    z1.fill(Limb::ZERO);
    wrapping_mul(x0, x1, z1, z1_scratch);

    // Multiply z1 by 2
    z1.shl1_assign();

    // Add 2z1•b to the output
    out.trailing_mut(half).carrying_add_assign(z1, Limb::ZERO);
}
