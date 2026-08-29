use ctutils::CtEq;

use super::ExtendedIntRef;
use crate::{Choice, Limb, UintRef};

/// A [`Limb`] paired with an explicit sign -- sign-magnitude representation, as opposed to
/// `Limb`'s own two's-complement wrapping arithmetic. Used as [`SignedLimbMatrix`]'s entries: the
/// various per-batch gcd matrices (`bingcd`'s, `safegcd`'s) need genuinely negative coefficients,
/// but only ever within a single limb's magnitude, so tracking the sign alongside a plain
/// non-negative `Limb` is cheaper than reasoning about `Limb`'s wraparound directly.
#[derive(Debug, Copy, Clone)]
pub struct SignedLimb {
    /// `true` iff this value is negative.
    pub sign: Choice,
    /// The magnitude -- always non-negative, regardless of `sign`.
    pub value: Limb,
}

impl SignedLimb {
    /// Pairs an already-non-negative `value` with an explicit `sign`.
    #[inline(always)]
    pub const fn new(value: Limb, sign: Choice) -> Self {
        Self { value, sign }
    }
}

impl CtEq for SignedLimb {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.value
            .ct_eq(&other.value)
            .and(self.sign.ct_eq(&other.sign))
    }
}

impl PartialEq for SignedLimb {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

/// A 2x2 matrix of [`SignedLimb`] entries, applied as a linear transform to a pair of big
/// integers: `(a, b) -> (r0.0*a + r0.1*b, r1.0*a + r1.1*b)`. The shared representation `bingcd`'s
/// (`BingcdMatrix`) and `safegcd`'s (`SafegcdMatrix`) own, differently-shaped per-batch matrices
/// both convert into before actually applying themselves -- either to the evolving `f`/`g` gcd
/// state directly, or (`vartime.rs`'s own coefficient-pair tracker) to a running composed
/// coefficient matrix instead. See [`Self::wrapping_apply`], [`Self::wrapping_apply_unsigned`],
/// and [`Self::wrapping_half_apply`] for the different operand-signedness/completeness tradeoffs
/// each apply variant makes.
#[derive(Debug, Copy, Clone)]
pub struct SignedLimbMatrix {
    /// Top row: computes the new `a`.
    pub r0: (SignedLimb, SignedLimb),
    /// Bottom row: computes the new `b`.
    pub r1: (SignedLimb, SignedLimb),
}

impl SignedLimbMatrix {
    /// Flips the sign of both entries in the top row, without touching their magnitudes.
    #[inline(always)]
    pub const fn negate_top_row(&mut self) {
        self.r0.0.sign = self.r0.0.sign.not();
        self.r0.1.sign = self.r0.1.sign.not();
    }

    /// Flips the sign of both entries in the bottom row, without touching their magnitudes.
    #[inline(always)]
    pub const fn negate_bottom_row(&mut self) {
        self.r1.0.sign = self.r1.0.sign.not();
        self.r1.1.sign = self.r1.1.sign.not();
    }

    /// Applies the same linear transform as [`Self::wrapping_apply`], but to *unsigned* operands
    /// (`a`, `b` are plain, non-negative `UintRef`s here, not signed `ExtendedIntRef`s) for
    /// callers that already keep their own sign bookkeeping outside the matrix machinery and would
    /// rather hand it non-negative buffers directly.
    ///
    /// Each row's value is computed, forced to its absolute value (`abs_assign`), and the true sign
    /// is reported back as one of the two returned `Choice`s (`a_negated`/`b_negated`).
    #[inline(always)]
    pub const fn wrapping_apply_unsigned<'a, 'b>(
        &self,
        a: &'a mut UintRef,
        b: &'b mut UintRef,
    ) -> (ExtendedIntRef<'a>, ExtendedIntRef<'b>, Choice, Choice) {
        let (a_lo, b_lo) = (&mut a.limbs, &mut b.limbs);
        assert!(a_lo.len() == b_lo.len());

        let (r0, r1) = (self.r0, self.r1);
        let (r0_neg, r1_neg) = (r0.0.sign.xor(r0.1.sign), r1.0.sign.xor(r1.1.sign));
        let (r0_mask, r1_mask) = (Limb::choice_to_mask(r0_neg), Limb::choice_to_mask(r1_neg));
        let (m00, m01, m10, m11) = (r0.0.value, r0.1.value, r1.0.value, r1.1.value);
        let mut carry: [Limb; 4] = [
            Limb::ZERO,
            Limb::ZERO,
            r0_mask.bitand(m00),
            r1_mask.bitand(m10),
        ];
        let mut i = 0;

        while i < a_lo.len() {
            let (b_m0, b_m1);
            (b_m0, carry[0]) = b_lo[i].carrying_mul_add(m01, Limb::ZERO, carry[0]);
            (b_m1, carry[1]) = b_lo[i].carrying_mul_add(m11, Limb::ZERO, carry[1]);
            ((a_lo[i], carry[2]), (b_lo[i], carry[3])) = (
                a_lo[i]
                    .bitxor(r0_mask)
                    .carrying_mul_add(m00, b_m0, carry[2]),
                a_lo[i]
                    .bitxor(r1_mask)
                    .carrying_mul_add(m10, b_m1, carry[3]),
            );
            i += 1;
        }

        let (a_carry, b_carry) = (
            carry[0].wrapping_add(carry[2]),
            carry[1].wrapping_add(carry[3]),
        );
        let (a_hi, b_hi) = (
            r0_mask.wrapping_mul(m00).wrapping_add(a_carry),
            r1_mask.wrapping_mul(m10).wrapping_add(b_carry),
        );
        let (mut a_ext, mut b_ext) = (ExtendedIntRef::new(a, a_hi), ExtendedIntRef::new(b, b_hi));
        let (a_pre_neg, a_carry) = a_ext.abs_assign_carry();
        let (b_pre_neg, b_carry) = b_ext.abs_assign_carry();
        // `pre_neg.xor(row.1.sign)` alone is wrong at exactly zero: `is_negative`-style sign
        // checks (matching what `wrapping_apply`'s own `a`/`b` would report) always read zero as
        // non-negative regardless of which "side" produced it, but the XOR flips it to `true`
        // whenever `row.1.sign` is set and the pre-abs value happened to be zero (its own
        // `pre_neg` is `false` either way, since zero has no sign bit set). Masking by `nonzero`
        // (read off the abs's own negation carry -- see `abs_assign_reporting_nonzero`) corrects
        // that one case without disturbing any other.
        let a_negated = a_pre_neg.xor(r0.1.sign).and(a_carry.is_zero());
        let b_negated = b_pre_neg.xor(r1.1.sign).and(b_carry.is_zero());
        (a_ext, b_ext, a_negated, b_negated)
    }

    /// Applies the matrix to signed operands `a`, `b` in-place, computing
    /// `a = r0.0*a + r0.1*b` and `b = r1.0*a + r1.1*b`
    #[inline(always)]
    pub const fn wrapping_apply(&self, a: &mut ExtendedIntRef<'_>, b: &mut ExtendedIntRef<'_>) {
        let (a_hi, b_hi) = (a.hi, b.hi);
        let (a_lo, b_lo) = (&mut a.lo.limbs, &mut b.lo.limbs);
        assert!(a_lo.len() == b_lo.len());

        // Each of the four terms gets its own independent two's-complement negation (its own XOR
        // mask plus a matching carry-seed). Each row's `b`-term is computed first and fed
        // straight in as the `addend` of the `a`-term's `carrying_mul_add`, so the two terms land
        // pre-summed with no separate `carrying_add` needed -- the `a`-side carry doubles as both
        // that term's multiply-carry and the row's sum-carry, threaded as-is into the next limb's
        // `a`-term call.
        let (r0, r1) = (self.r0, self.r1);
        let (m00, m01, m10, m11) = (r0.0.value, r0.1.value, r1.0.value, r1.1.value);
        let (m00_mask, m01_mask, m10_mask, m11_mask) = (
            Limb::choice_to_mask(r0.0.sign),
            Limb::choice_to_mask(r0.1.sign),
            Limb::choice_to_mask(r1.0.sign),
            Limb::choice_to_mask(r1.1.sign),
        );
        let mut carry = [
            m00_mask.bitand(m00),
            m01_mask.bitand(m01),
            m10_mask.bitand(m10),
            m11_mask.bitand(m11),
        ];
        let mut i = 0;

        while i < a_lo.len() {
            let (a_val, b_val) = (a_lo[i], b_lo[i]);
            let (term_b0, term_b1);

            (term_b0, carry[1]) =
                b_val
                    .bitxor(m01_mask)
                    .carrying_mul_add(m01, Limb::ZERO, carry[1]);
            (a_lo[i], carry[0]) = a_val
                .bitxor(m00_mask)
                .carrying_mul_add(m00, term_b0, carry[0]);

            (term_b1, carry[3]) =
                b_val
                    .bitxor(m11_mask)
                    .carrying_mul_add(m11, Limb::ZERO, carry[3]);
            (b_lo[i], carry[2]) = a_val
                .bitxor(m10_mask)
                .carrying_mul_add(m10, term_b1, carry[2]);

            i += 1;
        }

        // One more "logical limb" step for the hi extension. There's no limb beyond it, so any
        // further carry-out here is simply truncated.
        let term_b0_hi = b_hi
            .bitxor(m01_mask)
            .wrapping_mul(m01)
            .wrapping_add(carry[1]);
        let (a_hi_new, _) = a_hi
            .bitxor(m00_mask)
            .carrying_mul_add(m00, term_b0_hi, carry[0]);
        a.hi = a_hi_new;

        let term_b1_hi = b_hi
            .bitxor(m11_mask)
            .wrapping_mul(m11)
            .wrapping_add(carry[3]);
        let (b_hi_new, _) = a_hi
            .bitxor(m10_mask)
            .carrying_mul_add(m10, term_b1_hi, carry[2]);
        b.hi = b_hi_new;
    }
}

impl CtEq for SignedLimbMatrix {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.r0
            .0
            .ct_eq(&other.r0.0)
            .and(self.r0.1.ct_eq(&other.r0.1))
            .and(self.r1.0.ct_eq(&other.r1.0))
            .and(self.r1.1.ct_eq(&other.r1.1))
    }
}

impl PartialEq for SignedLimbMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}
