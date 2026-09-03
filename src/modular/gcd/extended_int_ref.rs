use core::fmt;

use crate::{Choice, Limb, NonZero, Odd, UintRef};

/// An extended integer reference type with an extra high limb.
pub struct ExtendedIntRef<'a> {
    pub(crate) hi: Limb,
    pub(crate) lo: &'a mut UintRef,
}

impl<'a> ExtendedIntRef<'a> {
    /// Constructs an extended integer from separate low limbs and a high (sign/overflow) limb.
    pub const fn new(lo: &'a mut UintRef, hi: Limb) -> Self {
        Self { lo, hi }
    }

    /// Takes the absolute value in place, then drops the (now-zero) high limb, returning the
    /// low limbs together with a `Choice` indicating whether `self` was negative.
    ///
    /// Panics if the high limb is not all-zeros or all-ones.
    #[inline(always)]
    #[track_caller]
    pub const fn abs_drop_extension(mut self) -> (&'a mut UintRef, Choice) {
        let neg = self.abs_assign();
        let lo = self.unsigned_drop_extension();
        (lo, neg)
    }

    /// Drops the high limb, returning the low limbs.
    ///
    /// Panics if the high limb is nonzero, i.e. if `self` does not fit in `lo` alone.
    #[inline(always)]
    #[track_caller]
    pub const fn unsigned_drop_extension(self) -> &'a mut UintRef {
        let (lo, hi) = self.split();
        assert!(hi.is_zero().to_bool_vartime(), "overflow");
        lo
    }

    /// Splits `self` into its low limbs and high (sign/overflow) limb.
    #[inline(always)]
    #[must_use]
    pub const fn split(self) -> (&'a mut UintRef, Limb) {
        (self.lo, self.hi)
    }
}

impl ExtendedIntRef<'_> {
    /// Returns whether `self` is negative as a [`Choice`].
    #[inline(always)]
    pub const fn is_negative(&self) -> Choice {
        self.hi.bit(Limb::HI_BIT)
    }

    /// Variable-time equivalent of [`Self::is_negative`].
    #[inline(always)]
    pub const fn is_negative_vartime(&self) -> bool {
        self.hi.bit_vartime(Limb::HI_BIT)
    }

    /// Sets `self` to its absolute value in place, returning a `Choice` indicating whether it
    /// was negative.
    #[inline(always)]
    pub const fn abs_assign(&mut self) -> Choice {
        self.abs_assign_carry().0
    }

    /// Like [`Self::abs_assign`], but also returns the carry value from the negation
    /// (unconditionally) indicating whether `self` is zero.
    #[inline(always)]
    pub const fn abs_assign_carry(&mut self) -> (Choice, Limb) {
        let apply = self.is_negative();
        let carry = self.conditional_carrying_neg_assign(apply);
        (apply, carry)
    }

    /// Shifts `self` right by `shift` bits (`shift < Limb::BITS`), sign-extending the vacated
    /// high bits.
    #[inline(always)]
    pub const fn shr_assign_limb(&mut self, shift: u32) {
        let sign_bits = Limb::choice_to_mask(self.is_negative()).unbounded_shl(Limb::BITS - shift);
        self.shr_assign_limb_unsigned(shift);
        self.hi = self.hi.bitor(sign_bits);
    }

    /// Shifts `self` right by `shift` bits (`shift < Limb::BITS`) without sign-extension.
    #[inline(always)]
    #[track_caller]
    pub const fn shr_assign_limb_unsigned(&mut self, shift: u32) {
        self.lo
            .shr_assign_limb_with_carry(shift, self.hi.unbounded_shl(Limb::BITS - shift));
        self.hi = self.hi.shr(shift);
    }

    /// Conditionally adds unsigned `rhs` into `self`'s low limbs, propagating the carry into
    /// the high limb.
    #[inline(always)]
    pub const fn conditional_wrapping_add_assign_unsigned(
        &mut self,
        rhs: &UintRef,
        choice: Choice,
    ) {
        let c = self.lo.conditional_add_assign(rhs, Limb::ZERO, choice);
        self.hi = self.hi.wrapping_add(c);
    }

    /// Conditionally negates `self` in place, returning the carry from negating `lo` (nonzero
    /// iff `lo` was zero).
    #[inline(always)]
    pub const fn conditional_carrying_neg_assign(&mut self, apply: Choice) -> Limb {
        let mut c = self.lo.conditional_carrying_neg_assign(apply);
        let neg;
        (neg, c) = self.hi.not().overflowing_add(c);
        self.hi = Limb::select(self.hi, neg, apply);
        c
    }

    /// Negates `self` in place (wrapping, unconditional).
    #[inline(always)]
    pub const fn wrapping_neg_assign(&mut self) {
        let c = self.lo.carrying_neg_assign();
        self.hi = self.hi.not().wrapping_add(c);
    }

    /// Adds `rhs * rhs_mul` into `self`'s low limbs, propagating the carry into the high limb
    /// and returning any carry out of that.
    #[inline(always)]
    pub const fn wrapping_add_assign_unsigned_mul_limb(
        &mut self,
        rhs: &UintRef,
        rhs_mul: Limb,
    ) -> Limb {
        let mut carry = self
            .lo
            .carrying_add_assign_mul_limb(rhs, rhs_mul, Limb::ZERO);
        (self.hi, carry) = self.hi.overflowing_add(carry);
        carry
    }

    /// Adds `rhs` into `self`, negating `rhs` first (in constant time) when `rhs_sign` is set.
    /// Requires `rhs` to have no more limbs than `self`'s low part.
    #[inline(always)]
    pub const fn wrapping_add_assign_signed(&mut self, rhs: &UintRef, rhs_sign: Choice) {
        debug_assert!(self.lo.nlimbs() >= rhs.nlimbs());

        let mask = Limb::choice_to_mask(rhs_sign);
        let mut carry = mask.shr(Limb::HI_BIT);
        let mut i = 0;
        let lhs_lo = &mut self.lo.limbs;
        while i < rhs.limbs.len() {
            (lhs_lo[i], carry) = lhs_lo[i].carrying_add(rhs.limbs[i].bitxor(mask), carry);
            i += 1;
        }
        while i < lhs_lo.len() {
            (lhs_lo[i], carry) = lhs_lo[i].carrying_add(mask, carry);
            i += 1;
        }
        self.hi = self.hi.wrapping_add(mask).wrapping_add(carry);
    }

    /// Performs one Montgomery-style division step: adds a multiple of `m` to zero out the low
    /// `k` bits (`k <= Limb::BITS`) of `self`, then shifts right by `k`, dividing `self` by
    /// `2^k` modulo `m`. `m_inv` must equal `-m^-1 mod 2^k`. This does not necessarily guarantee a
    /// result in the range [0, m).
    #[inline(always)]
    pub const fn limb_div2k_mod_assign(&mut self, m: &Odd<UintRef>, m_inv: Limb, k: u32) {
        let was_neg = self.is_negative();
        let quo = m_inv
            .wrapping_neg()
            .wrapping_mul(self.lo.limbs[0])
            .restrict_bits(k);
        let carry = self.wrapping_add_assign_unsigned_mul_limb(m.as_ref(), quo);
        self.shr_assign_limb_unsigned(k);
        let sign_and_carry = Limb::choice_to_mask(was_neg).wrapping_add(carry);
        self.hi = self.hi.bitor(sign_and_carry.unbounded_shl(Limb::BITS - k));
    }

    /// Performs a modular division by `2^k`.
    ///
    /// This method is variable-time in `k` only. For this use case `k` is public
    /// due to being part of a deterministic schedule or part of a variable-time reduction.
    ///
    /// Leaves the result unnormalized, the same as [`Self::limb_div2k_mod_assign`].
    #[inline(always)]
    pub const fn div2k_mod_assign_vartime(&mut self, m: &Odd<UintRef>, m_inv: Limb, mut k: u32) {
        // Perform full-limb reductions
        while k >= Limb::BITS {
            let was_neg = self.is_negative();
            let quo = m_inv.wrapping_neg().wrapping_mul(self.lo.limbs[0]);
            let carry = self.wrapping_add_assign_unsigned_mul_limb(m.as_ref(), quo);
            self.lo.unbounded_shr_assign_by_limbs(1);
            self.lo.limbs[self.lo.nlimbs() - 1] = self.hi;
            self.hi = Limb::choice_to_mask(was_neg).wrapping_add(carry);
            k -= Limb::BITS;
        }
        // Perform a sub-limb reduction if needed
        if k != 0 {
            self.limb_div2k_mod_assign(m, m_inv, k);
        }
    }

    /// Nudges `self` toward the range `[0, p)` with a single conditional correction: adds `p`
    /// if `self` is negative, or subtracts `p` otherwise.
    #[inline(always)]
    pub const fn try_reduce_mod(&mut self, p: &NonZero<UintRef>) {
        let sign = self.is_negative();
        self.wrapping_add_assign_signed(p.as_ref(), sign.not());
    }

    /// Vartime equivalent of [`Self::try_reduce_mod`].
    #[inline(always)]
    pub const fn try_reduce_mod_vartime(&mut self, p: &NonZero<UintRef>) {
        if self.is_negative_vartime() {
            self.wrapping_add_assign_signed(p.as_ref(), Choice::FALSE);
        } else if !self.hi.is_zero_vartime() || self.lo.cmp_vartime(p.as_ref()).is_ge() {
            self.wrapping_add_assign_signed(p.as_ref(), Choice::TRUE);
        }
    }
}

impl fmt::Debug for ExtendedIntRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExtendedIntRef(0x{self:X})")
    }
}

impl fmt::Display for ExtendedIntRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperHex::fmt(self, f)
    }
}

impl fmt::UpperHex for ExtendedIntRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "0x")?;
        }
        write!(f, "{:X}{:X}", &self.hi, &self.lo)?;
        Ok(())
    }
}

/// How many bits of a high (sign/overflow) limb `hi` carry information beyond a plain sign
/// extension of whatever value it's paired with; i.e. how far that value overflows the range
/// representable by its low limbs alone.
#[inline(always)]
pub(super) const fn hi_overflow_vartime(hi: Limb) -> u32 {
    Limb::BITS
        - if hi.bit_vartime(Limb::HI_BIT) {
            hi.leading_ones()
        } else {
            hi.leading_zeros()
        }
}
