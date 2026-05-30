use core::fmt;

use crate::{Choice, Limb, NonZero, Odd, UintRef, word};

/// An extended integer reference type with an extra high limb.
pub struct ExtendedIntRef<'a> {
    pub(crate) hi: Limb,
    pub(crate) lo: &'a mut UintRef,
}

impl<'a> ExtendedIntRef<'a> {
    pub const fn new(lo: &'a mut UintRef, hi: Limb) -> Self {
        Self { lo, hi }
    }

    #[inline(always)]
    pub const fn abs_drop_extension(mut self) -> (&'a mut UintRef, Choice) {
        let neg = self.abs_assign();
        let lo = self.unsigned_drop_extension();
        (lo, neg)
    }

    #[inline(always)]
    pub const fn unsigned_drop_extension(self) -> &'a mut UintRef {
        let (lo, hi) = self.split();
        assert!(hi.is_zero().to_bool_vartime(), "overflow");
        lo
    }

    #[inline(always)]
    #[must_use]
    pub const fn split(self) -> (&'a mut UintRef, Limb) {
        (self.lo, self.hi)
    }
}

impl ExtendedIntRef<'_> {
    const fn nlimbs(&self) -> usize {
        self.lo.nlimbs() + 1
    }

    #[inline(always)]
    pub const fn is_negative(&self) -> Choice {
        self.hi.bit(Limb::HI_BIT)
    }

    #[inline(always)]
    pub const fn is_negative_vartime(&self) -> bool {
        self.hi.bit_vartime(Limb::HI_BIT)
    }

    #[inline(always)]
    pub const fn low_limb(&self) -> Limb {
        self.lo.limbs[0]
    }

    #[inline(always)]
    pub const fn abs_assign(&mut self) -> Choice {
        let neg = self.is_negative();
        self.conditional_wrapping_neg_assign(neg);
        neg
    }

    #[inline(always)]
    pub const fn shr_assign_limb(&mut self, shift: u32) {
        let sign_bits =
            Limb(word::choice_to_mask(self.is_negative())).unbounded_shl(Limb::BITS - shift);
        self.shr_assign_limb_unsigned(shift);
        self.hi = self.hi.bitor(sign_bits);
    }

    #[inline(always)]
    pub const fn shr_assign_limb_unsigned(&mut self, shift: u32) {
        // we should only be shifting zeros
        debug_assert!(self.lo.limbs[0].restrict_bits(shift).is_zero_vartime());

        self.lo
            .shr_assign_limb_with_carry(shift, self.hi.unbounded_shl(Limb::BITS - shift));
        self.hi = self.hi.shr(shift);
    }

    #[inline(always)]
    pub const fn shl_assign_vartime(&mut self, shift: u32) {
        debug_assert!(shift < self.lo.bits_precision());
        let hi_pos = self.lo.bits_precision().wrapping_sub(shift);
        self.hi = if shift < Limb::BITS {
            self.hi
                .shl(shift)
                .bitor(self.lo.limbs[self.lo.nlimbs() - 1].unbounded_shr(Limb::BITS - shift))
        } else {
            Limb(self.lo.unbounded_extract_word_vartime(hi_pos))
        };
        self.lo.unbounded_shl_assign_vartime(shift);
    }

    #[inline(always)]
    pub const fn wrapping_sub_assign_mul_limb(&mut self, rhs: &Self, rhs_mul: Limb) {
        debug_assert!(self.nlimbs() == rhs.nlimbs());

        let mut carry = Limb::ZERO;
        let mut borrow = Limb::ZERO;
        let mut sub;
        let mut i = 0;

        while i < self.lo.limbs.len() {
            (sub, carry) = rhs_mul.carrying_mul_add(rhs.lo.limbs[i], Limb::ZERO, carry);
            (self.lo.limbs[i], borrow) = self.lo.limbs[i].borrowing_sub(sub, borrow);
            i += 1;
        }
        (sub, _) = rhs_mul.carrying_mul_add(rhs.hi, Limb::ZERO, carry);
        (self.hi, _) = self.hi.borrowing_sub(sub, borrow);
    }

    #[inline(always)]
    pub const fn conditional_wrapping_add_assign_unsigned(
        &mut self,
        rhs: &UintRef,
        choice: Choice,
    ) {
        let c = self.lo.conditional_add_assign(rhs, Limb::ZERO, choice);
        self.hi = Limb::select(self.hi, self.hi.wrapping_add(c), choice);
    }

    #[inline(always)]
    pub const fn conditional_wrapping_neg_assign(&mut self, neg: Choice) {
        let c = self.lo.conditional_wrapping_neg_assign(neg);
        self.hi = Limb::select(self.hi, self.hi.not(), neg).wrapping_add(c);
    }

    #[inline(always)]
    pub const fn wrapping_neg_assign(&mut self) {
        let c = self.lo.wrapping_neg_assign();
        self.hi = self.hi.not().wrapping_add(c);
    }

    #[inline(always)]
    pub const fn wrapping_add_assign_mul_limb(&mut self, rhs: &UintRef, rhs_mul: Limb) -> Limb {
        let mut carry = self
            .lo
            .carrying_add_assign_mul_limb(rhs, rhs_mul, Limb::ZERO);
        (self.hi, carry) = self.hi.carrying_add(Limb::ZERO, carry);
        carry
    }

    #[inline(always)]
    pub const fn wrapping_add_assign_signed(&mut self, rhs: &UintRef, rhs_sign: Choice) {
        debug_assert!(self.lo.nlimbs() == rhs.nlimbs());

        let mask = Limb(word::choice_to_mask(rhs_sign));
        let mut carry = Limb::select(Limb::ZERO, Limb::ONE, rhs_sign);
        let mut i = 0;
        let lo = &mut self.lo.limbs;
        while i < lo.len() {
            (lo[i], carry) = lo[i].carrying_add(rhs.limbs[i].bitxor(mask), carry);
            i += 1;
        }
        self.hi = self.hi.wrapping_add(mask).wrapping_add(carry);
    }

    #[inline(always)]
    pub const fn limb_div2k_mod_assign(&mut self, m: &Odd<UintRef>, m_inv: Limb, k: u32) {
        let quo = m_inv
            .wrapping_neg()
            .wrapping_mul(self.lo.limbs[0])
            .restrict_bits(k);
        self.wrapping_add_assign_mul_limb(m.as_ref(), quo);
        self.shr_assign_limb(k);
    }

    #[inline(always)]
    pub const fn div2k_mod_assign_vartime(&mut self, m: &Odd<UintRef>, m_inv: Limb, mut k: u32) {
        let mut is_neg = self.is_negative_vartime();
        while k >= Limb::BITS {
            let quo = m_inv.wrapping_neg().wrapping_mul(self.lo.limbs[0]);
            let carry = self.wrapping_add_assign_mul_limb(m.as_ref(), quo);
            is_neg &= carry.is_zero_vartime();
            let mut i = self.lo.nlimbs();
            let mut carry = self.hi;
            while i > 0 {
                i -= 1;
                (self.lo.limbs[i], carry) = (carry, self.lo.limbs[i]);
            }
            self.hi = if is_neg { Limb::MAX } else { Limb::ZERO };
            k -= Limb::BITS;
        }
        if k != 0 {
            self.limb_div2k_mod_assign(m, m_inv, k);
        }
    }

    #[inline(always)]
    pub const fn try_reduce_mod(&mut self, p: &NonZero<UintRef>) {
        let sign = self.is_negative();
        self.wrapping_add_assign_signed(p.as_ref(), sign.not());
    }

    #[inline(always)]
    pub const fn is_zero_vartime(&self) -> bool {
        self.lo.is_zero_vartime() && self.hi.is_zero_vartime()
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
