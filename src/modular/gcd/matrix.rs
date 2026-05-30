use ctutils::CtEq;

use super::ExtendedIntRef;
use crate::{Choice, Limb, UintRef, word};

#[derive(Debug, Copy, Clone)]
pub struct SignedLimb {
    pub sign: Choice,
    pub value: Limb,
}

impl SignedLimb {
    #[inline(always)]
    pub const fn new(value: Limb, sign: Choice) -> Self {
        Self { value, sign }
    }

    #[inline(always)]
    pub const fn from_limb(value: Limb) -> Self {
        let sign = value.bit(Limb::HI_BIT);
        Self {
            value: Limb::select(value, value.wrapping_neg(), sign),
            sign,
        }
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

#[derive(Debug, Copy, Clone)]
pub struct SignedLimbMatrix {
    pub m0: (SignedLimb, SignedLimb),
    pub m1: (SignedLimb, SignedLimb),
}

impl SignedLimbMatrix {
    #[inline(always)]
    pub const fn conditional_negate_top_row(&mut self, negate: Choice) {
        self.m0.0.sign = self.m0.0.sign.xor(negate);
        self.m0.1.sign = self.m0.1.sign.xor(negate);
    }

    #[inline(always)]
    pub const fn negate_top_row(&mut self) {
        self.m0.0.sign = self.m0.0.sign.not();
        self.m0.1.sign = self.m0.1.sign.not();
    }

    #[inline(always)]
    pub const fn conditional_negate_bottom_row(&mut self, negate: Choice) {
        self.m1.0.sign = self.m1.0.sign.xor(negate);
        self.m1.1.sign = self.m1.1.sign.xor(negate);
    }

    #[inline(always)]
    pub const fn negate_bottom_row(&mut self) {
        self.m1.0.sign = self.m1.0.sign.not();
        self.m1.1.sign = self.m1.1.sign.not();
    }

    #[inline(always)]
    pub const fn apply_unsigned<'a>(
        &self,
        a: &'a mut UintRef,
        b: &'a mut UintRef,
    ) -> ((ExtendedIntRef<'a>, Choice), (ExtendedIntRef<'a>, Choice)) {
        let (m0, m1) = (&self.m0, &self.m1);
        let m0_neg = m0.0.sign.xor(m0.1.sign);
        let m1_neg = m1.0.sign.xor(m1.1.sign);
        let m0_mask = Limb(word::choice_to_mask(m0_neg));
        let m1_mask = Limb(word::choice_to_mask(m1_neg));

        let (a_carry, b_carry) = apply_matrix_unsigned_carry(
            a,
            b,
            (m0.0.value, m0.1.value, m1.0.value, m1.1.value),
            m0_mask,
            m1_mask,
        );

        let a_hi = m0_mask.wrapping_mul(m0.0.value).wrapping_add(a_carry);
        let b_hi = m1_mask.wrapping_mul(m1.0.value).wrapping_add(b_carry);

        let (a, b) = (ExtendedIntRef::new(a, a_hi), ExtendedIntRef::new(b, b_hi));

        // return two choices indicating whether the results should be negated (ignored in update_fg)
        // based on the signs of the matrix terms
        ((a, m0_neg.and(m0.1.sign)), (b, m1_neg.and(m1.1.sign)))
    }

    #[inline(always)]
    pub const fn wrapping_apply(
        &self,
        a: &mut ExtendedIntRef<'_>,
        b: &mut ExtendedIntRef<'_>,
    ) -> (Choice, Choice) {
        let (m0, m1) = (&self.m0, &self.m1);
        let m0_neg = m0.0.sign.xor(m0.1.sign);
        let m1_neg = m1.0.sign.xor(m1.1.sign);
        let m0_mask = Limb(word::choice_to_mask(m0_neg));
        let m1_mask = Limb(word::choice_to_mask(m1_neg));
        let (a_hi, b_hi) = (a.hi, b.hi);

        let (a_carry, b_carry) = apply_matrix_unsigned_carry(
            a.lo,
            b.lo,
            (m0.0.value, m0.1.value, m1.0.value, m1.1.value),
            m0_mask,
            m1_mask,
        );

        a.hi = a_hi
            .bitxor(m0_mask)
            .wrapping_mul(m0.0.value)
            .wrapping_add(b_hi.wrapping_mul(m0.1.value))
            .wrapping_add(a_carry);

        b.hi = a_hi
            .bitxor(m1_mask)
            .wrapping_mul(m1.0.value)
            .wrapping_add(b_hi.wrapping_mul(m1.1.value))
            .wrapping_add(b_carry);

        // return two choices indicating whether the results should be negated
        // based on the signs of the matrix terms
        (m0_neg.and(m0.1.sign), m1_neg.and(m1.1.sign))
    }

    #[inline(always)]
    pub const fn wrapping_half_apply(
        &self,
        a: &mut ExtendedIntRef<'_>,
        b: &ExtendedIntRef,
    ) -> Choice {
        let m0 = &self.m0;
        let m0_neg = m0.0.sign.xor(m0.1.sign);
        let m0_mask = Limb(word::choice_to_mask(m0_neg));
        let (a_hi, b_hi) = (a.hi, b.hi);

        let a_carry =
            half_apply_matrix_unsigned_carry(a.lo, b.lo, (m0.0.value, m0.1.value), m0_mask);

        a.hi = a_hi
            .bitxor(m0_mask)
            .wrapping_mul(m0.0.value)
            .wrapping_add(b_hi.wrapping_mul(m0.1.value))
            .wrapping_add(a_carry);

        m0_neg.and(m0.1.sign)
    }
}

impl CtEq for SignedLimbMatrix {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.m0
            .0
            .ct_eq(&other.m0.0)
            .and(self.m0.1.ct_eq(&other.m0.1))
            .and(self.m1.0.ct_eq(&other.m1.0))
            .and(self.m1.1.ct_eq(&other.m1.1))
    }
}

impl PartialEq for SignedLimbMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

#[inline(always)]
const fn apply_matrix_unsigned_carry<'a>(
    a: &'a mut UintRef,
    b: &'a mut UintRef,
    (m00, m01, m10, m11): (Limb, Limb, Limb, Limb),
    m0_mask: Limb,
    m1_mask: Limb,
) -> (Limb, Limb) {
    let (a_lo, b_lo) = (&mut a.limbs, &mut b.limbs);

    let mut carry = [Limb::ZERO; 4];
    carry[2] = m0_mask.bitand(m00);
    carry[3] = m1_mask.bitand(m10);
    let mut i = 0;

    while i < a_lo.len() {
        let (b_m0, b_m1);
        (b_m0, carry[0]) = b_lo[i].carrying_mul_add(m01, Limb::ZERO, carry[0]);
        (b_m1, carry[1]) = b_lo[i].carrying_mul_add(m11, Limb::ZERO, carry[1]);
        ((a_lo[i], carry[2]), (b_lo[i], carry[3])) = (
            a_lo[i]
                .bitxor(m0_mask)
                .carrying_mul_add(m00, b_m0, carry[2]),
            a_lo[i]
                .bitxor(m1_mask)
                .carrying_mul_add(m10, b_m1, carry[3]),
        );
        i += 1;
    }

    let a_hi = carry[0].wrapping_add(carry[2]);
    let b_hi = carry[1].wrapping_add(carry[3]);
    (a_hi, b_hi)
}

#[inline(always)]
const fn half_apply_matrix_unsigned_carry<'a>(
    a: &'a mut UintRef,
    b: &'a UintRef,
    (m0, m1): (Limb, Limb),
    m_mask: Limb,
) -> Limb {
    let (a_lo, b_lo) = (&mut a.limbs, &b.limbs);

    let mut a_carry = Limb::ZERO;
    let mut b_carry = m_mask.bitand(m0);
    let mut b_m1;
    let mut i = 0;

    while i < a_lo.len() {
        (b_m1, b_carry) = b_lo[i].carrying_mul_add(m1, Limb::ZERO, b_carry);
        (a_lo[i], a_carry) = a_lo[i].bitxor(m_mask).carrying_mul_add(m0, b_m1, a_carry);
        i += 1;
    }

    a_carry.wrapping_add(b_carry)
}
