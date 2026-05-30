use crate::modular::gcd::{ExtendedIntRef, SignedLimb, SignedLimbMatrix};
use crate::{Choice, Limb, UintRef};

#[derive(Debug, Copy, Clone)]
pub struct BingcdMatrix {
    pub(crate) m0: (Limb, Limb),
    pub(crate) m1: (Limb, Limb),
    pub(crate) pattern: Choice,
    pub(crate) k: u32,
}

impl BingcdMatrix {
    pub const UNIT: Self = Self {
        m0: (Limb::ONE, Limb::ZERO),
        m1: (Limb::ZERO, Limb::ONE),
        pattern: Choice::TRUE,
        k: 0,
    };

    /// Swap the rows of this matrix.
    #[inline(always)]
    pub const fn swap_rows(&mut self) {
        (self.m0, self.m1) = (self.m1, self.m0);
        self.pattern = self.pattern.not();
    }

    /// Swap the rows of this matrix if `swap` is truthy. Otherwise, do nothing.
    #[inline(always)]
    pub(crate) const fn conditional_swap_rows(&mut self, swap: Choice) {
        Limb::ct_conditional_swap(&mut self.m0.0, &mut self.m1.0, swap);
        Limb::ct_conditional_swap(&mut self.m0.1, &mut self.m1.1, swap);
        self.pattern = self.pattern.xor(swap);
    }

    /// Subtract the bottom row from the top.
    #[inline(always)]
    pub const fn subtract_bottom_row_from_top(&mut self) {
        // NB: the matrix entries have implicit opposite signs
        self.m0.0 = self.m0.0.wrapping_add(self.m1.0);
        self.m0.1 = self.m0.1.wrapping_add(self.m1.1);
    }

    /// Subtract the bottom row from the top if `subtract` is truthy. Otherwise, do nothing.
    #[inline(always)]
    pub(crate) const fn conditional_subtract_bottom_row_from_top(&mut self, subtract: Choice) {
        // Note: because the signs of the internal representation are stored in `pattern`,
        // subtracting one row from another involves _adding_ these rows instead.
        self.m0.0 = Limb::select(self.m0.0, self.m0.0.wrapping_add(self.m1.0), subtract);
        self.m0.1 = Limb::select(self.m0.1, self.m0.1.wrapping_add(self.m1.1), subtract);
    }

    /// Double the bottom row of the matrix.
    #[inline(always)]
    pub const fn double_bottom_row(&mut self, n: u32) {
        self.m1.0 = self.m1.0.shl(n);
        self.m1.1 = self.m1.1.shl(n);
        self.k += n;
    }

    /// Double the bottom row of this matrix if `double` is truthy. Otherwise, do nothing.
    #[inline(always)]
    pub(crate) const fn conditional_double_bottom_row(&mut self, double: Choice) {
        let shift = double.select_u32(0, 1);
        self.m1.0 = self.m1.0.shl(shift);
        self.m1.1 = self.m1.1.shl(shift);
        self.k += shift;
    }

    #[inline(always)]
    pub const fn apply_unsigned(&self, a: &mut UintRef, b: &mut UintRef) -> SignedLimbMatrix {
        let mut matrix = self.signed_limb_matrix();
        let ((mut a, a_neg), (mut b, b_neg)) = matrix.apply_unsigned(a, b);

        a.conditional_wrapping_neg_assign(a_neg);
        matrix.conditional_negate_top_row(a_neg);
        a.shr_assign_limb_unsigned(self.k);
        a.unsigned_drop_extension();

        b.conditional_wrapping_neg_assign(b_neg);
        matrix.conditional_negate_bottom_row(b_neg);
        b.shr_assign_limb_unsigned(self.k);
        b.unsigned_drop_extension();

        matrix
    }

    #[inline(always)]
    pub const fn apply_unsigned_vartime(
        &self,
        a: &mut UintRef,
        b: &mut UintRef,
    ) -> SignedLimbMatrix {
        let mut matrix = self.signed_limb_matrix();
        let ((mut a, a_neg), (mut b, b_neg)) = matrix.apply_unsigned(a, b);

        if a_neg.to_bool_vartime() {
            a.wrapping_neg_assign();
            matrix.negate_top_row();
        }
        a.shr_assign_limb_unsigned(self.k);
        a.unsigned_drop_extension();

        if b_neg.to_bool_vartime() {
            b.wrapping_neg_assign();
            matrix.negate_bottom_row();
        }
        b.shr_assign_limb_unsigned(self.k);
        b.unsigned_drop_extension();

        matrix
    }

    #[inline(always)]
    pub const fn wrapping_apply_vartime(
        &self,
        a: &mut ExtendedIntRef<'_>,
        b: &mut ExtendedIntRef<'_>,
    ) -> SignedLimbMatrix {
        let mut matrix = self.signed_limb_matrix();
        let (a_neg, b_neg) = matrix.wrapping_apply(a, b);

        if a_neg.to_bool_vartime() {
            a.wrapping_neg_assign();
            matrix.negate_top_row();
        }
        if b_neg.to_bool_vartime() {
            b.wrapping_neg_assign();
            matrix.negate_bottom_row();
        }

        matrix
    }

    #[inline(always)]
    const fn signed_limb_matrix(&self) -> SignedLimbMatrix {
        let pat = self.pattern;
        SignedLimbMatrix {
            m0: (
                SignedLimb::new(self.m0.0, pat.not()),
                SignedLimb::new(self.m0.1, pat),
            ),
            m1: (
                SignedLimb::new(self.m1.0, pat),
                SignedLimb::new(self.m1.1, pat.not()),
            ),
        }
    }
}

impl PartialEq for BingcdMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.signed_limb_matrix().eq(&other.signed_limb_matrix())
    }
}
