use ctutils::Choice;

use super::UintRef;
use crate::{Limb, Odd, primitives::u32_min};

impl UintRef {
    #[inline(always)]
    #[track_caller]
    pub(crate) const fn bounded_div2k_mod_assign_with_reciprocal(
        &mut self,
        hi: &mut Self,
        k: u32,
        k_upper_bound: u32,
        m: &Odd<UintRef>,
        m_inv: Limb,
    ) {
        let m = m.as_ref();
        let m_neg_inv = m_inv.wrapping_neg();
        assert!(
            m.nlimbs() == self.nlimbs(),
            "input size does not match modulus size"
        );
        let k = u32_min(k, k_upper_bound);

        // Perform a single sub-limb reduction, which could be zero bits.
        let sub_limb_k = k & (Limb::BITS - 1);
        let quo = m_neg_inv
            .wrapping_mul(self.limbs[0])
            .restrict_bits(sub_limb_k);
        let mut carry = self.carrying_add_assign_mul_limb(m, quo, Limb::ZERO);
        carry = hi.add_assign_limb(carry);
        carry = carry.unbounded_shl(Limb::BITS - sub_limb_k);
        carry = hi.shr_assign_limb_with_carry(sub_limb_k, carry);
        self.shr_assign_limb_with_carry(sub_limb_k, carry);

        let k_upper_limbs = k_upper_bound >> Limb::LOG2_BITS;
        let k_limbs = k >> Limb::LOG2_BITS;
        let mut i = 0;

        while i < k_upper_limbs {
            let apply = Choice::from_u32_lt(i, k_limbs);

            let quo = Limb::select(Limb::ZERO, m_neg_inv.wrapping_mul(self.limbs[0]), apply);
            let mut carry = self.carrying_add_assign_mul_limb(m, quo, Limb::ZERO);
            carry = hi.add_assign_limb(carry);
            // Conditionally shift by zero or one limbs
            let mut j = hi.nlimbs();
            while j > 0 {
                j -= 1;
                Limb::ct_conditional_swap(&mut carry, &mut hi.limbs[j], apply);
            }
            j = self.nlimbs();
            while j > 0 {
                j -= 1;
                Limb::ct_conditional_swap(&mut carry, &mut self.limbs[j], apply);
            }

            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Limb, Odd, U128, Uint};

    #[test]
    fn div2k_mod_expected() {
        #[track_caller]
        fn check<const N: usize>(
            inp: Uint<N>,
            q: Odd<Uint<N>>,
            k: u32,
            k_upper: u32,
            expect: Uint<N>,
        ) {
            let mut res = inp;
            let recip = q.as_uint_ref().invert_mod_limb();
            let mut hi = Limb::ZERO;
            res.as_mut_uint_ref()
                .bounded_div2k_mod_assign_with_reciprocal(
                    hi.as_mut_uint_ref(),
                    k,
                    k_upper,
                    q.as_uint_ref(),
                    recip,
                );
            assert_eq!(hi, Limb::ZERO);
            assert_eq!(res, expect);
        }

        let q = U128::from(3u64).to_odd().unwrap();

        // Do nothing
        check(
            U128::ONE.shl_vartime(64),
            q,
            0,
            0,
            U128::ONE.shl_vartime(64),
        );

        // Simply shift out 5 factors
        check(
            U128::ONE.shl_vartime(64),
            q,
            5,
            5,
            U128::ONE.shl_vartime(59),
        );

        // Add in one factor of q
        check(U128::ONE, q, 1, 1, U128::from(2u64));

        // Add in many factors of q
        check(U128::from(8u64), q, 17, 17, U128::ONE);

        // Larger q
        let q = U128::from(2864434311u64).to_odd().unwrap();
        check(U128::from(8u64), q, 17, 17, U128::from(303681787u64));

        // Shift greater than Limb::BITS
        let q = U128::from_be_hex("0000AAAABBBB33330000AAAABBBB3333")
            .to_odd()
            .unwrap();
        check(
            U128::MAX,
            q,
            71,
            71,
            U128::from_be_hex("00002D6F169DBBF300002D6F169DBBF3"),
        );

        // Have k_bound restrict the number of shifts to 0
        check(U128::MAX, q, 71, 0, U128::MAX);

        // Have k_bound < k
        check(
            U128::MAX,
            q,
            71,
            30,
            U128::from_be_hex("000071EEB6013E76000071EEB6013E76"),
        );

        // Have k_bound >> k
        check(
            U128::MAX,
            q,
            30,
            127,
            U128::from_be_hex("000071EEB6013E76000071EEB6013E76"),
        );
    }
}
