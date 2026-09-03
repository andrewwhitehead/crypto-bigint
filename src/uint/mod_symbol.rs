//! Support for computing modular symbols.

use crate::{JacobiSymbol, Odd, Uint, modular::gcd};

impl<const LIMBS: usize> Uint<LIMBS> {
    /// Compute the Jacobi symbol `(self|rhs)`.
    ///
    /// For prime `rhs`, this corresponds to the Legendre symbol and
    /// indicates whether `self` is quadratic residue modulo `rhs`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn jacobi_symbol<const RHS_LIMBS: usize>(
        &self,
        rhs: &Odd<Uint<RHS_LIMBS>>,
    ) -> JacobiSymbol {
        let mut lhs = if const { LIMBS > RHS_LIMBS } {
            // Ensure `a` is reduced modulo `b` and operate on the smallest limb size
            self.rem(rhs.as_nz_ref())
        } else {
            self.resize::<RHS_LIMBS>()
        };
        let mut rhs = *rhs.as_ref();

        if const { LIMBS <= gcd::SMALL_THRESHOLD_LIMBS } {
            gcd::jacobi_symbol_small(lhs.as_mut_uint_ref(), rhs.as_mut_uint_ref())
        } else {
            gcd::jacobi_symbol(lhs.as_mut_uint_ref(), rhs.as_mut_uint_ref())
        }
    }

    /// Compute the Jacobi symbol `(self|rhs)`.
    ///
    /// For prime `rhs`, this corresponds to the Legendre symbol and
    /// indicates whether `self` is quadratic residue modulo `rhs`.
    ///
    /// This method executes in variable-time for both inputs.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn jacobi_symbol_vartime<const RHS_LIMBS: usize>(
        &self,
        rhs: &Odd<Uint<RHS_LIMBS>>,
    ) -> JacobiSymbol {
        let mut lhs = if const { LIMBS > RHS_LIMBS } {
            // Ensure `a` is reduced modulo `b` and operate on the smallest limb size
            self.rem_vartime(rhs.as_nz_ref())
        } else {
            self.resize::<RHS_LIMBS>()
        };
        let mut rhs = *rhs.as_ref();
        gcd::jacobi_symbol_vartime(lhs.as_mut_uint_ref(), rhs.as_mut_uint_ref())
    }
}

#[cfg(test)]
mod tests {
    use crate::{JacobiSymbol, U256};

    #[test]
    fn jacobi_quad_residue() {
        // Two semiprimes with no common factors, and
        // f is quadratic residue modulo g
        let f = U256::from(59u32 * 67);
        let g = U256::from(61u32 * 71).to_odd().unwrap();
        let res = f.jacobi_symbol(&g);
        let res_small = f.resize::<1>().jacobi_symbol(&g);
        let res_large = f.jacobi_symbol(&g.resize::<1>());
        let res_vartime = f.jacobi_symbol_vartime(&g);
        let res_vartime_small = f.resize::<1>().jacobi_symbol_vartime(&g);
        let res_vartime_large = f.jacobi_symbol_vartime(&g.resize::<1>());
        assert_eq!(res, JacobiSymbol::One);
        assert_eq!(res, res_small);
        assert_eq!(res, res_large);
        assert_eq!(res, res_vartime);
        assert_eq!(res, res_vartime_small);
        assert_eq!(res, res_vartime_large);
    }

    #[test]
    fn jacobi_non_quad_residue() {
        // f and g have no common factors, but
        // f is not quadratic residue modulo g
        let f = U256::from(59u32 * 67 + 2);
        let g = U256::from(61u32 * 71).to_odd().unwrap();
        let res = f.jacobi_symbol(&g);
        let res_small = f.resize::<1>().jacobi_symbol(&g);
        let res_large = f.jacobi_symbol(&g.resize::<1>());
        let res_vartime = f.jacobi_symbol_vartime(&g);
        let res_vartime_small = f.resize::<1>().jacobi_symbol_vartime(&g);
        let res_vartime_large = f.jacobi_symbol_vartime(&g.resize::<1>());
        assert_eq!(res, JacobiSymbol::MinusOne);
        assert_eq!(res, res_small);
        assert_eq!(res, res_large);
        assert_eq!(res, res_vartime);
        assert_eq!(res, res_vartime_small);
        assert_eq!(res, res_vartime_large);
    }

    #[test]
    fn jacobi_non_coprime() {
        let f = U256::from(4391633u32);
        let g = U256::from(2022161u32).to_odd().unwrap();
        let res = f.jacobi_symbol(&g);
        let res_small = f.resize::<1>().jacobi_symbol(&g);
        let res_large = f.jacobi_symbol(&g.resize::<1>());
        let res_vartime = f.jacobi_symbol_vartime(&g);
        let res_vartime_small = f.resize::<1>().jacobi_symbol_vartime(&g);
        let res_vartime_large = f.jacobi_symbol_vartime(&g.resize::<1>());
        assert_eq!(res, JacobiSymbol::Zero);
        assert_eq!(res, res_small);
        assert_eq!(res, res_large);
        assert_eq!(res, res_vartime);
        assert_eq!(res, res_vartime_small);
        assert_eq!(res, res_vartime_large);
    }

    #[test]
    fn jacobi_zero() {
        let f = U256::ZERO;
        let g = U256::ONE.to_odd().unwrap();
        let res = f.jacobi_symbol(&g);
        let res_small = f.resize::<1>().jacobi_symbol(&g);
        let res_large = f.jacobi_symbol(&g.resize::<1>());
        let res_vartime = f.jacobi_symbol_vartime(&g);
        let res_vartime_small = f.resize::<1>().jacobi_symbol_vartime(&g);
        let res_vartime_large = f.jacobi_symbol_vartime(&g.resize::<1>());
        assert_eq!(res, JacobiSymbol::One);
        assert_eq!(res, res_small);
        assert_eq!(res, res_large);
        assert_eq!(res, res_vartime);
        assert_eq!(res, res_vartime_small);
        assert_eq!(res, res_vartime_large);
    }

    #[test]
    fn jacobi_one() {
        let f = U256::ONE;
        let g = U256::ONE.to_odd().unwrap();
        let res = f.jacobi_symbol(&g);
        let res_small = f.resize::<1>().jacobi_symbol(&g);
        let res_large = f.jacobi_symbol(&g.resize::<1>());
        let res_vartime = f.jacobi_symbol_vartime(&g);
        let res_vartime_small = f.resize::<1>().jacobi_symbol_vartime(&g);
        let res_vartime_large = f.jacobi_symbol_vartime(&g.resize::<1>());
        assert_eq!(res, JacobiSymbol::One);
        assert_eq!(res, res_small);
        assert_eq!(res, res_large);
        assert_eq!(res, res_vartime);
        assert_eq!(res, res_vartime_small);
        assert_eq!(res, res_vartime_large);
    }

    #[test]
    // test from issue #1295 - variations in only the middle bits can trip up optimized binary GCD method
    fn jacobi_edge() {
        use crate::{Odd, U256};

        assert_eq!(
            U256::from_be_hex("0000000000000002108DEAFCB180F023912BEED0186CEEAD593A8507B7DA4E9B")
                .jacobi_symbol(&Odd::<U256>::from_be_hex(
                    "0000000000000002108DEAFCB180F023912BEED0186CEEED593A8507B7DA4E9B",
                )),
            JacobiSymbol::MinusOne
        );
    }
}
