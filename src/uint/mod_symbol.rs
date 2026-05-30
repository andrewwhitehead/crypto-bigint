//! Support for computing modular symbols.

use crate::{JacobiSymbol, Odd, Uint};

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
        if LIMBS > RHS_LIMBS {
            // Ensure `a` is reduced modulo `b` and operate on the smallest limb size
            let a = self.rem(rhs.as_nz_ref());
            return a.jacobi_symbol(rhs);
        }

        self.resize::<RHS_LIMBS>().bingcd_jacobi_symbol(rhs)
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
        self.bingcd_jacobi_symbol_vartime(rhs)
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
}
