//! Jacobi symbol calculation for integers in Montgomery form with a constant modulus.

use super::{ConstMontyForm, ConstMontyParams};
use crate::{JacobiMod, JacobiSymbol, Odd, PrecomputeInverter, Uint, modular::SafeGcdInverter};

impl<MOD: ConstMontyParams<SAT_LIMBS>, const SAT_LIMBS: usize, const UNSAT_LIMBS: usize>
    ConstMontyForm<MOD, SAT_LIMBS>
where
    Odd<Uint<SAT_LIMBS>>: PrecomputeInverter<
            Inverter = SafeGcdInverter<SAT_LIMBS, UNSAT_LIMBS>,
            Output = Uint<SAT_LIMBS>,
        >,
{
    /// Compute the Jacobi symbol `(self|modulus)`. For a prime modulus, this
    /// corresponds to the Legendre symbol.
    pub const fn jacobi(&self) -> JacobiSymbol {
        let inner = self.retrieve();
        <Odd<Uint<SAT_LIMBS>> as PrecomputeInverter>::Inverter::jacobi_uint_vartime(
            &inner,
            &MOD::MODULUS,
        )
    }
}

impl<MOD: ConstMontyParams<SAT_LIMBS>, const SAT_LIMBS: usize, const UNSAT_LIMBS: usize> JacobiMod
    for ConstMontyForm<MOD, SAT_LIMBS>
where
    Odd<Uint<SAT_LIMBS>>: PrecomputeInverter<
            Inverter = SafeGcdInverter<SAT_LIMBS, UNSAT_LIMBS>,
            Output = Uint<SAT_LIMBS>,
        >,
{
    fn jacobi_mod(&self) -> JacobiSymbol {
        Self::jacobi(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        JacobiSymbol, U256, const_monty_form, const_monty_params,
        modular::const_monty_form::ConstMontyParams,
    };

    const_monty_params!(
        Modulus,
        U256,
        "2523648240000001BA344D80000000086121000000000013A700000000000013"
    );

    #[test]
    fn jacobi_quad_residue() {
        let x =
            U256::from_be_hex("14BFAE46F4026E97C7A3FCD889B379A5F025719911C994A594FC6C5092AC58B1");
        let x_mod = const_monty_form!(x, Modulus);

        let jac = x_mod.jacobi();
        assert_eq!(jac, JacobiSymbol::One);
    }

    #[test]
    fn jacobi_quad_nonresidue() {
        let x =
            U256::from_be_hex("1D2EFB21D283A2DDE77004B9DE9A9624F7B15CEEF055CD02E9EF1A9F1B76F253");
        let x_mod = const_monty_form!(x, Modulus);

        let jac = x_mod.jacobi();
        assert_eq!(jac, JacobiSymbol::MinusOne);
    }
}
