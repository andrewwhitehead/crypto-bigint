use crate::{Choice, Limb, UintRef, WideWord, Word, word};

const HI_MASK: WideWord = WideWord::MAX << Word::BITS;

/// Reduce a pair of [`UintRef`] to a pair of [`WideWord`] representing the concatenated
/// result of a high word and a low word, where the high word ends at bit `n`.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
pub const fn compact_pair(a: &UintRef, b: &UintRef, n: u32) -> (WideWord, WideWord) {
    if a.nlimbs() < 2 {
        return (a.to_wide_word_unchecked(), b.to_wide_word_unchecked());
    }
    let top = (n - 1) >> Limb::LOG2_BITS;
    let shift = (top + 1) * Limb::BITS - n;
    let (mut a_hi, mut b_hi) = ((0, 0), (0, 0));
    let mut i = 1;
    while i < a.limbs.len() {
        let found = Choice::from_u32_eq(i as u32, top);
        a_hi = (
            word::select(a_hi.0, a.limbs[i - 1].0, found),
            word::select(a_hi.1, a.limbs[i].0, found),
        );
        b_hi = (
            word::select(b_hi.0, b.limbs[i - 1].0, found),
            word::select(b_hi.1, b.limbs[i].0, found),
        );
        i += 1;
    }
    let mut c1 = word::join(a_hi.0, a_hi.1);
    let mut c2 = word::join(b_hi.0, b_hi.1);
    c1 = ((c1 << shift) & HI_MASK) | (a.limbs[0].0 as WideWord);
    c2 = ((c2 << shift) & HI_MASK) | (b.limbs[0].0 as WideWord);
    (c1, c2)
}

#[inline(always)]
pub const fn compact_pair_vartime(a: &UintRef, b: &UintRef, n: u32) -> (WideWord, WideWord) {
    if a.nlimbs() < 2 {
        return (a.to_wide_word_unchecked(), b.to_wide_word_unchecked());
    }
    let top = (n - 1) >> Limb::LOG2_BITS;
    let shift = (top + 1) * Limb::BITS - n;
    let top = top as usize;
    let mut c1 = word::join(a.limbs[top - 1].0, a.limbs[top].0);
    let mut c2 = word::join(b.limbs[top - 1].0, b.limbs[top].0);
    c1 = ((c1 << shift) & HI_MASK) | (a.limbs[0].0 as WideWord);
    c2 = ((c2 << shift) & HI_MASK) | (b.limbs[0].0 as WideWord);
    (c1, c2)
}

#[cfg(test)]
mod tests {
    use super::{compact_pair, compact_pair_vartime};
    use crate::{U256, Uint, WideWord, Word};

    #[allow(clippy::cast_possible_truncation)]
    fn check_compact<const LIMBS: usize>(val: &Uint<LIMBS>, compact: WideWord, n: u32) {
        let lo = val.limbs[0].0;
        let hi = val
            .unbounded_shr(n.saturating_sub(Word::BITS))
            .restrict_bits(Word::BITS)
            .as_uint_ref()
            .to_wide_word_unchecked();
        assert_eq!(lo, compact as Word);
        assert_eq!(hi, compact >> Word::BITS);
    }

    #[test]
    fn test_compact() {
        let a =
            U256::from_be_hex("1971BC6285D8CBA9640AA3B3B9C01EF4186D1EBE9A17393A9E43586E0EBAED5B");
        let b =
            U256::from_be_hex("CFCF1535CEBE19BBF289933AB8645189397450A32BFEC57579FB7EB14E27D101");

        let n = 200;
        let (ca, cb) = compact_pair(a.as_uint_ref(), b.as_uint_ref(), n);
        check_compact(&a, ca, n);
        check_compact(&b, cb, n);
    }

    #[test]
    fn test_compact_vartime() {
        let a =
            U256::from_be_hex("CFCF1535CEBE19BBF289933AB8645189397450A32BFEC57579FB7EB14E27D101");
        let b =
            U256::from_be_hex("1971BC6285D8CBA9640AA3B3B9C01EF4186D1EBE9A17393A9E43586E0EBAED5B");

        let n = 200;
        let (ca, cb) = compact_pair_vartime(a.as_uint_ref(), b.as_uint_ref(), n);
        check_compact(&a, ca, n);
        check_compact(&b, cb, n);
    }
}
