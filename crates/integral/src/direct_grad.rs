//! Density-direct contraction of two-electron-integral nuclear derivatives.

use crate::engine::os::Prim;
use crate::engine::os_eri::{self, EriScratch, ShellPairData, ShellRef};
use crate::engine::rys;
use crate::grad::MAX_GRAD_L;
use crate::integrals::{
    canonical_shell_pairs, check_erf_omega, effective_coeffs, select_engine, Engine, EriKernel,
};
use crate::math::am::{cart_components, cart_index, n_cart};
use crate::shell::{Basis, Shell};
use crate::spherical::{shell_transform, transform_block4_into};
use crate::IntegralError;

/// One product term of a two-particle density: `factor · left ⊗ right`.
///
/// `left` and `right` are symmetric `nao × nao` row-major matrices. Whether the
/// product is Coulomb-type (`left[μν]·right[λσ]`) or exchange-type
/// (`left[μλ]·right[νσ]`) is decided by the list it is passed in to
/// [`DirectGradContractor::density`].
#[derive(Debug, Clone, Copy)]
pub struct GradDensityTerm<'a> {
    pub factor: f64,
    pub left: &'a [f64],
    pub right: &'a [f64],
}

#[derive(Debug)]
struct BoundedTerm<'a> {
    factor: f64,
    left: &'a [f64],
    right: &'a [f64],
    left_max: Vec<f64>,
    right_max: Vec<f64>,
}

/// A two-particle density prepared for [`DirectGradContractor::accumulate_bra_pair`]:
/// the product terms plus their per-shell-block magnitudes used for screening.
///
/// The magnitudes are partitioned by the shells of the plan that prepared the
/// density, so it can only be contracted by a plan with the same shell layout.
#[derive(Debug)]
pub struct GradDensity<'a> {
    coulomb: Vec<BoundedTerm<'a>>,
    exchange: Vec<BoundedTerm<'a>>,
    /// First-function offset of every shell of the preparing plan.
    offsets: Vec<usize>,
    nao: usize,
}

/// Reusable worker-local scratch. Construct one per worker and reuse it across
/// every bra shell pair assigned to that worker.
#[derive(Default)]
pub struct DirectGradWorkspace {
    eri: EriScratch,
    weight: Vec<f64>,
    tmp: Vec<f64>,
    raised: Vec<f64>,
    lowered: Vec<f64>,
}

impl DirectGradWorkspace {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Where the Cartesian component `a` of a shell lands after `a ± 1_axis`.
#[derive(Debug)]
struct ShiftTable {
    up: Vec<[usize; 3]>,
    /// `(index in shell l − 1, a_axis)`; `a_axis == 0.0` marks "no lowered term".
    down: Vec<[(usize, f64); 3]>,
}

impl ShiftTable {
    fn new(l: usize) -> Self {
        let comps = cart_components(l);
        let up = comps
            .iter()
            .map(|c| {
                std::array::from_fn(|axis| {
                    let mut r = *c;
                    r[axis] += 1;
                    cart_index(r)
                })
            })
            .collect();
        let down = comps
            .iter()
            .map(|c| {
                std::array::from_fn(|axis| {
                    if c[axis] == 0 {
                        (0, 0.0)
                    } else {
                        let mut r = *c;
                        r[axis] -= 1;
                        (cart_index(r), c[axis] as f64)
                    }
                })
            })
            .collect();
        Self { up, down }
    }
}

/// An owned plan for density-direct contraction of two-electron-integral
/// nuclear derivatives. It evaluates
///
/// ```text
///   F_c = Σ_{μνλσ} Γ_{μνλσ} · ∂(μν|λσ)/∂R_c        (one [x, y, z] per atom)
/// ```
///
/// for a two-particle density that is a sum of products of one-particle
/// matrices,
///
/// ```text
///   Γ_{μνλσ} = Σ_t f_t · L_t[μν] · R_t[λσ]          (Coulomb-type terms)
///            + Σ_t f_t · L_t[μλ] · R_t[νσ]          (exchange-type terms),
/// ```
///
/// without ever forming `Γ`. Shell metadata, effective and `2α`-weighted
/// coefficients, transforms, primitive-pair lists and Schwarz bounds are
/// computed once when the plan is created. It is the gradient counterpart of
/// [`DirectContractor`](crate::DirectContractor) and returns the same numbers
/// as [`Basis::eri_grad_contract`] applied to the dense `Γ` (same sign: the
/// derivative, not the force; same atom order: [`Basis::atoms`]).
///
/// Only the canonical shell quartets `(ij|kl)`, `i ≥ j`, `k ≥ l`, `(kl) ≤ (ij)`
/// are evaluated; each carries its permutational degeneracy and is contracted
/// with the symmetrized two-particle density of its whole function block.
/// Three centers are differentiated per quartet and the fourth follows from
/// translational invariance. Quartets are skipped by a density-weighted
/// Schwarz estimate, or when all four shells share one atom.
///
/// Every matrix in a term must be symmetric: the permutational reduction
/// relies on it.
///
/// The library stays single-threaded. One bra shell pair is the unit of work,
/// so a caller parallelizes by giving each worker a [`DirectGradWorkspace`] and
/// a private `[[f64; 3]]` accumulator and summing the accumulators.
#[derive(Debug)]
pub struct DirectGradContractor {
    shells: Vec<Shell>,
    engine: Engine,
    offsets: Vec<usize>,
    nfunc: Vec<usize>,
    nao: usize,
    shell_atom: Vec<usize>,
    natom: usize,
    eff: Vec<Vec<f64>>,
    eff_2a: Vec<Vec<f64>>,
    /// Transposed Cartesian→function transforms (`n_cart × n_func`), `None` for
    /// Cartesian shells: they pull a function-space weight block back to
    /// Cartesian components.
    s2c: Vec<Option<Vec<f64>>>,
    pairs: Vec<(usize, usize)>,
    pair_plain: Vec<ShellPairData>,
    /// Pair lists with `2α` folded into the first / second shell of the pair.
    pair_first_2a: Vec<ShellPairData>,
    pair_second_2a: Vec<ShellPairData>,
    schwarz: Vec<f64>,
    shifts: Vec<ShiftTable>,
}

impl DirectGradContractor {
    /// # Errors
    /// [`IntegralError::AngularMomentumTooHighForGradient`] if any shell has
    /// `l > MAX_GRAD_L`.
    pub fn new(basis: &Basis) -> Result<Self, IntegralError> {
        Self::with_engine(basis, Engine::Auto)
    }

    /// Like [`Self::new`] but forces a specific [`Engine`] for the Coulomb
    /// kernel. The `Erf` kernel always runs on the Rys engine.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn with_engine(basis: &Basis, engine: Engine) -> Result<Self, IntegralError> {
        if let Some(s) = basis.shells().iter().find(|s| s.l() > MAX_GRAD_L) {
            return Err(IntegralError::AngularMomentumTooHighForGradient {
                l: s.l(),
                max: MAX_GRAD_L,
            });
        }
        let shells = basis.shells().to_vec();
        let eff: Vec<Vec<f64>> = shells.iter().map(effective_coeffs).collect();
        let eff_2a: Vec<Vec<f64>> = shells
            .iter()
            .zip(&eff)
            .map(|(s, c)| {
                c.iter()
                    .zip(s.exponents())
                    .map(|(c, a)| 2.0 * a * c)
                    .collect()
            })
            .collect();
        let s2c = shells
            .iter()
            .map(|s| {
                shell_transform(s).map(|m| {
                    let (nf, nc) = (s.n_func(), s.n_cart());
                    let mut t = vec![0.0; nc * nf];
                    for f in 0..nf {
                        for c in 0..nc {
                            t[c * nf + f] = m[f * nc + c];
                        }
                    }
                    t
                })
            })
            .collect();
        let pairs = canonical_shell_pairs(shells.len());
        let pair_list = |first: &[Vec<f64>], second: &[Vec<f64>]| -> Vec<ShellPairData> {
            pairs
                .iter()
                .map(|&(i, j)| {
                    os_eri::shell_pair_data(
                        shell_ref(&shells[i], shells[i].l(), &first[i]),
                        shell_ref(&shells[j], shells[j].l(), &second[j]),
                    )
                })
                .collect()
        };
        let pair_plain = pair_list(&eff, &eff);
        let pair_first_2a = pair_list(&eff_2a, &eff);
        let pair_second_2a = pair_list(&eff, &eff_2a);
        Ok(Self {
            engine,
            offsets: basis.offsets(),
            nfunc: shells.iter().map(Shell::n_func).collect(),
            nao: basis.nao(),
            shell_atom: basis.shell_atom(),
            natom: basis.atoms().len(),
            eff,
            eff_2a,
            s2c,
            pairs,
            pair_plain,
            pair_first_2a,
            pair_second_2a,
            schwarz: basis.schwarz_bounds(),
            shifts: (0..=MAX_GRAD_L).map(ShiftTable::new).collect(),
            shells,
        })
    }

    #[must_use]
    pub fn nao(&self) -> usize {
        self.nao
    }

    /// Number of atoms, the required length of the output accumulator; atoms
    /// are ordered as in [`Basis::atoms`].
    #[must_use]
    pub fn natom(&self) -> usize {
        self.natom
    }

    #[must_use]
    pub fn bra_pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// Prepare a two-particle density from its Coulomb-type and exchange-type
    /// product terms (see [`DirectGradContractor`] for the definition).
    ///
    /// # Panics
    /// Panics if a matrix is not `nao × nao`.
    #[must_use]
    pub fn density<'a>(
        &self,
        coulomb: &[GradDensityTerm<'a>],
        exchange: &[GradDensityTerm<'a>],
    ) -> GradDensity<'a> {
        let bound = |t: &GradDensityTerm<'a>| BoundedTerm {
            factor: t.factor,
            left: t.left,
            right: t.right,
            left_max: self.shell_block_max(t.left),
            right_max: self.shell_block_max(t.right),
        };
        GradDensity {
            coulomb: coulomb.iter().map(bound).collect(),
            exchange: exchange.iter().map(bound).collect(),
            offsets: self.offsets.clone(),
            nao: self.nao,
        }
    }

    fn shell_block_max(&self, matrix: &[f64]) -> Vec<f64> {
        assert_eq!(matrix.len(), self.nao * self.nao);
        let nsh = self.shells.len();
        let mut bounds = vec![0.0_f64; nsh * nsh];
        for i in 0..nsh {
            for j in 0..nsh {
                let mut max = 0.0_f64;
                for a in 0..self.nfunc[i] {
                    let row = (self.offsets[i] + a) * self.nao + self.offsets[j];
                    for b in 0..self.nfunc[j] {
                        max = max.max(matrix[row + b].abs());
                    }
                }
                let (lo, hi) = (i.min(j), i.max(j));
                let sym = bounds[hi * nsh + lo].max(max);
                bounds[hi * nsh + lo] = sym;
                bounds[lo * nsh + hi] = sym;
            }
        }
        bounds
    }

    /// Add the contribution of every canonical ket pair of one canonical bra
    /// pair to `out` (length [`Self::natom`]). A quartet is skipped when
    /// `Q_ij · Q_kl`, or that product times the magnitude of its two-particle
    /// density block, is below `threshold`; `threshold = 0.0` keeps everything,
    /// including pairs whose Schwarz factor is exactly zero.
    ///
    /// The Schwarz factors bound the integrals rather than their derivatives,
    /// so the test is an estimate, as in direct SCF.
    ///
    /// # Panics
    /// Panics if `out.len() != natom`, if `bra` is not a pair from
    /// [`Self::bra_pairs`], if `density` was prepared by a plan with a
    /// different shell layout, or if `kernel` is `Erf` with a non-finite or
    /// non-positive `ω`.
    pub fn accumulate_bra_pair(
        &self,
        workspace: &mut DirectGradWorkspace,
        bra: (usize, usize),
        density: &GradDensity<'_>,
        kernel: EriKernel,
        threshold: f64,
        out: &mut [[f64; 3]],
    ) {
        assert_eq!(out.len(), self.natom);
        if let EriKernel::Erf { omega } = kernel {
            check_erf_omega(omega);
        }
        let (i, j) = bra;
        assert!(j <= i && i < self.shells.len());
        assert!(
            density.nao == self.nao && density.offsets == self.offsets,
            "density was prepared by a plan with a different shell layout"
        );
        let nsh = self.shells.len();
        let qab = self.schwarz[i * nsh + j];
        if qab == 0.0 && threshold > 0.0 {
            return;
        }
        let atoms = &self.shell_atom;
        for k in 0..=i {
            let l_top = if k == i { j } else { k };
            for l in 0..=l_top {
                if atoms[i] == atoms[j] && atoms[j] == atoms[k] && atoms[k] == atoms[l] {
                    continue;
                }
                let pair_bound = qab * self.schwarz[k * nsh + l];
                if pair_bound < threshold {
                    continue;
                }
                let degeneracy = (if i == j { 1.0 } else { 2.0 })
                    * (if k == l { 1.0 } else { 2.0 })
                    * (if i == k && j == l { 1.0 } else { 2.0 });
                let idx = [i, j, k, l];
                if pair_bound * degeneracy * density.block_bound(nsh, idx) < threshold {
                    continue;
                }
                self.quartet(workspace, idx, degeneracy, density, kernel, out);
            }
        }
    }

    /// Serial convenience: the full contraction over all bra pairs.
    ///
    /// # Panics
    /// As [`Self::accumulate_bra_pair`].
    #[must_use]
    pub fn contract(
        &self,
        density: &GradDensity<'_>,
        kernel: EriKernel,
        threshold: f64,
    ) -> Vec<[f64; 3]> {
        let mut out = vec![[0.0; 3]; self.natom];
        let mut workspace = DirectGradWorkspace::new();
        for &bra in &self.pairs {
            self.accumulate_bra_pair(&mut workspace, bra, density, kernel, threshold, &mut out);
        }
        out
    }

    fn quartet(
        &self,
        workspace: &mut DirectGradWorkspace,
        idx: [usize; 4],
        degeneracy: f64,
        density: &GradDensity<'_>,
        kernel: EriKernel,
        out: &mut [[f64; 3]],
    ) {
        self.weight_block(workspace, idx, degeneracy, density);
        let ls = idx.map(|s| self.shells[s].l());
        let dims = ls.map(n_cart);
        let rys_omega = match kernel {
            EriKernel::Erf { omega } => Some(Some(omega)),
            EriKernel::Coulomb => {
                let engine = match self.engine {
                    Engine::Auto => select_engine(
                        ls[0] + ls[1],
                        ls[2] + ls[3],
                        idx.iter().map(|&s| self.shells[s].n_prim()).product(),
                    ),
                    forced => forced,
                };
                (engine == Engine::Rys).then_some(None)
            }
        };

        let mut fourth = [0.0_f64; 3];
        for pos in 0..3 {
            let l = ls[pos];
            let mut up_dims = dims;
            up_dims[pos] = n_cart(l + 1);
            let up_len: usize = up_dims.iter().product();
            workspace.raised.clear();
            workspace.raised.resize(up_len, 0.0);
            self.shifted_block(
                &mut workspace.eri,
                idx,
                pos,
                l + 1,
                true,
                rys_omega,
                &mut workspace.raised,
            );
            if l > 0 {
                let mut down_dims = dims;
                down_dims[pos] = n_cart(l - 1);
                let down_len: usize = down_dims.iter().product();
                workspace.lowered.clear();
                workspace.lowered.resize(down_len, 0.0);
                self.shifted_block(
                    &mut workspace.eri,
                    idx,
                    pos,
                    l - 1,
                    false,
                    rys_omega,
                    &mut workspace.lowered,
                );
            }

            let outer: usize = dims[..pos].iter().product();
            let inner: usize = dims[pos + 1..].iter().product();
            let (n, n_up) = (dims[pos], up_dims[pos]);
            let n_down = if l > 0 { n_cart(l - 1) } else { 0 };
            let table = &self.shifts[l];
            let mut g = [0.0_f64; 3];
            for o in 0..outer {
                for a in 0..n {
                    let w = &workspace.weight[(o * n + a) * inner..][..inner];
                    for (axis, g_axis) in g.iter_mut().enumerate() {
                        let up = (o * n_up + table.up[a][axis]) * inner;
                        *g_axis += dot(w, &workspace.raised[up..up + inner]);
                        let (down_a, count) = table.down[a][axis];
                        if count != 0.0 {
                            let down = (o * n_down + down_a) * inner;
                            *g_axis -= count * dot(w, &workspace.lowered[down..down + inner]);
                        }
                    }
                }
            }
            let atom = self.shell_atom[idx[pos]];
            for axis in 0..3 {
                out[atom][axis] += g[axis];
                fourth[axis] -= g[axis];
            }
        }
        let atom = self.shell_atom[idx[3]];
        for axis in 0..3 {
            out[atom][axis] += fourth[axis];
        }
    }

    /// `degeneracy ×` the permutation-averaged two-particle density of the
    /// quartet's function block, pulled back to Cartesian components, left in
    /// `workspace.weight`.
    fn weight_block(
        &self,
        workspace: &mut DirectGradWorkspace,
        idx: [usize; 4],
        degeneracy: f64,
        density: &GradDensity<'_>,
    ) {
        let n = self.nao;
        let [oi, oj, ok, ol] = idx.map(|s| self.offsets[s]);
        let [ni, nj, nk, nl] = idx.map(|s| self.nfunc[s]);
        let w = &mut workspace.weight;
        w.clear();
        w.resize(ni * nj * nk * nl, 0.0);

        for term in &density.coulomb {
            let (left, right) = (term.left, term.right);
            let f = 0.5 * degeneracy * term.factor;
            for a in 0..ni {
                for b in 0..nj {
                    let mn = (oi + a) * n + oj + b;
                    let (l_mn, r_mn) = (f * left[mn], f * right[mn]);
                    let row = &mut w[(a * nj + b) * nk * nl..][..nk * nl];
                    for c in 0..nk {
                        let ls = (ok + c) * n + ol;
                        for d in 0..nl {
                            row[c * nl + d] += l_mn * right[ls + d] + r_mn * left[ls + d];
                        }
                    }
                }
            }
        }
        for term in &density.exchange {
            let (left, right) = (term.left, term.right);
            let f = 0.25 * degeneracy * term.factor;
            for a in 0..ni {
                let mu = (oi + a) * n;
                for b in 0..nj {
                    let nu = (oj + b) * n;
                    let row = &mut w[(a * nj + b) * nk * nl..][..nk * nl];
                    for c in 0..nk {
                        let lam = ok + c;
                        let (l_ml, r_ml) = (f * left[mu + lam], f * right[mu + lam]);
                        let (l_nl, r_nl) = (f * left[nu + lam], f * right[nu + lam]);
                        for d in 0..nl {
                            let sig = ol + d;
                            row[c * nl + d] += l_ml * right[nu + sig]
                                + l_nl * right[mu + sig]
                                + left[mu + sig] * r_nl
                                + left[nu + sig] * r_ml;
                        }
                    }
                }
            }
        }

        let mats = idx.map(|s| self.s2c[s].as_deref());
        transform_block4_into(w, [ni, nj, nk, nl], &mats, &mut workspace.tmp);
    }

    /// Accumulate into `out` the Cartesian block of the quartet with shell
    /// `pos` at angular momentum `lp`, with `2α` folded into that shell's
    /// primitives when `weighted`. `rys_omega`: `None` → OS/HGP, `Some(None)` →
    /// Rys Coulomb, `Some(Some(ω))` → Rys `erf(ω·r₁₂)/r₁₂`.
    #[allow(clippy::too_many_arguments)]
    fn shifted_block(
        &self,
        eri: &mut EriScratch,
        idx: [usize; 4],
        pos: usize,
        lp: usize,
        weighted: bool,
        rys_omega: Option<Option<f64>>,
        out: &mut [f64],
    ) {
        let l_at = |p: usize| {
            if p == pos {
                lp
            } else {
                self.shells[idx[p]].l()
            }
        };
        let coeffs_at = |p: usize| -> &[f64] {
            if p == pos && weighted {
                &self.eff_2a[idx[p]]
            } else {
                &self.eff[idx[p]]
            }
        };
        let Some(attenuation) = rys_omega else {
            let make = |p: usize| shell_ref(&self.shells[idx[p]], l_at(p), coeffs_at(p));
            let bra = tri(idx[0], idx[1]);
            let ket = tri(idx[2], idx[3]);
            let bra_pairs = match (weighted, pos) {
                (true, 0) => &self.pair_first_2a[bra],
                (true, 1) => &self.pair_second_2a[bra],
                _ => &self.pair_plain[bra],
            };
            let ket_pairs = match (weighted, pos) {
                (true, 2) => &self.pair_first_2a[ket],
                _ => &self.pair_plain[ket],
            };
            os_eri::coulomb_shell_pairs_into_scratch(
                eri,
                make(0),
                make(1),
                make(2),
                make(3),
                bra_pairs,
                ket_pairs,
                out,
            );
            return;
        };

        let s = idx.map(|i| &self.shells[i]);
        let prim = |p: usize, i: usize| Prim::new(s[p].exponents()[i], s[p].center(), l_at(p));
        for (pa, ca) in coeffs_at(0).iter().enumerate() {
            for (pb, cb) in coeffs_at(1).iter().enumerate() {
                for (pc, cc) in coeffs_at(2).iter().enumerate() {
                    for (pd, cd) in coeffs_at(3).iter().enumerate() {
                        let scale = ca * cb * cc * cd;
                        let (a, b, c, d) = (prim(0, pa), prim(1, pb), prim(2, pc), prim(3, pd));
                        match attenuation {
                            None => rys::coulomb_into(a, b, c, d, scale, out),
                            Some(omega) => rys::erf_coulomb_into(a, b, c, d, omega, scale, out),
                        }
                    }
                }
            }
        }
    }
}

impl GradDensity<'_> {
    /// Upper bound on the permutation-averaged two-particle density over the
    /// function block of shell quartet `idx`.
    fn block_bound(&self, nsh: usize, [i, j, k, l]: [usize; 4]) -> f64 {
        let at = |m: &[f64], a: usize, b: usize| m[a * nsh + b];
        let mut bound = 0.0;
        for t in &self.coulomb {
            bound += 0.5
                * t.factor.abs()
                * (at(&t.left_max, i, j) * at(&t.right_max, k, l)
                    + at(&t.left_max, k, l) * at(&t.right_max, i, j));
        }
        for t in &self.exchange {
            bound += 0.25
                * t.factor.abs()
                * (at(&t.left_max, i, k) * at(&t.right_max, j, l)
                    + at(&t.left_max, j, k) * at(&t.right_max, i, l)
                    + at(&t.left_max, i, l) * at(&t.right_max, j, k)
                    + at(&t.left_max, j, l) * at(&t.right_max, i, k));
        }
        bound
    }
}

impl Basis {
    /// # Errors
    /// As [`DirectGradContractor::new`].
    pub fn direct_grad_contractor(&self) -> Result<DirectGradContractor, IntegralError> {
        DirectGradContractor::new(self)
    }
}

fn shell_ref<'a>(shell: &'a Shell, l: usize, coeffs: &'a [f64]) -> ShellRef<'a> {
    ShellRef {
        center: shell.center(),
        l,
        exps: shell.exponents(),
        coeffs,
    }
}

fn tri(i: usize, j: usize) -> usize {
    debug_assert!(i >= j);
    i * (i + 1) / 2 + j
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
