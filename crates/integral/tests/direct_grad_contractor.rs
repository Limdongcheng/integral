//! `DirectGradContractor` validation — no external reference.
//!
//! The density-direct contraction over canonical quartets is checked four ways:
//!
//! 1. **Dense reference.** The same product-form two-particle density is
//!    expanded to an explicit `nao⁴` tensor and contracted by
//!    `eri_grad_contract` over all ordered quartets. Covered: Cartesian,
//!    spherical and mixed shells, every `l ≤ MAX_GRAD_L` at each shell index of
//!    a three-centre basis (so it reaches every quartet position), shells
//!    sharing a centre, both forced engines and the `Auto` dispatch on a basis
//!    where it picks both, and the `erf` kernel.
//! 2. **Finite difference** of the scalar `Σ Γ·(μν|λσ)` built from `Basis::eri`
//!    at displaced geometries, which involves no gradient code at all.
//! 3. **Exact invariants**: zero net force, an exactly zero result when every
//!    shell sits on one atom, additivity over the two term lists, and
//!    independence of the bra-pair order.
//! 4. **Screening**: threshold `0` reproduces the dense reference on a spatially
//!    extended system and a finite threshold stays within a stated bound.
//!
//! The `l = 5` cases are `#[ignore]`d (run in release with `--include-ignored`).

use integral::{
    select_engine, Basis, DirectGradContractor, DirectGradWorkspace, Engine, EriKernel,
    GradDensityTerm, Shell, ShellKind, MAX_GRAD_L,
};

const TOL: f64 = 1e-10;
/// Central difference at `h = 2e-4` reaches ~`1e-8` here; `1e-6` leaves margin.
const FD_TOL: f64 = 1e-6;
const H: f64 = 2e-4;

fn basis(kind: ShellKind) -> Basis {
    let a = [0.0, 0.0, 0.0];
    let b = [0.9, -0.4, 0.6];
    let c = [-0.7, 1.1, -0.3];
    Basis::new(vec![
        Shell::with_kind(0, a, vec![1.9, 0.45], vec![0.6, 0.5], kind).unwrap(),
        Shell::with_kind(1, a, vec![0.8, 0.3], vec![0.7, 0.4], kind).unwrap(),
        Shell::with_kind(2, b, vec![0.7], vec![1.0], kind).unwrap(),
        Shell::with_kind(0, b, vec![0.5], vec![1.0], kind).unwrap(),
        Shell::with_kind(3, c, vec![0.6], vec![1.0], kind).unwrap(),
        Shell::with_kind(1, c, vec![1.1, 0.4], vec![0.3, 0.8], kind).unwrap(),
    ])
}

fn symmetric(n: usize, seed: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for p in 0..n {
        for q in 0..=p {
            let v = (((p * 7 + q * 13 + seed * 5) % 23) as f64 - 11.0) * 0.017;
            m[p * n + q] = v;
            m[q * n + p] = v;
        }
    }
    m
}

/// Three distinct symmetric matrices combined into two Coulomb-type and two
/// exchange-type terms, with `left ≠ right` in one term of each list.
struct Densities {
    p: Vec<f64>,
    da: Vec<f64>,
    db: Vec<f64>,
}

impl Densities {
    fn new(n: usize) -> Self {
        Self {
            p: symmetric(n, 1),
            da: symmetric(n, 2),
            db: symmetric(n, 3),
        }
    }

    fn coulomb(&self) -> [GradDensityTerm<'_>; 2] {
        [
            GradDensityTerm {
                factor: 0.5,
                left: &self.p,
                right: &self.p,
            },
            GradDensityTerm {
                factor: 0.3,
                left: &self.da,
                right: &self.db,
            },
        ]
    }

    fn exchange(&self) -> [GradDensityTerm<'_>; 2] {
        [
            GradDensityTerm {
                factor: -0.5,
                left: &self.da,
                right: &self.da,
            },
            GradDensityTerm {
                factor: -0.2,
                left: &self.db,
                right: &self.p,
            },
        ]
    }
}

fn dense_gamma(
    n: usize,
    coulomb: &[GradDensityTerm<'_>],
    exchange: &[GradDensityTerm<'_>],
) -> Vec<f64> {
    let mut gamma = vec![0.0; n * n * n * n];
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    let mut v = 0.0;
                    for t in coulomb {
                        v += t.factor * t.left[p * n + q] * t.right[r * n + s];
                    }
                    for t in exchange {
                        v += t.factor * t.left[p * n + r] * t.right[q * n + s];
                    }
                    gamma[((p * n + q) * n + r) * n + s] = v;
                }
            }
        }
    }
    gamma
}

fn max_diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(u, v)| (u - v).abs()))
        .fold(0.0, f64::max)
}

fn max_abs(a: &[[f64; 3]]) -> f64 {
    a.iter().flatten().fold(0.0, |m, v| m.max(v.abs()))
}

fn dense_reference(
    basis: &Basis,
    engine: Engine,
    kernel: EriKernel,
    gamma: &[f64],
) -> Vec<[f64; 3]> {
    match kernel {
        EriKernel::Coulomb => basis.eri_grad_contract_with(engine, gamma).unwrap(),
        erf => basis.eri_grad_contract_kernel(gamma, erf).unwrap(),
    }
}

/// Contract the [`Densities`] of `basis` both ways and require agreement to
/// [`TOL`] plus a vanishing net force. Returns the direct result.
fn check_basis(basis: &Basis, engine: Engine, kernel: EriKernel, label: &str) -> Vec<[f64; 3]> {
    let n = basis.nao();
    let d = Densities::new(n);
    let (coulomb, exchange) = (d.coulomb(), d.exchange());
    let gamma = dense_gamma(n, &coulomb, &exchange);
    let reference = dense_reference(basis, engine, kernel, &gamma);

    let contractor = DirectGradContractor::with_engine(basis, engine).unwrap();
    assert_eq!(contractor.nao(), n);
    assert_eq!(contractor.natom(), basis.atoms().len());
    let density = contractor.density(&coulomb, &exchange);
    let direct = contractor.contract(&density, kernel, 0.0);
    let diff = max_diff(&direct, &reference);
    assert!(
        diff < TOL,
        "{label} {engine:?} {kernel:?}: max diff {diff:e}"
    );

    let total: [f64; 3] = std::array::from_fn(|x| direct.iter().map(|g| g[x]).sum());
    assert!(
        total.iter().all(|t| t.abs() < TOL),
        "{label} {engine:?} {kernel:?}: net force {total:?}"
    );
    direct
}

fn check(kind: ShellKind, engine: Engine, kernel: EriKernel) {
    let direct = check_basis(&basis(kind), engine, kernel, &format!("{kind:?}"));
    assert!(max_abs(&direct) > 1e-3);
}

#[test]
fn matches_dense_reference_cartesian_os() {
    check(ShellKind::Cartesian, Engine::OsHgp, EriKernel::Coulomb);
}

#[test]
fn matches_dense_reference_spherical_os() {
    check(ShellKind::Spherical, Engine::OsHgp, EriKernel::Coulomb);
}

#[test]
fn matches_dense_reference_cartesian_rys() {
    check(ShellKind::Cartesian, Engine::Rys, EriKernel::Coulomb);
}

#[test]
fn matches_dense_reference_spherical_auto() {
    check(ShellKind::Spherical, Engine::Auto, EriKernel::Coulomb);
}

#[test]
fn matches_dense_reference_erf_kernel() {
    check(
        ShellKind::Spherical,
        Engine::Auto,
        EriKernel::Erf { omega: 0.3 },
    );
}

#[test]
fn matches_dense_reference_erf_kernel_other_omegas() {
    check(
        ShellKind::Cartesian,
        Engine::Auto,
        EriKernel::Erf { omega: 1.7 },
    );
    check(
        ShellKind::Spherical,
        Engine::Auto,
        EriKernel::Erf { omega: 0.05 },
    );
    check_basis(
        &mixed_kind_basis(),
        Engine::Auto,
        EriKernel::Erf { omega: 0.8 },
        "mixed kinds",
    );
}

/// The `Erf` kernel has a single (Rys) implementation, so the engine the plan
/// was built with must not change a single bit of its result.
#[test]
fn erf_kernel_ignores_forced_engine() {
    let basis = basis(ShellKind::Spherical);
    let d = Densities::new(basis.nao());
    let kernel = EriKernel::Erf { omega: 0.45 };
    let run = |engine| {
        let contractor = DirectGradContractor::with_engine(&basis, engine).unwrap();
        let density = contractor.density(&d.coulomb(), &d.exchange());
        contractor.contract(&density, kernel, 0.0)
    };
    let auto = run(Engine::Auto);
    assert_eq!(run(Engine::OsHgp), auto);
    assert_eq!(run(Engine::Rys), auto);
}

fn shifted(c: [f64; 3], target: [f64; 3], d: [f64; 3]) -> [f64; 3] {
    if c == target {
        [c[0] + d[0], c[1] + d[1], c[2] + d[2]]
    } else {
        c
    }
}

/// Clone `basis` with every shell on atom `target` displaced by `d`.
fn shift_basis(basis: &Basis, target: [f64; 3], d: [f64; 3]) -> Basis {
    let shells = basis
        .shells()
        .iter()
        .map(|s| {
            Shell::with_kind(
                s.l(),
                shifted(s.center(), target, d),
                s.exponents().to_vec(),
                s.coefficients().to_vec(),
                s.kind(),
            )
            .unwrap()
        })
        .collect();
    Basis::new(shells)
}

fn unit(axis: usize, h: f64) -> [f64; 3] {
    let mut d = [0.0; 3];
    d[axis] = h;
    d
}

/// `Σ Γ_{μνλσ}·(μν|λσ)` from the value integrals alone.
fn contracted_energy(basis: &Basis, kernel: EriKernel, gamma: &[f64]) -> f64 {
    let eri = match kernel {
        EriKernel::Coulomb => basis.eri(),
        erf => basis.eri_kernel(erf),
    };
    eri.iter().zip(gamma).map(|(v, g)| v * g).sum()
}

fn fd_basis() -> Basis {
    let a = [0.0, 0.0, 0.0];
    let b = [0.85, -0.3, 0.5];
    let c = [-0.5, 0.9, 0.2];
    Basis::new(vec![
        Shell::new(0, a, vec![2.1, 0.5], vec![0.45, 0.6]).unwrap(),
        Shell::new(1, a, vec![0.9], vec![1.0]).unwrap(),
        Shell::new_spherical(2, b, vec![0.8], vec![1.0]).unwrap(),
        Shell::new(0, b, vec![0.6], vec![1.0]).unwrap(),
        Shell::new(1, c, vec![1.2, 0.4], vec![0.4, 0.7]).unwrap(),
        Shell::new(2, c, vec![0.65], vec![1.0]).unwrap(),
    ])
}

fn fd_check(kernel: EriKernel) {
    let basis = fd_basis();
    let n = basis.nao();
    let d = Densities::new(n);
    let (coulomb, exchange) = (d.coulomb(), d.exchange());
    let gamma = dense_gamma(n, &coulomb, &exchange);

    let contractor = basis.direct_grad_contractor().unwrap();
    let density = contractor.density(&coulomb, &exchange);
    let analytic = contractor.contract(&density, kernel, 0.0);
    assert!(max_abs(&analytic) > 1e-3);

    let mut worst = 0.0_f64;
    for (&atom, gradient) in basis.atoms().iter().zip(&analytic) {
        for (axis, &g) in gradient.iter().enumerate() {
            let plus = shift_basis(&basis, atom, unit(axis, H));
            let minus = shift_basis(&basis, atom, unit(axis, -H));
            let fd = (contracted_energy(&plus, kernel, &gamma)
                - contracted_energy(&minus, kernel, &gamma))
                / (2.0 * H);
            worst = worst.max((g - fd).abs());
        }
    }
    assert!(worst < FD_TOL, "{kernel:?} vs FD: worst {worst:.3e}");
}

#[test]
fn matches_finite_difference_of_contracted_eri() {
    fd_check(EriKernel::Coulomb);
}

#[test]
fn matches_finite_difference_of_contracted_erf_eri() {
    fd_check(EriKernel::Erf { omega: 0.6 });
}

/// One shell of angular momentum `l` at shell index `slot` among an s and a p
/// shell, all three on different atoms. Index 0 puts the shell in the second,
/// third and fourth quartet positions, index 2 in the first three.
fn sweep_basis(l: usize, slot: usize, kind: ShellKind) -> Basis {
    let mut shells = vec![
        Shell::with_kind(0, [0.0, 0.0, 0.0], vec![1.3, 0.4], vec![0.5, 0.6], kind).unwrap(),
        Shell::with_kind(1, [1.1, -0.5, 0.4], vec![0.7], vec![1.0], kind).unwrap(),
    ];
    let high = Shell::with_kind(l, [-0.6, 0.9, 0.7], vec![0.9], vec![1.0], kind).unwrap();
    shells.insert(slot, high);
    Basis::new(shells)
}

fn sweep(ls: std::ops::RangeInclusive<usize>, engines: &[Engine]) {
    for l in ls {
        for slot in 0..3 {
            for kind in [ShellKind::Cartesian, ShellKind::Spherical] {
                let basis = sweep_basis(l, slot, kind);
                for &engine in engines {
                    let label = format!("l={l} slot={slot} {kind:?}");
                    let direct = check_basis(&basis, engine, EriKernel::Coulomb, &label);
                    assert!(max_abs(&direct) > 1e-4, "{label}: trivial gradient");
                }
            }
        }
    }
}

#[test]
fn every_angular_momentum_up_to_g_in_every_slot() {
    sweep(0..=3, &[Engine::OsHgp, Engine::Rys, Engine::Auto]);
    sweep(4..=4, &[Engine::Auto]);
}

#[test]
#[ignore = "expensive: dense reference with an h (l=5) shell raised to l=6; run in release with --include-ignored"]
fn h_shell_in_every_slot_all_engines() {
    assert_eq!(MAX_GRAD_L, 5);
    sweep(4..=4, &[Engine::OsHgp, Engine::Rys]);
    sweep(5..=5, &[Engine::OsHgp, Engine::Rys, Engine::Auto]);
}

/// Two h shells on different atoms meet in one bra pair and in bra and ket, so
/// a raised (l=6) shell is paired with another l=5 shell.
#[test]
#[ignore = "expensive: dense reference over (hh|hh) quartets; run in release with --include-ignored"]
fn two_h_shells_on_different_atoms() {
    for kind in [ShellKind::Cartesian, ShellKind::Spherical] {
        let basis = Basis::new(vec![
            Shell::with_kind(5, [0.0, 0.0, 0.0], vec![0.9], vec![1.0], kind).unwrap(),
            Shell::with_kind(0, [0.0, 0.0, 0.0], vec![0.6], vec![1.0], kind).unwrap(),
            Shell::with_kind(5, [0.8, -0.5, 0.6], vec![0.7], vec![1.0], kind).unwrap(),
        ]);
        let label = format!("two h {kind:?}");
        let direct = check_basis(&basis, Engine::Auto, EriKernel::Coulomb, &label);
        assert!(max_abs(&direct) > 1e-4, "{label}: trivial gradient");
    }
}

/// Atom `a` carries three shells and atom `b` two, so the canonical list holds
/// `(ii|kl)` and `(ij|ij)` quartets, three-on-one-atom quartets in each
/// arrangement, and one-centre quartets that the contractor drops.
#[test]
fn shells_sharing_centers() {
    let a = [0.1, -0.2, 0.3];
    let b = [1.0, 0.7, -0.6];
    let basis = Basis::new(vec![
        Shell::new(0, a, vec![1.5, 0.4], vec![0.5, 0.6]).unwrap(),
        Shell::new(1, a, vec![0.8], vec![1.0]).unwrap(),
        Shell::new_spherical(2, a, vec![0.6], vec![1.0]).unwrap(),
        Shell::new(0, b, vec![0.7], vec![1.0]).unwrap(),
        Shell::new(1, b, vec![1.0, 0.35], vec![0.4, 0.7]).unwrap(),
    ]);
    assert_eq!(basis.atoms().len(), 2);
    for engine in [Engine::OsHgp, Engine::Rys, Engine::Auto] {
        let direct = check_basis(&basis, engine, EriKernel::Coulomb, "shared centers");
        assert!(max_abs(&direct) > 1e-3);
    }
    check_basis(
        &basis,
        Engine::Auto,
        EriKernel::Erf { omega: 0.4 },
        "shared centers",
    );
}

/// A lone pair of shells on two atoms: every surviving quartet has `i == j`,
/// `k == l` or bra == ket.
#[test]
fn two_shells_on_two_atoms() {
    let basis = Basis::new(vec![
        Shell::new(1, [0.0, 0.0, 0.0], vec![0.9], vec![1.0]).unwrap(),
        Shell::new_spherical(2, [0.7, 0.4, -0.9], vec![0.6], vec![1.0]).unwrap(),
    ]);
    for engine in [Engine::OsHgp, Engine::Rys] {
        let direct = check_basis(&basis, engine, EriKernel::Coulomb, "two shells");
        assert!(max_abs(&direct) > 1e-3);
    }
}

#[test]
fn single_atom_gradient_is_exactly_zero() {
    let at = [0.3, -1.2, 0.8];
    let basis = Basis::new(vec![
        Shell::new(0, at, vec![2.0, 0.5], vec![0.4, 0.7]).unwrap(),
        Shell::new(1, at, vec![0.9], vec![1.0]).unwrap(),
        Shell::new_spherical(2, at, vec![0.7], vec![1.0]).unwrap(),
        Shell::new(3, at, vec![0.6], vec![1.0]).unwrap(),
    ]);
    assert_eq!(basis.atoms().len(), 1);
    let d = Densities::new(basis.nao());
    for engine in [Engine::OsHgp, Engine::Rys, Engine::Auto] {
        let contractor = DirectGradContractor::with_engine(&basis, engine).unwrap();
        assert_eq!(contractor.natom(), 1);
        let density = contractor.density(&d.coulomb(), &d.exchange());
        for kernel in [EriKernel::Coulomb, EriKernel::Erf { omega: 0.5 }] {
            for threshold in [0.0, 1e-10] {
                let direct = contractor.contract(&density, kernel, threshold);
                assert_eq!(direct, vec![[0.0; 3]], "{engine:?} {kernel:?}");
            }
        }
    }
}

fn mixed_kind_basis() -> Basis {
    let a = [0.0, 0.0, 0.0];
    let b = [0.95, 0.3, -0.5];
    let c = [-0.4, -0.8, 0.6];
    Basis::new(vec![
        Shell::new(0, a, vec![1.7, 0.4], vec![0.5, 0.6]).unwrap(),
        Shell::new_spherical(2, a, vec![0.8], vec![1.0]).unwrap(),
        Shell::new(2, b, vec![0.7], vec![1.0]).unwrap(),
        Shell::new_spherical(1, b, vec![0.9, 0.3], vec![0.6, 0.5]).unwrap(),
        Shell::new_spherical(3, c, vec![0.65], vec![1.0]).unwrap(),
        Shell::new(1, c, vec![0.75], vec![1.0]).unwrap(),
    ])
}

#[test]
fn mixed_cartesian_and_spherical_shells() {
    let basis = mixed_kind_basis();
    assert!(basis.nao() < basis.nao_cart());
    for engine in [Engine::OsHgp, Engine::Rys, Engine::Auto] {
        let direct = check_basis(&basis, engine, EriKernel::Coulomb, "mixed kinds");
        assert!(max_abs(&direct) > 1e-3);
    }
}

/// Deep s/p contractions next to single-primitive d/f shells: under the
/// dispatch policy the low-`l_total` or highly contracted quartets go to
/// OS/HGP and the uncontracted `l_total ≥ 6` ones to Rys.
fn contracted_basis() -> Basis {
    let a = [0.0, 0.0, 0.0];
    let b = [1.0, -0.3, 0.45];
    let c = [-0.5, 0.85, -0.4];
    let exps =
        |k: usize, top: f64| -> Vec<f64> { (0..k).map(|i| top * 0.4_f64.powi(i as i32)).collect() };
    let coeffs = |k: usize| -> Vec<f64> { (0..k).map(|i| 0.3 + 0.1 * i as f64).collect() };
    Basis::new(vec![
        Shell::new(0, a, exps(6, 40.0), coeffs(6)).unwrap(),
        Shell::new(1, a, exps(4, 9.0), coeffs(4)).unwrap(),
        Shell::new_spherical(2, a, vec![0.8], vec![1.0]).unwrap(),
        Shell::new(0, b, exps(5, 20.0), coeffs(5)).unwrap(),
        Shell::new_spherical(3, b, vec![0.7], vec![1.0]).unwrap(),
        Shell::new(1, c, exps(3, 4.0), coeffs(3)).unwrap(),
        Shell::new_spherical(2, c, vec![0.6], vec![1.0]).unwrap(),
    ])
}

#[test]
fn contracted_shells_auto_dispatch_uses_both_engines() {
    let basis = contracted_basis();
    let shells = basis.shells();
    let contractor = basis.direct_grad_contractor().unwrap();
    let (mut os, mut rys) = (0, 0);
    for &(i, j) in contractor.bra_pairs() {
        for &(k, l) in contractor.bra_pairs() {
            let degree = [i, j, k, l].iter().map(|&s| shells[s].n_prim()).product();
            let engine = select_engine(
                shells[i].l() + shells[j].l(),
                shells[k].l() + shells[l].l(),
                degree,
            );
            if engine == Engine::OsHgp {
                os += 1;
            } else {
                rys += 1;
            }
        }
    }
    assert!(
        os > 0 && rys > 0,
        "dispatch is one-sided: os={os} rys={rys}"
    );

    let auto = check_basis(&basis, Engine::Auto, EriKernel::Coulomb, "contracted");
    assert!(max_abs(&auto) > 1e-3);

    let d = Densities::new(basis.nao());
    let forced = |engine| {
        let contractor = DirectGradContractor::with_engine(&basis, engine).unwrap();
        let density = contractor.density(&d.coulomb(), &d.exchange());
        contractor.contract(&density, EriKernel::Coulomb, 0.0)
    };
    let (os_hgp, rys) = (forced(Engine::OsHgp), forced(Engine::Rys));
    let cross = max_diff(&os_hgp, &rys);
    assert!(cross < TOL, "OsHgp vs Rys: {cross:e}");
    assert!(max_diff(&os_hgp, &auto) < TOL);
    assert!(max_diff(&rys, &auto) < TOL);
}

#[test]
fn coulomb_and_exchange_lists_are_additive() {
    let basis = basis(ShellKind::Spherical);
    let d = Densities::new(basis.nao());
    let contractor = basis.direct_grad_contractor().unwrap();
    let contract = |coulomb: &[GradDensityTerm<'_>], exchange: &[GradDensityTerm<'_>]| {
        let density = contractor.density(coulomb, exchange);
        contractor.contract(&density, EriKernel::Coulomb, 0.0)
    };
    let both = contract(&d.coulomb(), &d.exchange());
    let coulomb_only = contract(&d.coulomb(), &[]);
    let exchange_only = contract(&[], &d.exchange());
    assert!(max_abs(&coulomb_only) > 1e-3 && max_abs(&exchange_only) > 1e-3);
    let summed: Vec<[f64; 3]> = coulomb_only
        .iter()
        .zip(&exchange_only)
        .map(|(c, x)| std::array::from_fn(|axis| c[axis] + x[axis]))
        .collect();
    assert!(max_diff(&summed, &both) < 1e-12);

    let nothing = contract(&[], &[]);
    assert_eq!(nothing, vec![[0.0; 3]; contractor.natom()]);
}

#[test]
fn bra_pair_tasks_sum_in_any_order() {
    let basis = basis(ShellKind::Spherical);
    let n = basis.nao();
    let p = symmetric(n, 4);
    let terms = [GradDensityTerm {
        factor: 1.0,
        left: &p,
        right: &p,
    }];
    let contractor = basis.direct_grad_contractor().unwrap();
    let nsh = basis.shells().len();
    assert_eq!(contractor.bra_pairs().len(), nsh * (nsh + 1) / 2);
    let density = contractor.density(&terms, &terms);
    let serial = contractor.contract(&density, EriKernel::Coulomb, 0.0);

    let mut parts = vec![vec![[0.0; 3]; contractor.natom()]; 2];
    let mut workspaces = [DirectGradWorkspace::new(), DirectGradWorkspace::new()];
    for (t, &bra) in contractor.bra_pairs().iter().rev().enumerate() {
        contractor.accumulate_bra_pair(
            &mut workspaces[t % 2],
            bra,
            &density,
            EriKernel::Coulomb,
            0.0,
            &mut parts[t % 2],
        );
    }
    let summed: Vec<[f64; 3]> = (0..contractor.natom())
        .map(|a| std::array::from_fn(|x| parts[0][a][x] + parts[1][a][x]))
        .collect();
    assert!(max_diff(&summed, &serial) < 1e-12);
}

#[test]
fn screening_error_is_bounded_by_threshold_scale() {
    let far = [14.0, 0.0, 0.0];
    let mut shells = basis(ShellKind::Spherical).shells().to_vec();
    shells.push(Shell::new_spherical(1, far, vec![0.9], vec![1.0]).unwrap());
    shells.push(Shell::new_spherical(0, far, vec![1.4, 0.4], vec![0.5, 0.6]).unwrap());
    let basis = Basis::new(shells);
    let n = basis.nao();
    let p = symmetric(n, 5);
    let terms = [GradDensityTerm {
        factor: 1.0,
        left: &p,
        right: &p,
    }];
    let contractor = basis.direct_grad_contractor().unwrap();
    let density = contractor.density(&terms, &terms);
    let exact = contractor.contract(&density, EriKernel::Coulomb, 0.0);
    let screened = contractor.contract(&density, EriKernel::Coulomb, 1e-12);
    assert!(max_diff(&screened, &exact) < 1e-9);
}

/// Four atoms with every interatomic distance between 8.5 and 19.3 bohr and
/// diffuse valence shells, so shell-pair overlaps span many orders of magnitude.
fn extended_basis() -> Basis {
    let atoms = [
        [0.0, 0.0, 0.0],
        [8.5, 0.0, 0.5],
        [8.0, 9.0, -0.4],
        [19.0, 3.0, 1.0],
    ];
    let mut shells = Vec::new();
    for (a, &at) in atoms.iter().enumerate() {
        shells.push(Shell::new(0, at, vec![1.2, 0.25], vec![0.5, 0.6]).unwrap());
        shells.push(Shell::new(1, at, vec![0.3], vec![1.0]).unwrap());
        if a % 2 == 0 {
            shells.push(Shell::new_spherical(2, at, vec![0.28], vec![1.0]).unwrap());
        }
    }
    Basis::new(shells)
}

/// Threshold `0` keeps every quartet and must reproduce the dense reference.
/// For `τ > 0` the Schwarz factors bound the integrals, not their derivatives,
/// so the error is not `≤ τ`: it measures `0.2·τ`–`2·τ` here and is required to
/// stay below `100·τ`, two orders under the gradient itself even at `τ = 1e-6`.
/// An infinite threshold drops every quartet.
#[test]
fn screening_on_extended_system() {
    let basis = extended_basis();
    let n = basis.nao();
    let d = Densities::new(n);
    let (coulomb, exchange) = (d.coulomb(), d.exchange());
    let gamma = dense_gamma(n, &coulomb, &exchange);
    let reference = basis.eri_grad_contract(&gamma).unwrap();

    let contractor = basis.direct_grad_contractor().unwrap();
    let density = contractor.density(&coulomb, &exchange);
    let exact = contractor.contract(&density, EriKernel::Coulomb, 0.0);
    assert!(max_abs(&exact) > 1e-3);
    let diff = max_diff(&exact, &reference);
    assert!(diff < TOL, "threshold 0 vs dense reference: {diff:e}");

    for threshold in [1e-12, 1e-9, 1e-6] {
        let screened = contractor.contract(&density, EriKernel::Coulomb, threshold);
        let err = max_diff(&screened, &exact);
        let bound = 100.0 * threshold;
        assert!(
            err < bound,
            "threshold {threshold:e}: error {err:e} exceeds {bound:e}"
        );
    }

    let nothing = contractor.contract(&density, EriKernel::Coulomb, f64::INFINITY);
    assert_eq!(nothing, vec![[0.0; 3]; contractor.natom()]);
}

#[test]
fn rejects_shells_above_gradient_limit() {
    let basis = Basis::new(vec![Shell::new(6, [0.0; 3], vec![1.0], vec![1.0]).unwrap()]);
    assert!(matches!(
        DirectGradContractor::new(&basis),
        Err(integral::IntegralError::AngularMomentumTooHighForGradient { l: 6, max: 5 })
    ));
    assert!(basis.direct_grad_contractor().is_err());
    assert!(DirectGradContractor::with_engine(&basis, Engine::Rys).is_err());
}

#[test]
#[should_panic]
fn density_rejects_wrong_size_left_matrix() {
    let basis = basis(ShellKind::Spherical);
    let n = basis.nao();
    let (good, short) = (symmetric(n, 1), symmetric(n - 1, 1));
    let contractor = basis.direct_grad_contractor().unwrap();
    let _ = contractor.density(
        &[GradDensityTerm {
            factor: 1.0,
            left: &short,
            right: &good,
        }],
        &[],
    );
}

#[test]
#[should_panic]
fn density_rejects_wrong_size_exchange_right_matrix() {
    let basis = basis(ShellKind::Spherical);
    let n = basis.nao();
    let (good, long) = (symmetric(n, 1), symmetric(n + 1, 1));
    let contractor = basis.direct_grad_contractor().unwrap();
    let _ = contractor.density(
        &[],
        &[GradDensityTerm {
            factor: 1.0,
            left: &good,
            right: &long,
        }],
    );
}

fn accumulate_one(bra: (usize, usize), kernel: EriKernel, out_len_excess: usize) {
    let basis = basis(ShellKind::Spherical);
    let p = symmetric(basis.nao(), 1);
    let terms = [GradDensityTerm {
        factor: 1.0,
        left: &p,
        right: &p,
    }];
    let contractor = basis.direct_grad_contractor().unwrap();
    let density = contractor.density(&terms, &[]);
    let mut out = vec![[0.0; 3]; contractor.natom() + out_len_excess];
    contractor.accumulate_bra_pair(
        &mut DirectGradWorkspace::new(),
        bra,
        &density,
        kernel,
        0.0,
        &mut out,
    );
}

#[test]
fn accumulate_accepts_a_canonical_bra_pair() {
    accumulate_one((1, 0), EriKernel::Coulomb, 0);
}

#[test]
#[should_panic]
fn accumulate_rejects_wrong_length_output() {
    accumulate_one((1, 0), EriKernel::Coulomb, 1);
}

#[test]
#[should_panic]
fn accumulate_rejects_non_canonical_bra_pair() {
    accumulate_one((0, 1), EriKernel::Coulomb, 0);
}

#[test]
#[should_panic(expected = "finite omega > 0")]
fn accumulate_rejects_non_positive_omega() {
    accumulate_one((1, 0), EriKernel::Erf { omega: 0.0 }, 0);
}

#[test]
fn contractor_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DirectGradContractor>();
    fn assert_send<T: Send>() {}
    assert_send::<DirectGradWorkspace>();
}

/// A density carries per-shell-block bounds, so it belongs to the shell layout
/// of the plan that prepared it: equal `nao` is not enough.
#[test]
#[should_panic(expected = "different shell layout")]
fn accumulate_rejects_density_from_a_different_shell_layout() {
    let origin = [0.0, 0.0, 0.0];
    let one_p = Basis::new(vec![Shell::new(1, origin, vec![1.0], vec![1.0]).unwrap()]);
    let three_s = Basis::new(
        (0..3)
            .map(|i| Shell::new(0, [f64::from(i), 0.0, 0.0], vec![1.0], vec![1.0]).unwrap())
            .collect(),
    );
    assert_eq!(one_p.nao(), three_s.nao());
    let p = symmetric(one_p.nao(), 1);
    let terms = [GradDensityTerm {
        factor: 1.0,
        left: &p,
        right: &p,
    }];
    let density = one_p.direct_grad_contractor().unwrap().density(&terms, &[]);
    let _ = three_s
        .direct_grad_contractor()
        .unwrap()
        .contract(&density, EriKernel::Coulomb, 1e-12);
}

/// `threshold = 0.0` disables screening altogether, including for a bra pair
/// whose Schwarz bound underflows to exactly zero while its derivative
/// integrals do not.
#[test]
fn zero_threshold_keeps_pairs_with_zero_schwarz_bound() {
    let basis = Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![1.0], vec![1.0]).unwrap(),
        Shell::new(0, [13.0, 0.0, 0.0], vec![1.0], vec![1.0]).unwrap(),
    ]);
    let n = basis.nao();
    assert_eq!(basis.schwarz_bounds()[n], 0.0, "premise: Q_10 underflows");
    // Off-diagonal only, so the density selects the `(10|10)` quartet alone.
    let p = vec![0.0, 1.0, 1.0, 0.0];
    let terms = [GradDensityTerm {
        factor: 1.0,
        left: &p,
        right: &p,
    }];
    let gamma = dense_gamma(n, &terms, &[]);
    // The OS/HGP engine drops this primitive pair in both code paths, so only
    // the Rys reference is required to be non-zero.
    for engine in [Engine::Rys, Engine::OsHgp, Engine::Auto] {
        let reference = basis.eri_grad_contract_with(engine, &gamma).unwrap();
        let contractor = DirectGradContractor::with_engine(&basis, engine).unwrap();
        let density = contractor.density(&terms, &[]);
        let direct = contractor.contract(&density, EriKernel::Coulomb, 0.0);
        let scale = max_abs(&reference);
        if engine == Engine::Rys {
            assert!(scale > 0.0, "premise: the dense Rys reference is non-zero");
        }
        let diff = max_diff(&direct, &reference);
        assert!(
            diff <= 1e-10 * scale,
            "{engine:?}: direct {direct:?} vs reference {reference:?}"
        );
    }
}
