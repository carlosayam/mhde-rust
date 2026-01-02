# REPOSITORY ARCHIVED

-------------------------------------------------------------
**I'm no longer pursuing this in Rust, switching back to Python**
-------------------------------------------------------------


# Minimum Hellinger Distance Estimator in Rust

(c) 2025, Carlos Aya-Moreno

Example implementation of the MHDE in Rust / Burn.

Using [Burn](https://burn.dev/) as it seems a promising alternative to Python's torch,
although this estimator is _not_ a traditional NN model.

Implements the estimator in [paper.pdf](paper.pdf), equations (3) and (4).

1. Install Rust and Cargo

2. Run `cargo build` to compile

3. Run `cargo run --example cauchy -- [<loc> [<scale> [<num>]]] [-s <seed>] [--split]` to run an example, e.g.

```
cargo run --example cauchy -- 20.0 3.0 8000 -s 2
Sample size: 8000
Sum 0..8000
Starting params
Loc: 2.112353202855953
Scale: 0.974881241658409

HD^2 Hat: 0.4408844036401467 (10)
HD^2 Hat: 0.31012944844602164 (20)
HD^2 Hat: 0.2184262076825243 (30)
HD^2 Hat: 0.15171614051778848 (40)
HD^2 Hat: 0.10427249470985633 (50)
HD^2 Hat: 0.07203002349431054 (60)
HD^2 Hat: 0.050830821810698534 (70)
HD^2 Hat: 0.03592512086041766 (80)
HD^2 Hat: 0.022844375348942236 (90)
HD^2 Hat: 0.01014394620118697 (100)
HD^2 Hat: 0.001365748239268938 (110)
HD^2 Hat: -0.000349152193437563 (120)
HD^2 Hat: -0.000018442406594987304 (130)
HD^2 Hat: -0.00038761153067246035 (140)
HD^2 Hat: -0.00044375295298437756 (150)
HD^2 Hat: -0.0004334651083570673 (160)
HD^2 Hat: -0.0004532844375053635 (170)
HD^2 Hat: -0.0004535884403691348 (180)
HD^2 Hat: -0.000453962277132236 (190)
HD^2 Hat: -0.00045475307387010666 (200)
HD^2 Hat: -0.0004546729409364847 (210)
HD^2 Hat: -0.00045475121822757814 (220)

End params (iterations=226)
Loc: 20.025672551184265
Scale: 3.0323162228343508
```
