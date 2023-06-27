# RSYSY

## Overview

### What is RSYSY?

[RSYSY](https://github.com/EchoStone1101/rsys-compiler) is a compiler bulit for the [SYSY language](https://pku-minic.github.io/online-doc/#/preface/), written in Rust. It contains:

- **A user-friendly front-end.** The front-end is a robust parser for SYSY which parses SYSY into [Koopa IR](https://pku-minic.github.io/online-doc/#/lv0-env-config/koopa). It comes with rigorous syntatical and semantical checks, and a very nice Rust-like error prompt system.

- **A handful of Koopa IR middle-end passes,** including some flavors of dead code elimination, load-store analysis and more.

- **A dedicated RISCV back-end** that converts Koopa IR into executable RISCV code with *decent performance*, thanks to a prototype register allocation algorithm and even some *runtime* techniques.

RSYSY is the lab project for the 2023 Spring Compiler's Principle course at Peking University. By [this commit](https://github.com/EchoStone1101/rsys-compiler/commit/9c4c3b7df992738959e411edce59d6a10716af10), the implementation passes all the tests for both Koopa IR and RISCV generation, and ranks #2 on the performance benchmark rankings of all submissions, with a merely 4-second difference from the #1.
(It so happens that the #1 submission is also implemented in Rust. Rust for the Win!)


### Features

#### ✅　Syntactic Checks
The frontend reports the first syntax error with Rust-like prompts, and even tries to diagnose the cause:
```c
int main() {
  int c[3][2]
  return 0
}
```
```log
error: unexpected token, maybe missing ";"?
 --> Ln 4, Col 3
  |
4 |   return 0
  |
```
The real terminal output uses the `colored` [crate](https://crates.io/crates/colored) to highlight the unexpected token; in this case `return` would be highlighted.

#### ✅　Semantic Checks
Likewise, semantic errors and warnings are also reported:
```c
int global = 10;

int func() {
  int a;
  global = a;
}

int func2(int a[][5]) {
  return 1;
  global = 4;
}

int main() {
  int c[3][4];

  if (1) {/* do this */} else {/* that */}
  while (0) {/* I'm dead */}
  
  return func2(c);
}
```
```log
warning: non-void function does not return a value
note: which is defined here
 --> Ln 4, Col 5
  |
4 | int func() {
  |


warning: unreachable statement
  --> Ln 11, Col 3
   |
11 |   global = 4;
   |


warning: the condition evaluates to 1, so the if-branch is always taken
  --> Ln 17, Col 3
   |
17 |   if (1) {/* do this */} else {/* that */}
   |


warning: the condition evaluates to 0, so the loop is never entered
  --> Ln 18, Col 3
   |
18 |   while (0) {/* I'm dead */}
   |


error: mismatched types, expected *[i32, 5], found *[i32, 4]
  --> Ln 20, Col 16
   |
20 |   return func2(c);
   |
```

#### ✅　Richer SYSY Semantics
RSYSY supports a superset of the SYSY language, including:
- **All-scope function**, so that back-and-forth recursion is possible; another approach is to support function declaration.
- **Constant array evaluation**, so that you can use constant array elements to initialize other constants, and they are constantly evaluated to the best effort.
- **Global initialization with global value**; namely you can initialize one global variable with another.

#### ✅　Optimization Passes

RSYSY currently implements 5 optimization passes. See the optimizations sub-section for details.

#### ✅　RISCV ABI-Compliant Codegen

RSYSY takes the IR generated from a SYSY program, and lowers it to RISCV32IM object code, obeying all ABI constraints such as calling convention, register usage and memory layout (constant arrays are placed in `rodata` instead of `data`; safety matters!).

#### ✅　Register allocation And A Runtime Trick

RSYSY backend for RISCV implements an on-the-fly register allocation algorithm with decent performance. Also, inspirations were drawn from Python `memoize`, and we designed a simple caching scheme at *runtime* for expensive *simple* fucntions.

### Optimizations

RSYSY currently implements 5 passes in `middleend/opt.rs`:

- **ElimUnreachedBlocks**, for eliminating branches with known conditions.

- **ElimUselessBlocks**, for eliminating jump-only blocks, which are reminiscent of the raw IR generated from SYSY AST.

- **ElimUnusedValues**, for eliminating usused calculations in the raw IR, which are also mainly from the raw IR generation.

- **ElimLoadStore**, a *prototype* pass that aims to minimize memory accesses of local variables. Due to [this issue](https://github.com/pku-minic/koopa/issues/4), the pass cannot be implemented direcly at the IR level (at least without very tedious effort), so its is currently used in a makeshift way by the back-end.
Nevertheless, this pass eliminates all unnecessary loads and stores within basicblocks.

- **SimpleFucntionAnalysis**, an informative analysis that detects "simple" functions in a SYSY program. A "simple" function has no side-effect and does not access external state, hence its return value is strictly a function of its argument. This analysis enables *runtime caching* for some of the functions.

### Dependencies
```log
[dependencies]
koopa = "0.0.7"
lalrpop-util = {version = "0.20.0", features = ["lexer"]}
colored = "2"
clap = { version = "4.3.1", features = ["derive"] }
```

RSYSY depends heavily on [koopa](https://docs.rs/koopa/0.0.7/koopa/) for IR manipulation, and on [lalrpop](https://docs.rs/lalrpop/0.20.0/lalrpop/) for the frontend. [colored](https://docs.rs/colored/2.0.0/colored/) is a simple-to-use crate that delivers the nice-looking colored error prompts. Finally, [clap](https://docs.rs/clap/4.3.5/clap/) is my beloved way to generate CLI without writting any code; it generates a command line parser directly from struct declaration.

## Getting Started

Build simply with `cargo build`.

RSYSY currently runs in 3 modes: `-koopa`, `-riscv` and `-perf`. CLI arguments must also include a `path_to_source_file` and `-o path_to_output_file`.
In any case, the source file should contain SYSY source code in UTF-8 encoding. If RSYSY finds no syntactic and semantic errors in the source file,
the output file will contain the compiled code. 

In `-koopa` mode, the output file will contain Koopa IR in text form. The IR is guaranteed to be valid as a standalone Koopa IR program.

In `-riscv` mode, the output file will contain RISCV32IM object code for the `riscv-unknown-linux-gnu-gcc` architecture. RSYSY *do not*
handle any linking. Use `riscv-toolchain` for that purpose.

In `-perf` mode, the output file will contain RISCV object code, similar to `-riscv` mode, but with all optimization strategies applied.
Currently, this mode is only an alias of `-riscv`, in which all optimizations are also applied. This will change in the future.
Think of this mode like the aggressive optimization mode like `-O3` in `gcc` and other compilers alike.

Example:
```sh
rsysy -koopa hello.c -o hello.koopa
```
