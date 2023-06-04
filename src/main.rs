use std::fs::{read_to_string, File};
use std::io::{Result, Write};
use lalrpop_util::{lalrpop_mod, ErrorRecovery};
use koopa::ir::Program;
use koopa::back::KoopaGenerator;
use clap::{Parser, ValueEnum};

pub mod frontend;
pub mod middleend;
pub mod error;

use frontend::ast::CompUnit;
use middleend::SymbolTable;
use error::parse_general_error;

lalrpop_mod!(sysy);

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Type of the object code
    #[arg(value_enum)]
    mode: Mode,

    /// Path to the output file
    #[arg(short)]
    out_file: String,

    in_file: String,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Mode {
    #[value(name = "-koopa")]
    Koopa,
    #[value(name = "-riscv")]
    RiscV,
}

#[allow(unused)]
#[allow(dead_code)]
fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().map(|arg|
        match arg.as_str() {
            "-koopa" => "koopa".to_string(),
            "-riscv" => "riscv".to_string(),
            "-perf" => "perf".to_string(),
            s => s.to_string(),
        }
    ).collect();
    let cli = Cli::parse_from(args.into_iter());

    // Input
    let input = read_to_string(cli.in_file)?;

    // Output
    let mut outfile = File::create(cli.out_file)?;

    match cli.mode {
        Mode::Koopa => {
            // Parse using lalrpop
            let mut ast: Box<CompUnit> = sysy::CompUnitParser::new().parse(&input, &input)
            .unwrap_or_else(|error| 
                parse_general_error(&input.as_bytes(), ErrorRecovery {error, dropped_tokens: Vec::new()}));

            let mut program = Program::new();
            let mut sym_tab = SymbolTable::new();

            // First populate the symbol table with function
            // declarations.
            ast.register_func_decls(input.as_bytes(), &mut program, &mut sym_tab);
            // Then populate Program.
            ast.append_to_program(input.as_bytes(), &mut program, &mut sym_tab);

            let mut gen = KoopaGenerator::new(Vec::new());
            gen.generate_on(&program).unwrap();
            let koopa_ir = std::str::from_utf8(&gen.writer()).unwrap().to_string();
            outfile.write_all(koopa_ir.as_bytes())?;
        }
        _ => todo!(),
    }

    Ok(())
}
