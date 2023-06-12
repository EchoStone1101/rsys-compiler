use std::fs::{read_to_string, File};
use std::io::{Result, Write};
use lalrpop_util::{lalrpop_mod, ErrorRecovery};
use koopa::ir::{Program, FunctionData, Type};
use koopa::back::KoopaGenerator;
use koopa::opt::*;
use clap::{Parser, ValueEnum};

pub mod frontend;
pub mod middleend;
pub mod backend;
pub mod error;

use frontend::ast::CompUnit;
use middleend::{SymbolTable, Symbol, opt};
use error::parse_general_error;

lalrpop_mod!(sysy, "/frontend/sysy.rs");

#[derive(Parser, Debug)]
#[command(author = "echostone<EchoStone@gmail.com>")]
#[command(about = "A Rust compiler for SYSY, rich in error prompts.")]
#[command(version, long_about = None)]
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
    #[value(name = "koopa")]
    Koopa,
    #[value(name = "riscv")]
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
            // declarations. These incluse user defined functions
            // and SYSY library functions.
            let lib_funcs = vec![

                (
                    "@getint".to_string(),
                    Symbol::Function(program.new_func(FunctionData::new(
                        "@getint".into(), 
                        vec![],
                        Type::get_i32(),
                    )))
                ),

                (
                    "@getch".to_string(),
                    Symbol::Function(program.new_func(FunctionData::new(
                        "@getch".into(), 
                        vec![],
                        Type::get_i32(),
                    )))
                ),

                (
                    "@getarray".to_string(),
                    Symbol::Function(program.new_func(FunctionData::new(
                        "@getarray".into(), 
                        vec![Type::get_pointer(Type::get_i32())],
                        Type::get_i32(),
                    )))
                ),

                (
                    "@putint".to_string(),
                    Symbol::Function(program.new_func(FunctionData::new(
                        "@putint".into(), 
                        vec![Type::get_i32()],
                        Type::get_unit(),
                    )))
                ),

                (
                    "@putch".to_string(),
                    Symbol::Function(program.new_func(FunctionData::new(
                        "@putch".into(), 
                        vec![Type::get_i32()],
                        Type::get_unit(),
                    )))
                ),

                (
                    "@putarray".to_string(),
                    Symbol::Function(program.new_func(FunctionData::new(
                        "@putarray".into(), 
                        vec![Type::get_i32(), Type::get_pointer(Type::get_i32())],
                        Type::get_unit(),
                    )))
                ),

                (
                    "@starttime".to_string(),
                    Symbol::Function(program.new_func(FunctionData::new(
                        "@starttime".into(), 
                        vec![],
                        Type::get_unit(),
                    )))
                ),

                (
                    "@stoptime".to_string(),
                    Symbol::Function(program.new_func(FunctionData::new(
                        "@stoptime".into(), 
                        vec![],
                        Type::get_unit(),
                    )))
                ),
            ];
            sym_tab.init(&lib_funcs);

            ast.register_decls(input.as_bytes(), &mut program, &mut sym_tab);
            // Then populate Program.
            ast.append_to_program(input.as_bytes(), &mut program, &mut sym_tab);

            // Optionally apply optimizaton passes
            let mut passman = PassManager::new();
            passman.register(Pass::Function(Box::new(opt::ElimUnusedValue)));
            passman.register(Pass::Function(Box::new(opt::ElimUnreachableBlock)));
            passman.register(Pass::Function(Box::new(opt::ElimUselessBlock)));
            // Apply twice deliberately
            passman.register(Pass::Function(Box::new(opt::ElimUselessBlock)));
            passman.run_passes(&mut program);

            let mut gen = KoopaGenerator::new(Vec::new());
            gen.generate_on(&program).unwrap();
            let koopa_ir = std::str::from_utf8(&gen.writer()).unwrap().to_string();
            outfile.write_all(koopa_ir.as_bytes())?;
        }
        _ => todo!(),
    }

    Ok(())
}
