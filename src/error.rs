//! Various errors during syntactic and semantic analysis.

#[allow(unused_imports)]
use std::fmt;
use std::process::exit;
use koopa::ir::Type;
use crate::frontend::ast::{Ident, TokenPos};
use lalrpop_util::{ErrorRecovery, ParseError};
use lalrpop_util::lexer::Token;
use colored::*;

#[derive(Debug)]
struct Pos {
    pub lineno: usize,
    pub colno: usize,
    pub raw_position: usize,
    pub line_position: usize,
}

#[allow(unused)]
impl Pos {
    pub fn from_raw_position(raw_input: &[u8], location: usize) -> Self {
        if raw_input.len() == 0 {
            return Pos {lineno: 1, colno: 1, raw_position: 0, line_position: 0}
        }
        assert!(location < raw_input.len());
        let mut line_position = 0;
        let mut lineno = 1;
        let mut colno = 1;
        for idx in 0..location {
            // Deliberately not handling Unicode characters for now
            let ch = raw_input[idx];
            if ch == b'\n' {
                line_position = idx + 1;
                lineno += 1;
                colno = 1;
            }
            else {
                colno += 1;
            }
        }
        Pos {lineno, colno, raw_position: location, line_position}
    }

    pub fn adjust(&mut self, raw_input: &[u8], adjustment: usize) {
        let new_position = self.raw_position + adjustment;
        assert!((new_position as usize) <= raw_input.len());
        
        for idx in self.raw_position..new_position as usize {
            let ch = raw_input[idx];
            if ch == b'\n' {
                self.line_position = idx + 1;
                self.lineno += 1;
                self.colno = 1;
            }
            else {
                self.colno += 1;
            }
        }
        self.raw_position = new_position;
    }
}

fn find_start_of_token<F>(raw_input: &[u8], start: usize, predicate: F) -> usize
where F: Fn(u8) -> bool
{
    let mut location = start;
    if location == raw_input.len() && location >= 1 {
        location -= 1;
    }
    while location >= 1 {
        if !raw_input[location].is_ascii_whitespace() {
            // Get to the end of previous token, if currently at whitespace
            break;
        }
        location = location-1;
    }
    while location >= 1 {
        if raw_input[location-1].is_ascii_whitespace() {
            // Get to the start of this token
            break;
        }
        if !predicate(raw_input[location-1]) {
            break;
        }
        location = location-1;
    }
    location
}

#[derive(Debug, Clone, Copy)]
enum PromptMode {
    Error,
    Warning,
    Note,
}

fn prompt_error(
    raw_input: &[u8],
    start: usize,
    end: usize,
    msg: &str,
    mode: PromptMode,
) {
    let mut start_pos = Pos::from_raw_position(raw_input, start);
    let mut lineno = start_pos.lineno;
    let colno = start_pos.colno;
    let line_position = start_pos.line_position;
    start_pos.adjust(raw_input, end - start);
    let end_pos = start_pos;

    match mode {
        PromptMode::Error => eprint!("{}: ", "error".red()),
        PromptMode::Warning => eprint!("{}: ", "warning".yellow()),
        PromptMode::Note => eprint!("{}: ", "note".blue()),
    };
    eprintln!("{}", msg);

    if start == 0 && end == 0 {
        return
    }

    let lineno_len = end_pos.lineno.to_string().len();
    let mut tab = String::with_capacity(lineno_len);
    for _ in 0..lineno_len {
        tab.push(' ')
    }

    eprintln!("{}{} Ln {}, Col {}", tab, "-->".blue(), lineno, colno);
    eprintln!("{} {}", tab, "|".blue());

    let bad_range = String::from(std::str::from_utf8(&raw_input[start..end]).unwrap());
    let mut lines = bad_range.lines();
    
    prompt_lineno(lineno, lineno_len);
    eprint!("{} {}", "|".blue(), String::from(std::str::from_utf8(&raw_input[line_position..start]).unwrap()));
    let mut line = lines.next();
    while let Some(line_str) = line {
        match mode {
            PromptMode::Error => eprint!("{}", line_str.red()),
            PromptMode::Warning => eprint!("{}", line_str.yellow()),
            PromptMode::Note => eprint!("{}", line_str.blue()),
        }

        line = lines.next();
        if line.is_some() {
            lineno += 1;
            eprintln!();
            prompt_lineno(lineno, lineno_len);
            eprint!("{} ", "|".blue());
        }
        else {
            let mut idx = end;
            while idx < raw_input.len() {
                if raw_input[idx] == b'\n' {
                    break;
                }
                eprint!("{}", raw_input[idx] as char);
                idx += 1;
            }
            eprintln!();
        }
    }
    eprintln!("{} {}", tab, "|".blue());
    eprintln!();

    // eprint!("{} {} ", tab, "|".blue());
    // idx = line_position;
    // for _ in 0..colno-1 {
    //     if raw_input[idx] == b'\t' {
    //         eprint!("\t")
    //     }
    //     else {
    //         eprint!(" ")
    //     }
    //     idx += 1;
    // }
    // for _ in 0..end-start {
    //     eprint!("{}", "^".red())
    // }
    eprintln!();
}

fn prompt_lineno(lineno: usize, width: usize) {
    let lineno_str = lineno.to_string();
    eprint!("{}", lineno_str.blue());
    eprint!(" ");
    for _ in lineno_str.len()..width {
        eprint!(" ")
    }
}

pub fn parse_general_error<'input>(
    raw_input: &[u8], 
    recovery: ErrorRecovery<usize, Token<'input>, &'static str>,
) -> ! {

    match recovery.error {
        ParseError::InvalidToken { location } => {
            let mut end = location;
            while end < raw_input.len() {
                if raw_input[end].is_ascii_whitespace() {
                    break;
                }
                end += 1;
            }
            prompt_error(raw_input, location, end, "invalid token", PromptMode::Error);
        },
        ParseError::UnrecognizedEof { location, expected } => {
            let start = find_start_of_token(raw_input, location, |_| true);
            if let Some(tok) = expected.iter()
                .find(|&s| s == "\"}\"" || s == "\"]\"" || s == "\")\"") 
            {
                prompt_error(raw_input, start, location, format!("missing {} at EOF", tok).as_str(), PromptMode::Error);
            }
            else if let Some(_) = expected.iter().find(|&s| s == "\";\"") {
                prompt_error(raw_input, start, location, format!("unexpected EOF, maybe missing \";\"?").as_str(), PromptMode::Error);
            }
            else {
                prompt_error(raw_input, start, location, "unexpected EOF", PromptMode::Error);
                eprintln!("{}: {:?}", "expected tokens".blue(), expected);
            }
        },
        ParseError::UnrecognizedToken { token: (start, tok, end), expected } => {
            if let Some(tok) = expected.iter()
                .find(|&s| s == "\"}\"" || s == "\"]\"" || s == "\")\"") 
            {
                prompt_error(raw_input, start, end, format!("missing {} before this", tok).as_str(), PromptMode::Error);
            }
            else if tok.1 == "}" || tok.1 == "]" || tok.1 == ")" {
                prompt_error(raw_input, start, end, format!("perhaps extraneous \"{}\" here", tok.1).as_str(), PromptMode::Error);
            }
            else if let Some(_) = expected.iter().find(|&s| s == "\";\"") {
                prompt_error(raw_input, start, end, format!("unexpected token, maybe missing \";\"?").as_str(), PromptMode::Error);
            }
            else {
                prompt_error(raw_input, start, end, "unexpected token", PromptMode::Error);
                eprintln!("{}: {:?}", "expected tokens".blue(), expected);
            }
        },
        _ => unreachable!() /* ::User is not used, and ::ExtraToken will not occur */
    }

    exit(-1)
}

#[derive(Debug)]
pub enum SemanticError {
    IntegerOutOfRange,
    TypeMismatch(Type, Type),
    ExtraIndex(TokenPos, Type),
    ConstExpected,
    UndefinedIdent,
    MisuseOfFuncIdent(Ident),
    RedefOfIdent(Ident),
    DividedByZero,
    ArrayPartialDeref(Type),
    ConstUninit,
    ConstAssignment(TokenPos),
    NegArrayBound(i32),
}

pub fn semantic_error(raw_input: &[u8], start: usize, end: usize, error: &SemanticError) -> ! {

    match error {
        SemanticError::IntegerOutOfRange => 
            prompt_error(raw_input, start, end, "integer out of range [0, 2^31-1]", PromptMode::Error),
        SemanticError::TypeMismatch(expected, found) => {
            prompt_error(raw_input, start, end, format!("mismatched types, expected {}, found {}", expected, found).as_str(), PromptMode::Error)
        },
        SemanticError::ExtraIndex(pos, ty) => {
            prompt_error(raw_input, start, end, "extraneous index", PromptMode::Error);
            prompt_error(raw_input, pos.0, pos.1, format!("the base has type {}", ty).as_str(), PromptMode::Note);
        },
        SemanticError::ConstExpected => 
            prompt_error(raw_input, start, end, "value is unknown at compile time, expected a constant", PromptMode::Error),
        SemanticError::UndefinedIdent => 
            prompt_error(raw_input, start, end, "use of undeclared identifier", PromptMode::Error),
        SemanticError::MisuseOfFuncIdent(func_id) => {
            prompt_error(raw_input, start, end, "identifier is a function; cannot be used like this", PromptMode::Error);
            let pos = func_id.token_pos;
            prompt_error(raw_input, pos.0, pos.1, "originally defined here", PromptMode::Note);
        },
        SemanticError::RedefOfIdent(id) => {
            prompt_error(raw_input, start, end, "redefiniton of identifier", PromptMode::Error);
            let pos = id.token_pos;
            prompt_error(raw_input, pos.0, pos.1, "originally defined here", PromptMode::Note);
        },
        SemanticError::DividedByZero => {
            prompt_error(raw_input, start, end, "the divisor evaluates to 0", PromptMode::Error);
        },
        SemanticError::ArrayPartialDeref(ty) => {
            prompt_error(raw_input, start, end, "partially dereferenced value can only be used as function argument", PromptMode::Error);
            prompt_error(raw_input, 0, 0, format!("the expression has type {}", ty).as_str(), PromptMode::Note);
        }
        SemanticError::ConstUninit => {
            prompt_error(raw_input, start, end, "constant must be initialized", PromptMode::Error);
        },
        SemanticError::ConstAssignment(pos) => {
            prompt_error(raw_input, start, end, "assigning a constant value", PromptMode::Error);
            prompt_error(raw_input, pos.0, pos.1, "which is defined here", PromptMode::Note);
        },
        SemanticError::NegArrayBound(i) => {
            prompt_error(raw_input, start, end, format!("array bound {} must not be negative", i).as_str(), PromptMode::Error);
        }
    };

    exit(-1)
}

#[derive(Debug)]
pub enum Warning {
    ValueUndef,
    VarUninit(TokenPos),
    IndexOutOfBound(i32, TokenPos, usize),
    IntegerOverflow,
    MissingRet(TokenPos),
}

pub fn warning(raw_input: &[u8], start: usize, end: usize, warning: &Warning) {

    match warning {
        Warning::ValueUndef =>
            prompt_error(raw_input, start, end, "the value of this is undefined, using 0 as default", PromptMode::Warning),
        Warning::VarUninit(pos) => {
            prompt_error(raw_input, start, end, "this variable may be uninitialized", PromptMode::Warning);
            prompt_error(raw_input, pos.0, pos.1, "originally defined here", PromptMode::Note);
        }
        Warning::IndexOutOfBound(index, pos, bound) => {
            prompt_error(raw_input, start, end, format!("index value {} is out of bound", *index).as_str(), PromptMode::Warning);
            prompt_error(raw_input, pos.0, pos.1, format!("the base has bound [0, {})", *bound).as_str(), PromptMode::Note);
        },
        Warning::IntegerOverflow => 
            prompt_error(raw_input, start, end, "evaluating this value yields integer overflow, defauting to wrapping", PromptMode::Warning),
            Warning::MissingRet(pos) => {
            prompt_error(raw_input, start, end, "non-void function does not return a value", PromptMode::Warning);
            prompt_error(raw_input, pos.0, pos.1, "which is defined here", PromptMode::Note);
        }
    }
}