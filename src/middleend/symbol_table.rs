//! Implements symbol table for SYSY.
//! SYSY currently emulates strict C behavior. Consequently,
//! it does not support any kind of overloading, or name mangling.
//! All symbols with the same name must have only one type.

#[allow(unused_imports)]
use koopa::ir::*;
use crate::error::{semantic_error, SemanticError};
use crate::frontend::ast::Ident;
use std::collections::HashMap;

/// A symbol table for SYSY.
#[derive(Debug, Default)]
pub struct SymbolTable {
    symbols: HashMap<Ident, Vec<(usize, Symbol)>>,
    scope_stack: Vec<(usize, Ident)>,
    scope: usize,
}

impl SymbolTable {
    pub fn new() -> Self {
        Default::default()
    }

    /// Increment the scope count.
    pub fn push(&mut self) {
        self.scope += 1;
    }

    /// Decrement the scope count, and drop symbols that
    /// goes out of scope.
    pub fn pop(&mut self) {
        assert!(self.scope >= 1);
        self.scope -= 1;
        while let Some((version, name)) = self.scope_stack.last() {
            if *version > self.scope {
                // Pop this symbol
                self.symbols.remove(name).expect("no such symbol");
                self.scope_stack.pop();
            }
            else {
                break
            }
        }
    }

    pub fn add(&mut self, raw_input: &[u8], ident: &Ident, symbol: Symbol) {
        if let Some((id, versioned_symbol)) = self.symbols.get_key_value(&ident) {
            if !versioned_symbol.is_empty() {
                assert!(versioned_symbol.last().unwrap().0 <= self.scope);
                if versioned_symbol.last().unwrap().0 == self.scope {
                    // Redefinition of the same symbol within current scope
                    semantic_error(raw_input, ident.token_pos.0, ident.token_pos.1,
                        &SemanticError::RedefOfIdent(id.clone()))
                }
            }
            let versioned_symbol = self.symbols.get_mut(&ident).unwrap();
            versioned_symbol.push((self.scope, symbol));
        }
        else {
            self.symbols.insert(ident.clone(), vec![(self.scope, symbol)]);
        }
        self.scope_stack.push((self.scope, ident.clone()));
    }

    pub fn get(&self, ident: &Ident) -> Option<Symbol> {
        if let Some(v) = self.symbols.get(&ident) {
            if let Some((_, sym)) = v.last() {
                return Some(*sym)
            }
        }
        None
    }

    pub fn get_ident(&self, ident: &Ident) -> Option<(Symbol, Ident)> {
        if let Some((id, v)) = self.symbols.get_key_value(&ident) {
            if let Some((_, sym)) = v.last() {
                return Some((*sym, id.clone()))
            }
        }
        None
    }
}

/// A SYSY semantic symbol.
/// A Symbol is versioned, and must be unique with respect to (version, name).
#[derive(Debug, Clone, Copy)]
pub enum Symbol {
    Function(Function),
    Value(Value, bool), // Value, isConst
}
