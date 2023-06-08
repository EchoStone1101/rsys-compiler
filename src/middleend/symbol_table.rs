//! Implements symbol table for SYSY.
//! SYSY currently emulates strict C behavior. Consequently,
//! it does not support any kind of overloading, or name mangling.
//! All symbols (in the same scope) with the same name must have 
//! only one type.

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

    /// A stack that stores the resuming BBs in current
    /// analysis. For now, this includes only a `while`
    /// loop. When a `continue` or `break` terminates a
    /// BB, it consults this stack to determine where to
    /// jump to.
    /// The format is (ContBB, BreakBB).
    resume_bbs: Vec<(BasicBlock, BasicBlock)>, 
}

impl SymbolTable {

    pub fn new() -> Self {
        Default::default()
    }

    /// Initialize with a set of predefined symbols.
    /// Used to handle things like library functions.
    pub fn init(&mut self, pre_def: &Vec<(String, Symbol)>) {
        
        assert!(self.symbols.is_empty() && self.scope == 0, "[SYMTAB] init() should only be called on empty table");

        for (name, sym) in pre_def.iter() {
            let ident = Ident { ident: name.clone(), token_pos: (0, 0) };
            assert!(self.symbols.insert(ident, vec![(self.scope, *sym)]).is_none(), "[SYMTAB] `pre_def` is not unique");
        }
        
        // Use a push to make sure these symbols are globally visible,
        // but can also be shadowed.
        // These symbols are never popped.
        self.push();
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
                let popped = self.symbols.get_mut(name)
                    .expect("[SYMTAB] no such symbol")
                    .pop()
                    .expect("[SYMTAB] no such symbol");
                assert!(popped.0 == *version, "[SYMTAB] version mismatch");
                self.scope_stack.pop();
            }
            else {
                break
            }
        }
    }

    pub fn push_resume_bb(&mut self, resume_bb: (BasicBlock, BasicBlock)) {
        self.resume_bbs.push(resume_bb)
    }

    pub fn pop_resume_bb(&mut self) -> (BasicBlock, BasicBlock) {
        self.resume_bbs.pop().expect("[SYMTAB] pop_resume_bb() at empty stack")
    }

    pub fn get_resume_bb(&self) -> Option<(BasicBlock, BasicBlock)> {
        self.resume_bbs.last().map(|bb| *bb)
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

    pub fn replace(&mut self, ident: &Ident, new_sym: Symbol) -> Option<Symbol> {
        if let Some(v) = self.symbols.get_mut(ident) {
            if let Some((version, sym)) = v.last_mut() {
                assert!(*version == self.scope, "[symtab]: replace must happen in the same scope");
                let old_sym = *sym;
                *sym = new_sym;
                return Some(old_sym)
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
    Value(Value, bool, bool, bool), // Value, isConst, isInit, isGlobal
}
