//! Implements the AST structures for SYSY.

use koopa::ir::*;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

#[derive(Debug)]
pub enum CompUnit {
    Decl(Option<Box<CompUnit>>, Box<Decl>),
    FuncDef(Option<Box<CompUnit>>, Box<FuncDef>),
}

#[derive(Debug)]
pub enum Decl {
    ConstDecl(Box<ConstDecl>),
    VarDecl(Box<VarDecl>),
}

#[derive(Debug)]
pub struct ConstDecl {
    pub btype: BType,
    pub const_def: Vec<Box<ConstDef>>,
}


impl ConstDecl {
    pub fn new(btype: BType, const_def: Vec<Box<ConstDef>>) -> Self {
        ConstDecl { btype, const_def }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BType {
    Int,
}

impl Into<Type> for BType {
    fn into(self) -> Type {
        match self {
            BType::Int => Type::get_i32(),
        }
    }
}

#[derive(Debug)]
pub struct ConstDef {
    pub ident: Ident,
    pub const_exp: Vec<Box<ConstExp>>, 
    pub const_init_val: Option<Rc<ConstInitVal>>,
}

impl ConstDef {
    pub fn new(ident: Ident, const_exp: Vec<Box<ConstExp>>, const_init_val: Option<Rc<ConstInitVal>>) -> Self {
        ConstDef { ident, const_exp, const_init_val }
    }
}

#[derive(Debug)]
pub enum ConstInitVal {
    ConstExp(Box<ConstExp>),
    ValList(Vec<Rc<ConstInitVal>>, TokenPos),
}

#[derive(Debug)]
pub struct VarDecl {
    pub btype: BType,
    pub var_def: Vec<Box<VarDef>>,
}

#[derive(Debug)]
pub struct VarDef {
    pub ident: Ident,
    pub const_exp: Vec<Box<ConstExp>>,
    pub init_val: Option<Rc<InitVal>>,
}

impl VarDef {
    pub fn new(ident: Ident, const_exp: Vec<Box<ConstExp>>, init_val: Option<Rc<InitVal>>) -> Self {
        VarDef { ident, const_exp, init_val }
    }
}

#[derive(Debug)]
pub enum InitVal {
    Exp(Box<Exp>),
    ValList(Vec<Rc<InitVal>>, TokenPos),
}

#[derive(Debug)]
pub struct FuncDef {
    pub func_type: FuncType,
    pub ident: Ident,
    pub func_fparams: Option<Box<FuncFParams>>,
    pub block: Box<Block>,
    pub (crate) params: Vec<(Ident, Type)>,
}

impl FuncDef {
    pub fn new(func_type: FuncType, ident: Ident, func_fparams: Option<Box<FuncFParams>>, block: Box<Block>) -> Self {
        FuncDef { func_type, ident, func_fparams, block, params: Vec::new(), }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum FuncType {
    Void,
    Int,
}

impl Into<Type> for FuncType {
    fn into(self) -> Type {
        match self {
            FuncType::Int => Type::get_i32(),
            FuncType::Void => Type::get_unit(),
        }
    }
}

#[derive(Debug)]
pub struct FuncFParams {
    pub params: Vec<Box<FuncFParam>>,
}

#[derive(Debug)]
pub struct FuncFParam {
    pub btype: BType,
    pub ident: Ident,
    pub dimensions: Option<Vec<Box<ConstExp>>>, // If Some, then there is an implicit non-sized dimension
}

impl FuncFParam {
    pub fn new(btype: BType, ident: Ident, dimensions: Option<Vec<Box<ConstExp>>>) -> Self {
        FuncFParam { btype, ident, dimensions }
    }
}

#[derive(Debug)]
pub struct Block {
    pub block_item: Vec<Box<BlockItem>>,
}

#[derive(Debug)]
pub enum BlockItem {
    Decl(Box<Decl>, TokenPos),
    Stmt(Box<Stmt>, TokenPos),
}

#[derive(Debug)]
pub enum Stmt {
    Assign(Box<LVal>, Box<Exp>),
    Exp(Option<Box<Exp>>),
    Block(Box<Block>),
    If(Box<Exp>, Box<Stmt>, Option<Box<Stmt>>, TokenPos),
    While(Box<Exp>, Box<Stmt>, TokenPos),
    Break(TokenPos),
    Continue(TokenPos),
    Ret(Option<Box<Exp>>, TokenPos),
}

#[derive(Debug)]
pub enum Exp {
    LOrExp(Box<LOrExp>, TokenPos),
}

#[derive(Debug)]
pub struct LVal {
    pub ident: Ident,
    pub indices: Vec<Box<Exp>>,
    pub token_pos: TokenPos,
}

impl LVal {
    pub fn new(ident: Ident, indices: Vec<Box<Exp>>, token_pos: TokenPos) -> Self {
        LVal { ident, indices, token_pos }
    }
}

#[derive(Debug)]
pub enum PrimaryExp {
    Bracketed(Box<Exp>),
    LVal(Box<LVal>),
    Number(Number),
}

#[derive(Debug)]
pub enum Number {
    IntConst(IntConst),
}

#[derive(Debug)]
pub enum UnaryExp {
    PrimaryExp(Box<PrimaryExp>, TokenPos),
    FuncCall(Ident, Option<Box<FuncRParams>>, TokenPos),
    Op(UnaryOp, Box<UnaryExp>, TokenPos),
}

#[derive(Debug)]
pub enum UnaryOp {
    Plus,
    Minus,
    Not,
}

#[derive(Debug)]
pub struct FuncRParams {
    pub params: Vec<Box<Exp>>,
}

#[derive(Debug)]
pub enum MulExp {
    UnaryExp(Box<UnaryExp>),
    Mul(Box<MulExp>, Box<UnaryExp>, TokenPos),
    Div(Box<MulExp>, Box<UnaryExp>, TokenPos),
    Rem(Box<MulExp>, Box<UnaryExp>, TokenPos),
}

#[derive(Debug)]
pub enum AddExp {
    MulExp(Box<MulExp>),
    Add(Box<AddExp>, Box<MulExp>, TokenPos),
    Sub(Box<AddExp>, Box<MulExp>, TokenPos),
}

#[derive(Debug)]
pub enum RelExp {
    AddExp(Box<AddExp>),
    Lt(Box<RelExp>, Box<AddExp>, TokenPos),
    Gt(Box<RelExp>, Box<AddExp>, TokenPos),
    Le(Box<RelExp>, Box<AddExp>, TokenPos),
    Ge(Box<RelExp>, Box<AddExp>, TokenPos),
}

#[derive(Debug)]
pub enum EqExp {
    RelExp(Box<RelExp>),
    Eq(Box<EqExp>, Box<RelExp>, TokenPos),
    Neq(Box<EqExp>, Box<RelExp>, TokenPos),
}

#[derive(Debug)]
pub enum LAndExp {
    EqExp(Box<EqExp>),
    And(Box<LAndExp>, Box<EqExp>, TokenPos),
}

#[derive(Debug)]
pub enum LOrExp {
    LAndExp(Box<LAndExp>),
    Or( Box<LOrExp>, Box<LAndExp>, TokenPos),
}

#[derive(Debug)]
pub struct ConstExp {
    pub exp: Box<Exp>,
    pub pos: TokenPos,
}

pub type TokenPos = (usize, usize);

#[derive(Debug, Clone)]
pub struct Ident {
    pub ident: String,
    pub token_pos: TokenPos,
}

impl Ident {
    pub fn to_string(&self) -> &String {
        &self.ident
    }
}

impl PartialEq for Ident {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident
    }
}

impl Eq for Ident {}

impl Hash for Ident {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ident.hash(state)
    }
}

type IntConst = i32;