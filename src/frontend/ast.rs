//! Implements the AST structures for SYSY.

#[allow(unused_imports)]
use koopa::ir::*;
use koopa::ir::builder_traits::*;
use koopa::ir::entities::ValueData;
use std::hash::{Hash, Hasher};
use crate::error::{semantic_error, SemanticError, warning, Warning};
use crate::middleend::{SymbolTable, Symbol};

#[derive(Debug)]
pub enum CompUnit {
    Decl(Option<Box<CompUnit>>, Box<Decl>),
    FuncDef(Option<Box<CompUnit>>, Box<FuncDef>),
}

impl CompUnit {

    pub fn register_func_decls(&mut self, raw_input: &[u8], program: &mut Program, sym_tab: &mut SymbolTable) {
        match self {
            CompUnit::Decl(_, _) => {}
            CompUnit::FuncDef(more, func_def) => {
                if let Some(more) =  more {
                    more.register_func_decls(raw_input, program, sym_tab);
                }
                func_def.func_decl(raw_input, program, sym_tab);
            }
        }
    }

    pub fn append_to_program(&self, raw_input: &[u8], program: &mut Program, sym_tab: &mut SymbolTable) {
        match self {
            CompUnit::Decl(more, decl) => {
                if let Some(more) =  more {
                    more.append_to_program(raw_input, program, sym_tab);
                }
                decl.decl(raw_input, program, None, sym_tab, None);
            }
            CompUnit::FuncDef(more, func_def) => {
                if let Some(more) =  more {
                    more.append_to_program(raw_input, program, sym_tab);
                }
                func_def.func_def(raw_input, program, sym_tab);
            }
        }
    }
}

#[derive(Debug)]
pub enum Decl {
    ConstDecl(Box<ConstDecl>),
    VarDecl(Box<VarDecl>),
}

impl Decl {
    fn decl(
        &self, 
        raw_input: &[u8], 
        program: &mut Program,
        function: Option<Function>,
        sym_tab: &mut SymbolTable,
        bb: Option<BasicBlock>,
    ) {
        match self {
            Decl::ConstDecl(const_decl) => const_decl.const_decl(raw_input, program, function, sym_tab),
            Decl::VarDecl(var_decl) => var_decl.var_decl(raw_input, program, function, sym_tab, bb),
        }
    }
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

    fn const_decl(
        &self, 
        _raw_input: &[u8], 
        _program: &mut Program,
        _function: Option<Function>,
        _sym_tab: &mut SymbolTable,
    ) {
        todo!()
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
    pub const_init_val: Box<ConstInitVal>,
}

impl ConstDef {
    pub fn new(ident: Ident, const_exp: Vec<Box<ConstExp>>, const_init_val: Box<ConstInitVal>) -> Self {
        ConstDef { ident, const_exp, const_init_val }
    }
}

#[derive(Debug)]
pub enum ConstInitVal {
    ConstExp(Box<ConstExp>),
    ValList(Vec<Box<ConstInitVal>>),
}

#[derive(Debug)]
pub struct VarDecl {
    pub btype: BType,
    pub var_def: Vec<Box<VarDef>>,
}

impl VarDecl {
    fn var_decl(
        &self, 
        raw_input: &[u8], 
        program: &mut Program,
        function: Option<Function>,
        sym_tab: &mut SymbolTable,
        bb: Option<BasicBlock>,
    ) {
        for def in self.var_def.iter() {
            def.var_def(raw_input, program, function, sym_tab, bb, self.btype);
        }
    }
}

#[derive(Debug)]
pub struct VarDef {
    pub ident: Ident,
    pub const_exp: Vec<Box<ConstExp>>,
    pub init_val: Option<Box<InitVal>>,
}

impl VarDef {
    pub fn new(ident: Ident, const_exp: Vec<Box<ConstExp>>, init_val: Option<Box<InitVal>>) -> Self {
        VarDef { ident, const_exp, init_val }
    }

    fn var_def(
        &self, 
        raw_input: &[u8], 
        program: &mut Program,
        function: Option<Function>,
        sym_tab: &mut SymbolTable,
        bb: Option<BasicBlock>,
        btype: BType,
    ) {
        // TODO: parse self.init_val

        // Must then first evaluate the actual type of this definition
        let def = if let Some(func) = function {

            // Inside a function

            let mut ty = match btype {
                BType::Int => Type::get_i32(),
            };
            for exp in self.const_exp.iter().rev() {
                // `const_exp` already checks if the Exp is constant
                let index = exp.const_exp(raw_input, program, function, sym_tab, bb);
                let index_data = program.func(func).dfg().value(index);
                check_value_type(raw_input, Type::get_i32(), index_data.ty().clone(), &exp.pos);
                let (bound, _) = to_i32(index_data).unwrap();
                ty = Type::get_array(ty, bound as usize);
            }
            let alloc = program.func_mut(func).dfg_mut().new_value().alloc(ty);
            program.func_mut(func).layout_mut().bbs_mut()
                .front_node_mut().unwrap().insts_mut().push_key_back(alloc).unwrap();
            alloc
        }
        else {

            // Global scope

            let mut ty = match btype {
                BType::Int => Type::get_i32(),
            };
            for exp in self.const_exp.iter().rev() {
                // `const_exp` already checks if the Exp is constant
                let index = exp.const_exp(raw_input, program, function, sym_tab, None);
                let index_data = &program.borrow_value(index);
                check_value_type(raw_input, Type::get_i32(), index_data.ty().clone(), &exp.pos);
                let (bound, _) = to_i32(index_data).unwrap();
                ty = Type::get_array(ty, bound as usize);
            }
            todo!()
        };

        // Add to symbol table; may err
        sym_tab.add(raw_input, &self.ident, Symbol::Value(def, false));
        

    }
}

#[derive(Debug)]
pub enum InitVal {
    Exp(Box<Exp>),
    ValList(Vec<Box<InitVal>>),
}

#[derive(Debug)]
pub struct FuncDef {
    pub func_type: FuncType,
    pub ident: Ident,
    pub func_fparams: Option<Box<FuncFParams>>,
    pub block: Box<Block>,
    params: Vec<(Ident, Type)>,
}

impl FuncDef {
    pub fn new(func_type: FuncType, ident: Ident, func_fparams: Option<Box<FuncFParams>>, block: Box<Block>) -> Self {
        FuncDef { func_type, ident, func_fparams, block, params: Vec::new(), }
    }

    /// This is marked as public, and meant to be called before calling `func_def()`,
    /// in order to first set up the symbol table.
    /// With out this special treatment, it will be impossible to write altertaive
    /// recursion (because in SYSY there is no function declaration).
    pub fn func_decl(&mut self, raw_input: &[u8], program: &mut Program, sym_tab: &mut SymbolTable) {
        let params = self.func_fparams.as_ref().map_or(
            vec![], 
            |params| params.func_f_params(raw_input, program, sym_tab)
        );

        let new_func = program.new_func(FunctionData::with_param_names(
            self.ident.to_string().clone(), 
            params
                .iter()
                .map(|(id, ty)| (Some(id.to_string().clone()), ty.clone()))
                .collect(),
            self.func_type.into()
        ));

        // Add function ident to symbol table; may error
        sym_tab.add(raw_input, &self.ident, Symbol::Function(new_func));
        self.params.extend(params);
    }
    
    fn func_def(&self, raw_input: &[u8], program: &mut Program, sym_tab: &mut SymbolTable) {
        assert!(!self.params.is_empty());
        let params = &self.params;

        if let Some(Symbol::Function(func)) = sym_tab.get(&self.ident) {
            let new_func_data = program.func_mut(func);

            // Prepare basic blocks
            let alloc = new_func_data.dfg_mut().new_bb().basic_block(Some("%alloc".into()));
            let entry = new_func_data.dfg_mut().new_bb().basic_block(Some("%body".into()));
            new_func_data.layout_mut().bbs_mut().push_key_back(alloc).unwrap();
            new_func_data.layout_mut().bbs_mut().push_key_back(entry).unwrap();

            // Into a new scope
            sym_tab.push();

            for ((param, _), idx) in params.iter().zip(0..new_func_data.params().len()) {
                let value = new_func_data.params()[idx];

                // Allocate space for param
                // NOTE: It is completely find from the KoopaIR perspective
                // to use param idents as Value's. However, when converting to
                // object code, this may cause problems, as function params behave
                // similarly to local variables, but may have their value in registers.
                // If we do not allocate backing storage here, we will have to track
                // individually how their values change.
                let ty = new_func_data.dfg().value(value).ty().clone();
                let addr = new_func_data.dfg_mut().new_value().alloc(ty);
                let store = new_func_data.dfg_mut().new_value().store(value, addr);
                new_func_data.layout_mut().bb_mut(alloc).insts_mut().extend([addr, store]);

                // Add function param ident to symbol table; may error
                sym_tab.add(raw_input, &param, Symbol::Value(addr, false));
            }

            // Handle the function body
            self.block.block(raw_input, program, func, sym_tab, entry);

            // Out of the scope
            sym_tab.pop();
        }
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

impl FuncFParams {
    fn func_f_params(&self, 
        raw_input: &[u8],
        program: &mut Program,
        sym_tab: &mut SymbolTable,
    ) -> Vec<(Ident, Type)> {
        Vec::from_iter(self.params.iter().map(|p| p.func_f_param(raw_input, program, sym_tab)))
    }
}

#[derive(Debug)]
pub struct FuncFParam {
    pub btype: BType,
    pub ident: Ident,
    pub dimensions: Option<Vec<Box<ConstExp>>>, // If Some, then there is an implicit non-sized dimension
}

impl FuncFParam {
    fn func_f_param(
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        sym_tab: &mut SymbolTable,
    ) -> (Ident, Type) {
        let mut ptype: Type = self.btype.into();
        let ident = self.ident.clone();

        if let Some(dim) = &self.dimensions {
            for const_exp in dim.iter().rev() {
                let index = const_exp.const_exp(raw_input, program, None, sym_tab, None);
                let index = program.borrow_value(index);
                if !index.ty().is_i32() {
                    // `const_exp` must have type i32
                    todo!()
                }
                if let ValueKind::Integer(index) = index.kind() {
                    if index.value() < 0 {
                        // `const_exp` must not be negative
                        todo!()
                    }
                    ptype = Type::get_array(ptype, index.value() as usize);
                }
                else {
                    // `const_exp` must be evaluated to Integer
                    todo!()
                }
            }
            ptype = Type::get_pointer(ptype);
        }
        (ident, ptype)
    }
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

impl Block {
    fn block(
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        function: Function,
        sym_tab: &mut SymbolTable,
        bb: BasicBlock,
    ) {
        for item in self.block_item.iter() {
            if item.block_item(raw_input, program, function, sym_tab, bb).is_none() {
                // Terminating block item 
                break;
            }
        }
    }
}

#[derive(Debug)]
pub enum BlockItem {
    Decl(Box<Decl>),
    Stmt(Box<Stmt>),
}

impl BlockItem {
    fn block_item(
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        function: Function,
        sym_tab: &mut SymbolTable,
        bb: BasicBlock,
    ) -> Option<BasicBlock> {
        match self {
            BlockItem::Decl(decl) => {
                decl.decl(raw_input, program, Some(function), sym_tab, Some(bb));
                Some(bb) // resume BB; see Stmt::stmt()
            },
            BlockItem::Stmt(stmt) => stmt.stmt(raw_input, program, function, sym_tab, bb),
        }
    }
}

#[derive(Debug)]
pub enum Stmt {
    Assign(Box<LVal>, Box<Exp>),
    Exp(Option<Box<Exp>>),
    Block(Box<Block>),
    If(Box<Exp>, Box<Stmt>, Option<Box<Stmt>>),
    While(Box<Exp>, Box<Stmt>),
    Break(TokenPos),
    Continue(TokenPos),
    Ret(Option<Box<Exp>>),
}

impl Stmt {
    /// Each Stmt takes a BasicBlock as argument, and returns
    /// the next BasicBlock after this Stmt.
    /// The return type is an Option, because Stmt like Break,
    /// Continue and Ret terminates the current BasicBlock.
    /// The next BasicBlock may also change, because of If and
    /// While, which introduce branches.
    fn stmt(
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        function: Function,
        sym_tab: &mut SymbolTable,
        bb: BasicBlock,
    ) -> Option<BasicBlock> {
        match self {
            Stmt::Ret(ret_val) => {

                let ret_val = match ret_val {
                    Some(exp) => Some(exp.exp(raw_input, program, Some(function), sym_tab, Some(bb))),
                    None => None,
                };
                let func_data = program.func_mut(function);
                let ret = func_data.dfg_mut().new_value().ret(ret_val);
                func_data.layout_mut().bb_mut(bb).insts_mut().push_key_back(ret).unwrap();

                // Ret terminates current BasicBlock
                None
            },
            _ => todo!(),
        }
    }
}

#[derive(Debug)]
pub enum Exp {
    LOrExp(Box<LOrExp>, TokenPos),
}

/// Checks that `expected` matches `found`, or report error.
fn check_value_type(raw_input: &[u8], expected: Type, found: Type, pos: &TokenPos) {
    if expected != found {
        semantic_error(raw_input, pos.0, pos.1, 
            &SemanticError::TypeMismatch(expected, found));
    }
}

/// Convert `value` it into an i32, if possible.
/// `UNDEF` is defaulted to 0, but will return also `true` as
/// an indication.
fn to_i32(data: &ValueData) -> Option<(i32, bool)> {
    match data.kind() {
        ValueKind::Integer(i) => Some((i.value(), false)),
        ValueKind::ZeroInit(_) => Some((0, false)),
        ValueKind::Undef(_) => Some((0, true)),
        _ => None
    }
}

fn binary_op_eval(op: &BinaryOp, lvalue: i32, rvalue: i32) -> i32 {
    match op {
        BinaryOp::Or => (lvalue != 0 || rvalue != 0) as i32,
        BinaryOp::And => (lvalue != 0 && rvalue != 0) as i32,
        BinaryOp::Eq => (lvalue == rvalue) as i32,
        BinaryOp::NotEq => (lvalue != rvalue) as i32,
        BinaryOp::Lt => (lvalue < rvalue) as i32,
        BinaryOp::Gt => (lvalue > rvalue) as i32,
        BinaryOp::Le => (lvalue <= rvalue) as i32,
        BinaryOp::Ge => (lvalue >= rvalue) as i32,
        BinaryOp::Mul => i32::wrapping_mul(lvalue, rvalue),
        BinaryOp::Div => i32::wrapping_div(lvalue, rvalue),
        BinaryOp::Mod => i32::wrapping_rem(lvalue, rvalue),
        BinaryOp::Add => i32::wrapping_add(lvalue, rvalue),
        BinaryOp::Sub => i32::wrapping_sub(lvalue, rvalue),
        _ => unimplemented!(),
    }
}


/// Common code for handling binary expressions.
fn binary_exp(
    args: &(BinaryOp, Value, Value, TokenPos, TokenPos, TokenPos),
    raw_input: &[u8],
    program: &mut Program,
    function: Option<Function>, 
    bb: Option<BasicBlock>,
) -> Value {
    let (op, lexp, rexp, lpos, rpos, pos) = args;
    if let Some(func) = function {
        let func_data = program.func(func);
        let lexp_data = func_data.dfg().value(*lexp);
        let rexp_data = func_data.dfg().value(*rexp);
        // Check for types
        check_value_type(raw_input, Type::get_i32(), lexp_data.ty().clone(), lpos);
        check_value_type(raw_input, Type::get_i32(), rexp_data.ty().clone(), rpos);

        // Try evaluate constantly
        let lvalue = to_i32(lexp_data);
        let rvalue = to_i32(rexp_data);

        let exp = if lvalue.is_some() && rvalue.is_some() {
            let lvalue = lvalue.unwrap();
            let rvalue = rvalue.unwrap();
            if lvalue.1 {
                warning(raw_input, lpos.0, lpos.1, &Warning::ValueUndef)
            }
            if rvalue.1 {
                warning(raw_input, rpos.0, rpos.1, &Warning::ValueUndef)
            }

            program.func_mut(func).dfg_mut().new_value()
                .integer(binary_op_eval(op, lvalue.0, rvalue.0))
        }
        else {
            let inst = program.func_mut(func).dfg_mut().new_value().binary(*op, *lexp, *rexp);
            let bb = bb.unwrap();
            program.func_mut(func).layout_mut().bb_mut(bb).insts_mut().push_key_back(inst).unwrap();
            inst
        };
        exp
    }
    else {
        let lexp_data = program.borrow_value(*lexp);
        let rexp_data = program.borrow_value(*rexp);
        // Check for types
        check_value_type(raw_input, Type::get_i32(), lexp_data.ty().clone(), lpos);
        check_value_type(raw_input, Type::get_i32(), rexp_data.ty().clone(), rpos);

        // When not inside a function, must then be a constant
        let lvalue = to_i32(&lexp_data);
        let rvalue = to_i32(&rexp_data);
        drop(lexp_data);
        drop(rexp_data);

        let exp = if lvalue.is_some() && rvalue.is_some() {
            let lvalue = lvalue.unwrap();
            let rvalue = rvalue.unwrap();
            if lvalue.1 {
                warning(raw_input, lpos.0, lpos.1, &Warning::ValueUndef)
            }
            if rvalue.1 {
                warning(raw_input, rpos.0, rpos.1, &Warning::ValueUndef)
            }

            program.new_value()
                .integer(binary_op_eval(op, lvalue.0, rvalue.0))
        }
        else {
            semantic_error(raw_input, pos.0, pos.1, &SemanticError::ConstExpected)
        };
        exp
    }
}


impl Exp {

    fn token_pos(&self) -> TokenPos {
        match self {
            Exp::LOrExp(_, pos) => *pos,
        }
    }

    /// Semantically parses this Exp, adding the Value into the
    /// Program (or FunctionData), and returns the Value handle.
    fn exp(
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>, // May not be inside a function
        sym_tab: &mut SymbolTable,
        bb: Option<BasicBlock>, // May not be inside a function
    ) -> Value {
        match self {
            Exp::LOrExp(lor_exp, _) => lor_exp.lor_exp(raw_input, program, function, sym_tab, bb)
        }
    }
}

impl LOrExp {

    fn token_pos(&self) -> TokenPos {
        match self {
            LOrExp::LAndExp(land_exp) => land_exp.token_pos(),
            LOrExp::Or(_,_,pos) => *pos,
        }
    }

    fn lor_exp (
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>, 
        sym_tab: &mut SymbolTable,
        bb: Option<BasicBlock>,
    ) -> Value {
        match self {
            LOrExp::LAndExp(land_exp) => land_exp.land_exp(raw_input, program, function, sym_tab, bb),
            LOrExp::Or(exp, more, pos) => {
                let lexp = exp.lor_exp(raw_input, program, function, sym_tab, bb);
                let rexp = more.land_exp(raw_input, program, function, sym_tab, bb);

                let args = (BinaryOp::Or, lexp, rexp, exp.token_pos(), more.token_pos(), *pos);
                binary_exp(&args, raw_input, program, function, bb)
            },
        }
    }
}

impl LAndExp {

    fn token_pos(&self) -> TokenPos {
        match self {
            LAndExp::EqExp(eq_exp) => eq_exp.token_pos(),
            LAndExp::And(_,_,pos) => *pos,
        }
    }

    fn land_exp (
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>, 
        sym_tab: &mut SymbolTable,
        bb: Option<BasicBlock>,
    ) -> Value {
        match self {
            LAndExp::EqExp(eq_exp) => eq_exp.eq_exp(raw_input, program, function, sym_tab, bb),
            LAndExp::And(exp, more, pos) => {
                let lexp = exp.land_exp(raw_input, program, function, sym_tab, bb);
                let rexp = more.eq_exp(raw_input, program, function, sym_tab, bb);

                let args = (BinaryOp::And, lexp, rexp, exp.token_pos(), more.token_pos(), *pos);
                binary_exp(&args, raw_input, program, function, bb)
            },
        }
    }
}

impl EqExp {

    fn token_pos(&self) -> TokenPos {
        match self {
            EqExp::RelExp(rel_exp) => rel_exp.token_pos(),
            EqExp::Eq(_,_,pos) | EqExp::Neq(_,_,pos) => *pos,
        }
    }

    fn eq_exp (
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>, 
        sym_tab: &mut SymbolTable,
        bb: Option<BasicBlock>,
    ) -> Value {
        match self {
            EqExp::RelExp(rel_exp) => rel_exp.rel_exp(raw_input, program, function, sym_tab, bb),
            EqExp::Eq(exp, more, pos) |
            EqExp::Neq(exp, more, pos) => {
                let lexp = exp.eq_exp(raw_input, program, function, sym_tab, bb);
                let rexp = more.rel_exp(raw_input, program, function, sym_tab, bb);

                let op = match self {
                    EqExp::Eq(_,_,_) => BinaryOp::Eq,
                    EqExp::Neq(_,_,_) => BinaryOp::NotEq,
                    _ => unreachable!(),
                };
                let args = (op, lexp, rexp, exp.token_pos(), more.token_pos(), *pos);
                binary_exp(&args, raw_input, program, function, bb)
            },
        }
    }

}

impl RelExp {

    fn token_pos(&self) -> TokenPos {
        match self {
            RelExp::AddExp(add_exp) => add_exp.token_pos(),
            RelExp::Le(_,_,pos) | RelExp::Ge(_,_,pos) |
            RelExp::Gt(_,_,pos) | RelExp::Lt(_,_,pos) => *pos,
        }
    }

    fn rel_exp (
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>, 
        sym_tab: &mut SymbolTable,
        bb: Option<BasicBlock>,
    ) -> Value {
        match self {
            RelExp::AddExp(add_exp) => add_exp.add_exp(raw_input, program, function, sym_tab, bb),
            RelExp::Lt(exp, more, pos) |
            RelExp::Gt(exp, more, pos) |
            RelExp::Le(exp, more, pos) |
            RelExp::Ge(exp, more, pos) => {
                let lexp = exp.rel_exp(raw_input, program, function, sym_tab, bb);
                let rexp = more.add_exp(raw_input, program, function, sym_tab, bb);

                let op = match self {
                    RelExp::Lt(_,_,_) => BinaryOp::Lt,
                    RelExp::Gt(_,_,_) => BinaryOp::Gt,
                    RelExp::Le(_,_,_) => BinaryOp::Le,
                    RelExp::Ge(_,_,_) => BinaryOp::Ge,
                    _ => unreachable!(),
                };
                let args = (op, lexp, rexp, exp.token_pos(), more.token_pos(), *pos);
                binary_exp(&args, raw_input, program, function, bb)
            },
        }
    }

}

impl AddExp {

    fn token_pos(&self) -> TokenPos {
        match self {
            AddExp::MulExp(mul_exp) => mul_exp.token_pos(),
            AddExp::Add(_,_,pos) | AddExp::Sub(_,_,pos) => *pos,
        }
    }

    fn add_exp (
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>, 
        sym_tab: &mut SymbolTable,
        bb: Option<BasicBlock>,
    ) -> Value {
        match self {
            AddExp::MulExp(mul_exp) => mul_exp.mul_exp(raw_input, program, function, sym_tab, bb),
            AddExp::Add(exp, more, pos) |
            AddExp::Sub(exp, more, pos) => {
                let lexp = exp.add_exp(raw_input, program, function, sym_tab, bb);
                let rexp = more.mul_exp(raw_input, program, function, sym_tab, bb);

                let op = match self {
                    AddExp::Add(_,_,_) => BinaryOp::Add,
                    AddExp::Sub(_,_,_) => BinaryOp::Sub,
                    _ => unreachable!(),
                };
                let args = (op, lexp, rexp, exp.token_pos(), more.token_pos(), *pos);
                binary_exp(&args, raw_input, program, function, bb)
            },
        }
    }

}

impl MulExp {

    fn token_pos(&self) -> TokenPos {
        match self {
            MulExp::UnaryExp(unary_exp) => unary_exp.token_pos(),
            MulExp::Mul(_,_,pos) | MulExp::Div(_,_,pos) | MulExp::Rem(_,_,pos) => *pos,
        }
    }

    fn mul_exp (
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>, 
        sym_tab: &mut SymbolTable,
        bb: Option<BasicBlock>,
    ) -> Value {
        match self {
            MulExp::UnaryExp(unary_exp) => unary_exp.unary_exp(raw_input, program, function, sym_tab, bb),
            MulExp::Mul(exp, more, pos) |
            MulExp::Div(exp, more, pos) |
            MulExp::Rem(exp, more, pos) => {
                let lexp = exp.mul_exp(raw_input, program, function, sym_tab, bb);
                let rexp = more.unary_exp(raw_input, program, function, sym_tab, bb);

                let op = match self {
                    MulExp::Mul(_,_,_) => BinaryOp::Mul,
                    MulExp::Div(_,_,_) => BinaryOp::Div,
                    MulExp::Rem(_,_,_) => BinaryOp::Mod,
                    _ => unreachable!(),
                };
                let args = (op, lexp, rexp, exp.token_pos(), more.token_pos(), *pos);
                binary_exp(&args, raw_input, program, function, bb)
            },
        }
    }

}

impl UnaryExp {

    fn token_pos(&self) -> TokenPos {
        match self {
            UnaryExp::PrimaryExp(_, pos) |
            UnaryExp::FuncCall(_,_,pos) |
            UnaryExp::Op(_,_,pos) => *pos,
        }
    }

    fn unary_exp (
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>, 
        sym_tab: &mut SymbolTable,
        bb: Option<BasicBlock>,
    ) -> Value {
        // program.func_mut(function.unwrap()).dfg_mut().new_value().integer(2)
        match self {
            UnaryExp::PrimaryExp(pexp, _) => pexp.primary_exp(raw_input, program, function, sym_tab, bb),
            UnaryExp::Op(op, more, pos) => {
                match op {
                    UnaryOp::Plus => more.unary_exp(raw_input, program, function, sym_tab, bb),
                    UnaryOp::Minus => {
                        let lexp = if let Some(func) = function {
                            program.func_mut(func).dfg_mut().new_value().integer(0)
                        }
                        else {
                            program.new_value().integer(0)
                        };
                        let rexp = more.unary_exp(raw_input, program, function, sym_tab, bb);
                        let args = (BinaryOp::Sub, lexp, rexp, (0, 0), more.token_pos(), *pos);
                        binary_exp(&args, raw_input, program, function, bb)
                    },
                    UnaryOp::Not => {
                        let lexp = if let Some(func) = function {
                            program.func_mut(func).dfg_mut().new_value().integer(0)
                        }
                        else {
                            program.new_value().integer(0)
                        };
                        let rexp = more.unary_exp(raw_input, program, function, sym_tab, bb);
                        let args = (BinaryOp::NotEq, lexp, rexp, (0, 0), more.token_pos(), *pos);
                        binary_exp(&args, raw_input, program, function, bb)
                    }
                }
            },
            UnaryExp::FuncCall(_,_,_) => todo!(),
        }
    }

}

impl PrimaryExp {
    fn primary_exp (
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>, 
        sym_tab: &mut SymbolTable,
        bb: Option<BasicBlock>,
    ) -> Value {
        match self {
            PrimaryExp::Bracketed(exp) => exp.exp(raw_input, program, function, sym_tab, bb),
            PrimaryExp::Number(number) => {
                match number {
                    Number::IntConst(i) => {
                        if let Some(func) = function {
                            program.func_mut(func).dfg_mut().new_value().integer(*i)
                        }
                        else {
                            program.new_value().integer(*i)
                        }
                    },
                }
            },
            PrimaryExp::LVal(lval) => lval.lval(raw_input, program, function, sym_tab, bb),
        }
    }
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

    fn lval (
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>, 
        sym_tab: &mut SymbolTable,
        bb: Option<BasicBlock>,
    ) -> Value {

        // Check symbol table
        let sym = sym_tab.get_ident(&self.ident);
        if let None = sym {
            semantic_error(raw_input, self.token_pos.0, self.token_pos.1, &SemanticError::UndefinedIdent)
        }

        let (sym, sym_id) = sym.unwrap();
        match sym {
            Symbol::Function(_) => semantic_error(raw_input, self.token_pos.0, self.token_pos.1, &SemanticError::MisuseOfFuncIdent(sym_id)),
            Symbol::Value(lval, mut is_const) => {

                // Then check the indices over the type
                if let Some(func) = function {

                    // Inside a function

                    let lval_data = program.func(func).dfg().value(lval);
                    
                    let mut indexed_lval = lval.clone();
                    let mut insts = Vec::new();
                    let mut ty = lval_data.ty().clone();
                    if let TypeKind::Pointer(btype) = ty.kind() {
                        let btype = btype.clone();
                        ty = btype.clone();
                        if let TypeKind::Pointer(_) = btype.kind() {
                            let load = program.func_mut(func).dfg_mut().new_value()
                                    .load(lval);
                            insts.push(load);
                        }
                    }
                    else {
                        panic!("[AST] Type of any ident should always be pointer")
                    }

                    for exp in self.indices.iter() {
                        let index = exp.exp(raw_input, program, function, sym_tab, bb);
                        let index_data = program.func(func).dfg().value(index);
                        check_value_type(raw_input, Type::get_i32(), index_data.ty().clone(), &exp.token_pos());

                        match ty.kind() {
                            TypeKind::Pointer(btype) => {
                                let mut is_zero = false;
                                if let Some((index, undef)) = to_i32(index_data) {
                                    let pos = exp.token_pos();
                                    if undef {
                                        warning(raw_input, pos.0, pos.1, &Warning::ValueUndef)
                                    }
                                    is_zero = index == 0;
                                }
                                ty = btype.clone();
                                let locate = if is_zero {
                                    *insts.last().unwrap_or(&lval)
                                }
                                else {
                                    let locate = program.func_mut(func).dfg_mut().new_value()
                                        .get_ptr(*insts.last().unwrap_or(&lval), index);
                                    insts.push(locate);
                                    locate
                                };
                                let load = program.func_mut(func).dfg_mut().new_value()
                                    .load(locate);
                                insts.push(load);
                                is_const = false;

                            },
                            TypeKind::Array(btype, bound) => {
                                if let Some((index, undef)) = to_i32(index_data) {
                                    let pos = exp.token_pos();
                                    if undef {
                                        warning(raw_input, pos.0, pos.1, &Warning::ValueUndef)
                                    }
                                    if index >= *bound as i32 || index < 0 {
                                        warning(raw_input, pos.0, pos.1, 
                                            &Warning::IndexOutOfBound(index, sym_id.token_pos, *bound))
                                    }
                                    if is_const {
                                        if let ValueKind::Aggregate(agg) 
                                            = program.func(func).dfg().value(indexed_lval).kind() 
                                        {
                                            indexed_lval = agg.elems().get(index as usize).unwrap().clone();
                                        }
                                    }
                                }
                                else {
                                    is_const = false;
                                }
                                ty = btype.clone();
                                let inst = program.func_mut(func).dfg_mut().new_value()
                                    .get_elem_ptr(*insts.last().unwrap_or(&lval), index);
                                insts.push(inst);
                            },
                            _ => {
                                let btype = program.func(func).dfg().value(lval).ty().clone();
                                let extra_pos = exp.token_pos();
                                semantic_error(raw_input, extra_pos.0, extra_pos.1,
                                    &SemanticError::ExtraIndex(sym_id.token_pos, btype))
                            }
                        }
                    }
                    let indexed_lval_data = program.func(func).dfg().value(indexed_lval);
                    if is_const {
                        // Can replace with a constant value
                        assert!(indexed_lval_data.kind().is_const());
                        return if let Some((i, undef)) = to_i32(indexed_lval_data) {
                            if undef {
                                warning(raw_input, self.token_pos.0, self.token_pos.1, &Warning::ValueUndef)
                            }
                            program.func_mut(func).dfg_mut().new_value().integer(i)
                        }
                        else {
                            if let ValueKind::Aggregate(agg) = indexed_lval_data.kind() {
                                let elems = Vec::from(agg.elems());
                                program.func_mut(func).dfg_mut().new_value().aggregate(elems)
                            }
                            else {
                                unreachable!()
                            }
                        }
                        
                    }
                    else {
                        // Must calculate with instructions
                        let last = insts.last().unwrap_or(&lval);
                        if !matches!(program.func(func).dfg().value(*last).kind(), ValueKind::Load(_)) {
                            let load = program.func_mut(func).dfg_mut().new_value().load(*last);
                            insts.push(load);
                        }
                        let last = insts.last().unwrap().clone();
                        program.func_mut(func).layout_mut().bb_mut(bb.unwrap()).insts_mut().extend(insts);
                        return last;
                    }
                }
                else {

                    // Global scope

                    let lval_data = program.borrow_value(lval);
                    let mut ty = lval_data.ty().clone();
                    let mut indexed_lval = lval.clone();
                    if let TypeKind::Pointer(btype) = ty.kind() {
                        ty = btype.clone();
                    }
                    else {
                        panic!("[AST] Type of any ident should always be pointer")
                    }

                    // We start with `is_const` as true, because in global scope,
                    // even non-const identifiers.
                    drop(lval_data);

                    for exp in self.indices.iter() {
                        let index = exp.exp(raw_input, program, function, sym_tab, bb);
                        let index_data = program.borrow_value(index);
                        check_value_type(raw_input, Type::get_i32(), index_data.ty().clone(), &exp.token_pos());

                        match ty.kind() {
                            TypeKind::Pointer(_) => unreachable!(),
                            TypeKind::Array(btype, bound) => {
                                if let Some((index, undef)) = to_i32(&index_data) {
                                    let pos = exp.token_pos();
                                    if undef {
                                        warning(raw_input, pos.0, pos.1, &Warning::ValueUndef)
                                    }
                                    if index >= *bound as i32 || index < 0 {
                                        warning(raw_input, pos.0, pos.1, 
                                            &Warning::IndexOutOfBound(index, sym_id.token_pos, *bound))
                                    }
                                    if let ValueKind::Aggregate(agg) 
                                        = program.borrow_value(indexed_lval).kind() 
                                    {
                                        indexed_lval = agg.elems().get(index as usize).unwrap().clone();
                                    }
                                }
                                else {
                                    // Oops
                                    let pos = exp.token_pos();
                                    semantic_error(raw_input, pos.0, pos.1,
                                        &SemanticError::ConstExpected)
                                }
                                ty = btype.clone();
                            },
                            _ => {
                                let btype = program.borrow_value(lval).ty().clone();
                                let extra_pos = exp.token_pos();
                                semantic_error(raw_input, extra_pos.0, extra_pos.1,
                                    &SemanticError::ExtraIndex(sym_id.token_pos, btype))
                            }
                        }
                    }

                    let indexed_lval_data = program.borrow_value(indexed_lval);
                    // Can replace with a constant value
                    assert!(indexed_lval_data.kind().is_const());
                    return if let Some((i, undef)) = to_i32(&indexed_lval_data) {
                        drop(indexed_lval_data);
                        if undef {
                            warning(raw_input, self.token_pos.0, self.token_pos.1, &Warning::ValueUndef)
                        }
                        program.new_value().integer(i)
                    }
                    else {
                        if let ValueKind::Aggregate(agg) = indexed_lval_data.kind() {
                            let elems = Vec::from(agg.elems());
                            drop(indexed_lval_data);
                            program.new_value().aggregate(elems)
                        }
                        else {
                            unreachable!()
                        }
                    }
                }
            },
        }
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

impl ConstExp {
    fn const_exp(
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>, 
        sym_tab: &mut SymbolTable,
        bb: Option<BasicBlock>, 
    ) -> Value {
        let (exp, pos) = (self.exp.exp(raw_input, program, function, sym_tab, bb), self.pos);
        let is_const = if let Some(func) = function {
            program.func(func).dfg().value(exp).kind().is_const()
        }
        else {
            program.borrow_value(exp).kind().is_const()
        };
        if !is_const {
            semantic_error(raw_input, pos.0, pos.1,
                &SemanticError::ConstExpected)
        }
        exp
    }
}

pub type TokenPos = (usize, usize);

#[derive(Debug, Clone)]
pub struct Ident {
    pub ident: String,
    pub token_pos: TokenPos,
}

impl Ident {
    fn to_string(&self) -> &String {
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
