//! Implements the AST structures for SYSY.

#[allow(unused_imports)]
use koopa::ir::*;
use koopa::ir::builder_traits::*;
use koopa::ir::entities::ValueData;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use crate::error::{semantic_error, SemanticError, warning, Warning};
use crate::middleend::{SymbolTable, Symbol};

#[derive(Debug)]
pub enum CompUnit {
    Decl(Option<Box<CompUnit>>, Box<Decl>),
    FuncDef(Option<Box<CompUnit>>, Box<FuncDef>),
}

impl CompUnit {

    pub fn register_decls(&mut self, raw_input: &[u8], program: &mut Program, sym_tab: &mut SymbolTable) {
        match self {
            CompUnit::Decl(more, decl) => {
                if let Some(more) =  more {
                    more.register_decls(raw_input, program, sym_tab);
                }
                let mut no_bb = None;
                decl.decl(raw_input, program, None, sym_tab, &mut no_bb);
            },
            CompUnit::FuncDef(more, func_def) => {
                if let Some(more) =  more {
                    more.register_decls(raw_input, program, sym_tab);
                }
                func_def.func_decl(raw_input, program, sym_tab);
            }
        }
    }

    pub fn append_to_program(self, raw_input: &[u8], program: &mut Program, sym_tab: &mut SymbolTable) {
        match self {
            CompUnit::Decl(more, _) => {
                if let Some(more) =  more {
                    more.append_to_program(raw_input, program, sym_tab);
                }
                // decls are already handled
            },
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
        bb: &mut Option<BasicBlock>,
    ) {
        match self {
            Decl::ConstDecl(const_decl) => const_decl.const_decl(raw_input, program, function, sym_tab, bb),
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
        raw_input: &[u8], 
        program: &mut Program,
        function: Option<Function>,
        sym_tab: &mut SymbolTable,
        bb: &mut Option<BasicBlock>, 
    ) {
        for def in self.const_def.iter() {
            def.const_def(raw_input, program, function, sym_tab, bb, self.btype);
        }
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

    fn const_def(
        &self, // Todo: may fix this?
        raw_input: &[u8], 
        program: &mut Program,
        function: Option<Function>,
        sym_tab: &mut SymbolTable,
        bb: &mut Option<BasicBlock>, 
        btype: BType,
    ) {
        if self.const_init_val.is_none() {
            semantic_error(raw_input, self.ident.token_pos.0, self.ident.token_pos.1,
                &SemanticError::ConstUninit)
        }
        let args = (&self.ident, &self.const_exp, &self.const_init_val.clone().map(|x| x as Rc<dyn IsInitVal>));
        def(args, raw_input, program, function, sym_tab, bb, btype, true);
    }
}

#[derive(Debug)]
pub enum ConstInitVal {
    ConstExp(Box<ConstExp>),
    ValList(Vec<Rc<ConstInitVal>>),
}

impl ConstInitVal {
    fn const_init_val(
        &self,
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>,
        sym_tab: &mut SymbolTable,
        bb: &mut Option<BasicBlock>,
        alloc: Value,
    ) -> Option<Value> {
        match self {
            ConstInitVal::ConstExp(init_exp) => {
                let exp = init_exp.const_exp(raw_input, program, function, sym_tab, bb);
                let args = (exp, &init_exp.token_pos());
                init_single(args, raw_input, program, function, bb, alloc, true)
            },
            ConstInitVal::ValList(_) => todo!(),
        }
    }
}

impl IsInitVal for ConstInitVal {
    fn init(
        &self,
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>,
        sym_tab: &mut SymbolTable,
        bb: &mut Option<BasicBlock>,
        alloc: Value,
        _is_const: bool,
    ) -> Option<Value> {
        self.const_init_val(raw_input, program, function, sym_tab, bb, alloc)
    }
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
        bb: &mut Option<BasicBlock>,
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
    pub init_val: Option<Rc<InitVal>>,
}

/// Common code for handling an identifier definition.
fn def(
    args: (&Ident, &Vec<Box<ConstExp>>, &Option<Rc<dyn IsInitVal>>),
    raw_input: &[u8], 
    program: &mut Program,
    function: Option<Function>,
    sym_tab: &mut SymbolTable,
    bb: &mut Option<BasicBlock>,
    btype: BType,
    is_const: bool, 
) {
    let (ident, const_exp, init_val) = args;
    // Must first evaluate the actual type of this definition
    let def = if let Some(func) = function {

        // Inside a function

        let mut ty = match btype {
            BType::Int => Type::get_i32(),
        };
        for exp in const_exp.iter().rev() {
            // `const_exp` already checks if the Exp is constant
            let index = exp.const_exp(raw_input, program, function, sym_tab, bb);
            let index_data = program.func(func).dfg().value(index);
            check_value_type(raw_input, &Type::get_i32(), index_data.ty(), &exp.pos);
            let (bound, _) = to_i32(index_data).unwrap();
            if bound < 0 {
                let pos = exp.token_pos();
                semantic_error(raw_input, pos.0, pos.1,
                    &SemanticError::NegArrayBound(bound))
            }
            ty = Type::get_array(ty, bound as usize);
        }

        if is_const {
            // Local const is handled as a global alloc.
            // For now, init as undef.
            let undef = program.new_value().undef(ty);
            let alloc = program.new_value().global_alloc(undef);
            alloc
        }
        else {
            // Do a local alloc
            let alloc = program.func_mut(func).dfg_mut().new_value().alloc(ty);
            program.func_mut(func).layout_mut().bb_mut(bb.unwrap()).insts_mut().push_key_back(alloc).unwrap();
            alloc
        }
    }
    else {

        // Global scope
        let mut ty = match btype {
            BType::Int => Type::get_i32(),
        };
        for exp in const_exp.iter().rev() {
            let mut no_bb = None;
            let index = exp.const_exp(raw_input, program, function, sym_tab, &mut no_bb);
            let index_data = &program.borrow_value(index);
            check_value_type(raw_input, &Type::get_i32(), index_data.ty(), &exp.pos);
            let (bound, _) = to_i32(index_data).unwrap();
            ty = Type::get_array(ty, bound as usize);
        }

        // For now, init as undef.
        let undef = program.new_value().undef(ty);
        let alloc = program.new_value().global_alloc(undef);
        alloc
    };

    // Add to symbol table; may err
    sym_tab.add(raw_input, ident, Symbol::Value(def, is_const, init_val.is_some(), 
        function.is_none() || is_const /* Either a global symbol, or a local const symbol */));

    // Then handle the init value; this order is consistent with C.
    // For example:
    // const int x = 1;
    // {
    //    const int x[x+1] = {x};
    // }
    // the array will be initialized with the address of itself, rather than 1.
    if let Some(init_val) = init_val {
        let init = init_val.init(raw_input, program, function, sym_tab, bb, def, 
            is_const || function.is_none() /* Global init also requires const init val */);
        if let Some(init) = init {
            // Replace the `init` value of `def`
            program.remove_value(def);
            let alloc = program.new_value().global_alloc(init);
            _ = sym_tab.replace(ident, Symbol::Value(alloc, is_const, true, true));
        }
        // Otherwise, `init()` already generated instructions for the initialization.
    }
    // Otherwise the value is left as undef.
}

impl VarDef {
    pub fn new(ident: Ident, const_exp: Vec<Box<ConstExp>>, init_val: Option<Rc<InitVal>>) -> Self {
        VarDef { ident, const_exp, init_val }
    }

    fn var_def(
        &self, 
        raw_input: &[u8], 
        program: &mut Program,
        function: Option<Function>,
        sym_tab: &mut SymbolTable,
        bb: &mut Option<BasicBlock>,
        btype: BType,
    ) {
        let init_val = self.init_val.clone().map(|x| x as Rc<dyn IsInitVal>);
        let args = (&self.ident, &self.const_exp, &init_val);
        def(args, raw_input, program, function, sym_tab, bb, btype,false);
    }
}

#[derive(Debug)]
pub enum InitVal {
    Exp(Box<Exp>),
    ValList(Vec<Rc<InitVal>>),
}

/// Common code for initializing with a single Exp.
fn init_single(
    args: (Value, &TokenPos),
    raw_input: &[u8],
    program: &mut Program,
    function: Option<Function>,
    bb: &mut Option<BasicBlock>,
    alloc: Value,
    is_const: bool,
) -> Option<Value> {
    let (exp, pos) = args;

    let init_ty_origin = if let Some(func) = function {
        if !is_const {
            program.func(func).dfg().value(alloc).ty().clone()
        }
        else {
            program.borrow_value(alloc).ty().clone()
        }
    }
    else {
        program.borrow_value(alloc).ty().clone()
    };
    let mut init_ty: &Type = &init_ty_origin;

    let exp_ty = if let Some(func) = function { 
        program.func(func).dfg().value(exp).ty().clone()
    }
    else {
        program.borrow_value(exp).ty().clone()
    };

    if let TypeKind::Pointer(ty) = init_ty.kind() {
        // `alloc` should be a pointer
        init_ty = ty;
    }
    else {
        unreachable!();
    }

    check_value_type(raw_input, init_ty, &exp_ty, pos);

    if is_const {
        if let Some(func) = function {
            // A local const value is still stored in global
            // data section, and needs an initializer.
            let exp_data = program.func(func).dfg().value(exp);
            if let Some((i, _)) = to_i32(exp_data) {
                // Add to global scope because eventually it is stored there.
                return Some(program.new_value().integer(i))
            }
        }
        else {
            let exp_data = program.borrow_value(exp);
            if let Some((i, _)) = to_i32(&exp_data) {
                drop(exp_data);
                return Some(program.new_value().integer(i))
            }
        }
        semantic_error(raw_input, pos.0, pos.1,
            &SemanticError::ConstExpected)
    }

    // Must be local variable init; generate a store
    if let Some(func) = function {
        let init = program.func_mut(func).dfg_mut().new_value()
            .store(exp, alloc);
        program.func_mut(func).layout_mut().bb_mut(bb.unwrap())
            .insts_mut().push_key_back(init).unwrap();
        None
    }
    else {
        unreachable!()
    }
}

trait IsInitVal {
    fn init(
        &self,
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>,
        sym_tab: &mut SymbolTable,
        bb: &mut Option<BasicBlock>,
        alloc: Value,
        is_const: bool,
    ) -> Option<Value>;
}

impl InitVal {

    /// Returns Some(`init`) in case of `is_const` init, or None if 
    /// otherwise.
    /// NOTE: `is_const` should be true for anything that is not a
    /// local variable init, which includes all const inits and all
    /// global inits.
    fn init_val(
        &self,
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>,
        sym_tab: &mut SymbolTable,
        bb: &mut Option<BasicBlock>,
        alloc: Value,
        is_const: bool,
    ) -> Option<Value> {
        match self {
            InitVal::Exp(init_exp) => {
                let exp = init_exp.exp(raw_input, program, function, sym_tab, bb, false);
                let args = (exp, &init_exp.token_pos());
                init_single(args, raw_input, program, function, bb, alloc, is_const)
            },
            InitVal::ValList(_) => todo!(),
        }
    }
}

impl IsInitVal for InitVal {
    fn init(
        &self,
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>,
        sym_tab: &mut SymbolTable,
        bb: &mut Option<BasicBlock>,
        alloc: Value,
        is_const: bool,
    ) -> Option<Value> {
        self.init_val(raw_input, program, function, sym_tab, bb, alloc, is_const)
    }
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
    
    fn func_def(self, raw_input: &[u8], program: &mut Program, sym_tab: &mut SymbolTable) {
        let params = &self.params;

        if let Some(Symbol::Function(func)) = sym_tab.get(&self.ident) {
            let new_func_data = program.func_mut(func);

            // Prepare entry block
            let entry = new_func_data.dfg_mut().new_bb().basic_block(Some("%entry".into()));
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
                new_func_data.layout_mut().bb_mut(entry).insts_mut().extend([addr, store]);

                // Add function param ident to symbol table; may error
                sym_tab.add(raw_input, &param, Symbol::Value(addr, false, true, false));
            }

            // Handle the function body
            let ret_ty = match self.func_type {
                FuncType::Int => Type::get_i32(),
                FuncType::Void => Type::get_unit(),
            };
            let mut bb = Some(entry);
            self.block.block(raw_input, program, func, sym_tab, &mut bb, &ret_ty);

            assert!(sym_tab.get_resume_bb().is_none());

            // Check for missing `return`
            // which boils down to checking whether `bb` is None.
            // If not, then some control flow is not terminated by
            // a `return` and is expecting further code.
            if !ret_ty.is_unit() {
                if bb.is_some() {
                    let bb = bb.unwrap();
                    // TODO: map from BB into Stmt that generates it,
                    // to report where the return is missing.
                    warning(raw_input, 0, 0,
                        &Warning::MissingRet(self.ident.token_pos));

                    let undef = program.func_mut(func).dfg_mut().new_value().undef(ret_ty);
                    let ret = program.func_mut(func).dfg_mut().new_value()
                        .ret(Some(undef));
                    program.func_mut(func).layout_mut().bb_mut(bb).insts_mut()
                        .push_key_back(ret).unwrap();
                }
            }
            
            // Remove useless basic blocks
            // TODO: beautify this
            let empty_bbs: Vec<BasicBlock> = program.func(func).layout().bbs().iter()
                .filter(|(_, data)| data.insts().is_empty())
                .map(|(bb,_) | *bb)
                .collect();
            
            for bb in empty_bbs.iter() {
                program.func_mut(func).layout_mut().bbs_mut().remove(bb);
            }

            // Out of the scope
            sym_tab.pop();
        }
        else {
            unreachable!()
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
                let mut no_bb = None;
                let index = const_exp.const_exp(raw_input, program, None, sym_tab, &mut no_bb);
                let index = program.borrow_value(index);
                check_value_type(raw_input, &Type::get_i32(), index.ty(), &const_exp.pos);
                let index_i32 = to_i32(&index);
                if let Some((i, _)) = index_i32 {
                    if i < 0 {
                        // `const_exp` must not be negative
                        let pos = const_exp.pos;
                        semantic_error(raw_input, pos.0, pos.1,
                            &SemanticError::NegArrayBound(i))
                    }
                    ptype = Type::get_array(ptype, i as usize);
                }
                else {
                    let pos = const_exp.pos;
                    semantic_error(raw_input, pos.0, pos.1,
                        &SemanticError::ConstExpected)
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
        bb: &mut Option<BasicBlock>,
        ret_ty: &Type,
    ) {
        sym_tab.push();
        for item in self.block_item.iter() {
            item.block_item(raw_input, program, function, sym_tab, bb, ret_ty);
            if bb.is_none() {
                // Terminating block item 
                break;
            }
        }
        sym_tab.pop();
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
        bb: &mut Option<BasicBlock>,
        ret_ty: &Type,
    ) {
        match self {
            BlockItem::Decl(decl) => decl.decl(raw_input, program, Some(function), sym_tab, bb),
            BlockItem::Stmt(stmt) => stmt.stmt(raw_input, program, function, sym_tab, bb, ret_ty),
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
    Ret(Option<Box<Exp>>, TokenPos),
}

impl Stmt {
    /// Each Stmt takes a BasicBlock as argument, and may mutate
    /// it as necessary.
    /// The type is an Option, because Stmt like Break,
    /// Continue and Ret terminates the current BasicBlock.
    /// The next BasicBlock may also change, because of If and
    /// While, which introduce branches.
    fn stmt(
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        function: Function,
        sym_tab: &mut SymbolTable,
        bb: &mut Option<BasicBlock>,
        ret_ty: &Type,
    ) {
        assert!(bb.is_some());
        match self {
            Stmt::Ret(ret_val, pos) => {
                let ret_val = match ret_val {
                    Some(exp) => {
                        let exp_val = exp.exp(raw_input, program, Some(function), sym_tab, bb, false);
                        // Check for return type
                        check_value_type(raw_input, ret_ty, 
                            program.func(function).dfg().value(exp_val).ty(),
                            &exp.token_pos()
                        );
                        Some(exp_val)
                    },
                    None => {
                        // Check for return type
                        check_value_type(raw_input, ret_ty, 
                            &Type::get_unit(),
                            pos
                        );
                        None
                    },
                };    
                let func_data = program.func_mut(function);
                let ret = func_data.dfg_mut().new_value().ret(ret_val);
                func_data.layout_mut().bb_mut(bb.unwrap()).insts_mut().push_key_back(ret).unwrap();

                // Ret terminates current BasicBlock
                *bb = None;
            },
            Stmt::Block(block) => {
                block.block(raw_input, program, function, sym_tab, bb, ret_ty);
            },
            Stmt::Exp(exp) => {
                if let Some(exp) = exp {
                    exp.exp(raw_input, program, Some(function), sym_tab, bb, false);
                }
            },
            Stmt::Assign(lval, exp) => {
                let rexp = exp.exp(raw_input, program, Some(function), sym_tab, bb, false);
                let lvalue = lval.lval(raw_input, program, Some(function), sym_tab, bb, true, false);

                let ldata = program.func(function).dfg().value(lvalue);
                let rdata = program.func(function).dfg().value(rexp);
                let mut lty = ldata.ty().clone();
                let rty = rdata.ty().clone();
                // `lty` should be a pointer
                match lty.kind() {
                    TypeKind::Array(btype,_) | TypeKind::Pointer(btype) => lty = btype.clone(),
                    _ => unreachable!(),
                }          
                check_value_type(raw_input, &lty, &rty, &exp.token_pos());

                // All clear; generate a store
                assert!(bb.is_some());
                let store = program.func_mut(function).dfg_mut()
                    .new_value().store(rexp, lvalue);
                program.func_mut(function).layout_mut().bb_mut(bb.unwrap())
                    .insts_mut().push_key_back(store).unwrap();
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
fn check_value_type(raw_input: &[u8], expected: &Type, found: &Type, pos: &TokenPos) {
    if expected != found {
        semantic_error(raw_input, pos.0, pos.1, 
            &SemanticError::TypeMismatch(expected.clone(), found.clone()));
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

fn binary_op_eval(op: &BinaryOp, lvalue: i32, rvalue: i32) -> (i32, bool) {
    match op {
        BinaryOp::Or => ((lvalue != 0 || rvalue != 0) as i32, false),
        BinaryOp::And => ((lvalue != 0 && rvalue != 0) as i32, false),
        BinaryOp::Eq => ((lvalue == rvalue) as i32, false),
        BinaryOp::NotEq => ((lvalue != rvalue) as i32, false),
        BinaryOp::Lt => ((lvalue < rvalue) as i32, false),
        BinaryOp::Gt => ((lvalue > rvalue) as i32, false),
        BinaryOp::Le => ((lvalue <= rvalue) as i32, false),
        BinaryOp::Ge => ((lvalue >= rvalue) as i32, false),
        BinaryOp::Mul => i32::overflowing_mul(lvalue, rvalue),
        BinaryOp::Div => i32::overflowing_div(lvalue, rvalue),
        BinaryOp::Mod => i32::overflowing_rem(lvalue, rvalue),
        BinaryOp::Add => i32::overflowing_add(lvalue, rvalue),
        BinaryOp::Sub => i32::overflowing_sub(lvalue, rvalue),
        _ => unimplemented!(),
    }
}


/// Common code for handling binary expressions.
fn binary_exp(
    args: &(BinaryOp, Value, Value, TokenPos, TokenPos, TokenPos),
    raw_input: &[u8],
    program: &mut Program,
    function: Option<Function>, 
    bb: &mut Option<BasicBlock>,
) -> Value {
    let (op, lexp, rexp, lpos, rpos, pos) = args;
    if let Some(func) = function {
        let func_data = program.func(func);
        let lexp_data = func_data.dfg().value(*lexp);
        let rexp_data = func_data.dfg().value(*rexp);
        // Check for types
        check_value_type(raw_input, &Type::get_i32(), lexp_data.ty(), lpos);
        check_value_type(raw_input, &Type::get_i32(), rexp_data.ty(), rpos);

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

            let (val, overflow) = binary_op_eval(op, lvalue.0, rvalue.0);
            if overflow {
                warning(raw_input, pos.0, pos.1, &Warning::IntegerOverflow)
            }
            program.func_mut(func).dfg_mut().new_value()
                .integer(val)
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
        check_value_type(raw_input, &Type::get_i32(), lexp_data.ty(), lpos);
        check_value_type(raw_input, &Type::get_i32(), rexp_data.ty(), rpos);

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

            let (val, overflow) = binary_op_eval(op, lvalue.0, rvalue.0);
            if overflow {
                warning(raw_input, pos.0, pos.1, &Warning::IntegerOverflow)
            }
            program.new_value()
                .integer(val)
        }
        else {
            semantic_error(raw_input, pos.0, pos.1, &SemanticError::ConstExpected)
        };
        exp
    }
}

/// Common code for handling boolean expressions.
fn boolean_exp(
    args: &(Value, TokenPos),
    raw_input: &[u8],
    program: &mut Program,
    function: Option<Function>, 
    bb: &mut Option<BasicBlock>,
) -> Value {
    let (exp, pos) = args;
    if let Some(func) = function {
        let func_data = program.func(func);
        let exp_data = func_data.dfg().value(*exp);

        // Check for types
        check_value_type(raw_input, &Type::get_i32(), exp_data.ty(), pos);

        // Try evaluate constantly
        let value = to_i32(exp_data);

        let evaled_exp = if value.is_some() {
            let value = value.unwrap();
            if value.1 {
                warning(raw_input, pos.0, pos.1, &Warning::ValueUndef)
            }
            program.func_mut(func).dfg_mut().new_value()
                .integer((value.0 != 0) as i32)
        }
        else {
            let zero = program.func_mut(func).dfg_mut().new_value().integer(0);
            let inst = program.func_mut(func).dfg_mut().new_value().binary(BinaryOp::NotEq, *exp, zero);
            let bb = bb.unwrap();
            program.func_mut(func).layout_mut().bb_mut(bb).insts_mut().push_key_back(inst).unwrap();
            inst
        };
        evaled_exp
    }
    else {
        let exp_data = program.borrow_value(*exp);
        // Check for types
        check_value_type(raw_input, &Type::get_i32(), exp_data.ty(), pos);

        // When not inside a function, must then be a constant
        let value = to_i32(&exp_data);
        drop(exp_data);

        let evaled_exp = if value.is_some() {
            let value = value.unwrap();
            if value.1 {
                warning(raw_input, pos.0, pos.1, &Warning::ValueUndef)
            }
            program.new_value()
                .integer((value.0 != 0) as i32)
        }
        else {
            semantic_error(raw_input, pos.0, pos.1, &SemanticError::ConstExpected)
        };
        evaled_exp
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
        bb: &mut Option<BasicBlock>, // May not be inside a function
        is_ptr: bool,
    ) -> Value {
        match self {
            Exp::LOrExp(lor_exp, _) => lor_exp.lor_exp(raw_input, program, function, sym_tab, bb, is_ptr)
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
        bb: &mut Option<BasicBlock>,
        is_ptr: bool,
    ) -> Value {
        match self {
            LOrExp::LAndExp(land_exp) => land_exp.land_exp(raw_input, program, function, sym_tab, bb, is_ptr),
            LOrExp::Or(exp, more, pos) => {

                // Short-circuit may be considered.
                //   %a = ...
                //   br %a, done, calc_R
                // or_R:
                //   %or = %a || ...
                // done:

                if let Some(func) = function {

                    // Fucntion scope
                    
                    // Intermediate block
                    let or_r = program.func_mut(func).dfg_mut().new_bb()
                        .basic_block(Some("%or_R".into()));
                    
                    // Left exp
                    let lexp = exp.lor_exp(raw_input, program, function, sym_tab, bb, is_ptr);
                    let lexp = boolean_exp(&(lexp, exp.token_pos()), raw_input, program, function, bb);

                    // Right exp
                    program.func_mut(func).layout_mut().bbs_mut().push_key_back(or_r).unwrap();
                    let or_l = bb.replace(or_r).unwrap();
                    let rexp = more.land_exp(raw_input, program, function, sym_tab, bb, is_ptr);
                    let rexp = boolean_exp(&(rexp, exp.token_pos()), raw_input, program, function, bb);

                    // Some of the blocks may be stripped.

                    let mut lval = None;
                    let mut rval = None;

                    if let Some((lvalue, _)) = to_i32(program.func(func).dfg().value(lexp)) {
                        // If `lexp` is constant?
                        if lvalue == 1 {
                            // Short-circuited; contents of %or_R should be removed.
                            program.func_mut(func).layout_mut().bb_mut(or_r).insts_mut().clear();
                        }
                        lval = Some(lvalue);
                    }
                    if let Some ((rvalue, _)) = to_i32(program.func(func).dfg().value(rexp)) {
                        // If `rexp` is constant?
                        rval = Some(rvalue);
                    }

                    let mut or_done = program.func_mut(func).dfg_mut().new_bb()
                        .basic_block(Some("%or_done".into()));
                    
                    let or = match (lval, rval) {
                        (Some(1), _) | (_, Some(1)) => {
                            // In the latter case, the code to generate `lexp` , if existent,
                            // is not cleared here, because they may have side effect.
                            program.func_mut(func).dfg_mut().new_value()
                                .integer(1)
                        },
                        (Some(0), Some(0)) => {
                            program.func_mut(func).dfg_mut().new_value()
                                .integer(0)
                        },
                        (None, Some(0)) => {
                            lexp
                        },
                        (Some(0), None) => {
                            rexp
                        },
                        (None, None) => {
                            // Only here we generate a branch
                            // and use a parameterized basic block
                            program.func_mut(func).dfg_mut().remove_bb(or_done);
                            or_done = program.func_mut(func).dfg_mut().new_bb()
                                .basic_block_with_param_names(
                                    Some("%or_done".into()), 
                                    vec![(Some("%or".into()), Type::get_i32())]
                                );

                            let cond = lexp;
                            let br = program.func_mut(func).dfg_mut().new_value()
                                .branch_with_args(cond, or_done, or_r, 
                                    vec![lexp], 
                                    vec![]);
                            program.func_mut(func).layout_mut().bb_mut(or_l).insts_mut()
                                .push_key_back(br).unwrap();
                            
                            let jump = program.func_mut(func).dfg_mut().new_value()
                                .jump_with_args(or_done, vec![rexp]);
                            program.func_mut(func).layout_mut().bb_mut(or_r).insts_mut()
                                .push_key_back(jump).unwrap();
                            program.func(func).dfg().bb(or_done).params()[0]
                        },
                        _ => unreachable!(),
                    };
                    // Done
                    program.func_mut(func).layout_mut().bbs_mut().push_key_back(or_done).unwrap();
                    _ = bb.replace(or_done);
                    or
                }
                else {
                    // Global scope
                    assert!(bb.is_none());
                    let lexp = exp.lor_exp(raw_input, program, function, sym_tab, bb, is_ptr);
                    let lexp = boolean_exp(&(lexp, exp.token_pos()), raw_input, program, function, bb);
                    assert!(bb.is_none());
                    let rexp = more.land_exp(raw_input, program, function, sym_tab, bb, is_ptr);
                    let rexp = boolean_exp(&(rexp, exp.token_pos()), raw_input, program, function, bb);

                    let args = (BinaryOp::Or, lexp, rexp, exp.token_pos(), more.token_pos(), *pos);
                    binary_exp(&args, raw_input, program, function, bb)
                }
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
        bb: &mut Option<BasicBlock>,
        is_ptr: bool,
    ) -> Value {
        match self {
            LAndExp::EqExp(eq_exp) => eq_exp.eq_exp(raw_input, program, function, sym_tab, bb, is_ptr),
            LAndExp::And(exp, more, pos) => {
                
                // Short-circuit may be considered.
                //   %a = ...
                //   br %a, done, and_R
                // and_R:
                //   %and = %a && ...
                // and_done:

                if let Some(func) = function {

                    // Fucntion scope
                    
                    // Intermediate block
                    let and_r = program.func_mut(func).dfg_mut().new_bb()
                        .basic_block(Some("%and_R".into()));
                    
                    // Left exp
                    let lexp = exp.land_exp(raw_input, program, function, sym_tab, bb, is_ptr);
                    let lexp = boolean_exp(&(lexp, exp.token_pos()), raw_input, program, function, bb);

                    // Right exp
                    program.func_mut(func).layout_mut().bbs_mut().push_key_back(and_r).unwrap();
                    let and_l = bb.replace(and_r).unwrap();
                    let rexp = more.eq_exp(raw_input, program, function, sym_tab, bb, is_ptr);
                    let rexp = boolean_exp(&(rexp, exp.token_pos()), raw_input, program, function, bb);

                    // Some of the blocks may be stripped.

                    let mut lval = None;
                    let mut rval = None;

                    if let Some((lvalue, _)) = to_i32(program.func(func).dfg().value(lexp)) {
                        // If `lexp` is constant?
                        if lvalue == 0 {
                            // Short-circuited; contents of %and_R should be removed.
                            program.func_mut(func).layout_mut().bb_mut(and_r).insts_mut().clear();
                        }
                        lval = Some(lvalue);
                    }
                    if let Some ((rvalue, _)) = to_i32(program.func(func).dfg().value(rexp)) {
                        // If `rexp` is constant?
                        rval = Some(rvalue);
                    }

                    let mut and_done = program.func_mut(func).dfg_mut().new_bb()
                        .basic_block(Some("%and_done".into()));

                    let and = match (lval, rval) {
                        (Some(0), _) | (_, Some(0)) => {
                            // In the latter case, the code to generate `lexp` , if existent,
                            // is not cleared here, because they may have side effect.
                            program.func_mut(func).dfg_mut().new_value()
                                .integer(0)
                        },
                        (Some(1), Some(1)) => {
                            program.func_mut(func).dfg_mut().new_value()
                                .integer(1)
                        },
                        (None, Some(1)) => {
                            lexp
                        },
                        (Some(1), None) => {
                            rexp
                        },
                        (None, None) => {
                            // Only here we generate a branch
                            // and use a parameterized basic block
                            program.func_mut(func).dfg_mut().remove_bb(and_done);
                            and_done = program.func_mut(func).dfg_mut().new_bb()
                                .basic_block_with_param_names(
                                    Some("%and_done".into()), 
                                    vec![(Some("%and".into()), Type::get_i32())]
                                );

                            let cond = lexp;
                            let br = program.func_mut(func).dfg_mut().new_value()
                                .branch_with_args(cond, and_r, and_done, 
                                    vec![], 
                                    vec![lexp]);
                            program.func_mut(func).layout_mut().bb_mut(and_l).insts_mut()
                                .push_key_back(br).unwrap();
                            
                            let jump = program.func_mut(func).dfg_mut().new_value()
                                .jump_with_args(and_done, vec![rexp]);
                            program.func_mut(func).layout_mut().bb_mut(and_r).insts_mut()
                                .push_key_back(jump).unwrap();
                            program.func(func).dfg().bb(and_done).params()[0]
                        },
                        _ => unreachable!(),
                    };
                    // Done
                    program.func_mut(func).layout_mut().bbs_mut().push_key_back(and_done).unwrap();
                    _ = bb.replace(and_done);
                    and
                }
                else {
                    // Global scope
                    assert!(bb.is_none());
                    let lexp = exp.land_exp(raw_input, program, function, sym_tab, bb, is_ptr);
                    let lexp = boolean_exp(&(lexp, exp.token_pos()), raw_input, program, function, bb);
                    assert!(bb.is_none());
                    let rexp = more.eq_exp(raw_input, program, function, sym_tab, bb, is_ptr);
                    let rexp = boolean_exp(&(rexp, exp.token_pos()), raw_input, program, function, bb);

                    let args = (BinaryOp::And, lexp, rexp, exp.token_pos(), more.token_pos(), *pos);
                    binary_exp(&args, raw_input, program, function, bb)
                }
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
        bb: &mut Option<BasicBlock>,
        is_ptr: bool,
    ) -> Value {
        match self {
            EqExp::RelExp(rel_exp) => rel_exp.rel_exp(raw_input, program, function, sym_tab, bb, is_ptr),
            EqExp::Eq(exp, more, pos) |
            EqExp::Neq(exp, more, pos) => {
                let lexp = exp.eq_exp(raw_input, program, function, sym_tab, bb, is_ptr);
                let rexp = more.rel_exp(raw_input, program, function, sym_tab, bb, is_ptr);

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
        bb: &mut Option<BasicBlock>,
        is_ptr: bool,
    ) -> Value {
        match self {
            RelExp::AddExp(add_exp) => add_exp.add_exp(raw_input, program, function, sym_tab, bb, is_ptr),
            RelExp::Lt(exp, more, pos) |
            RelExp::Gt(exp, more, pos) |
            RelExp::Le(exp, more, pos) |
            RelExp::Ge(exp, more, pos) => {
                let lexp = exp.rel_exp(raw_input, program, function, sym_tab, bb, is_ptr);
                let rexp = more.add_exp(raw_input, program, function, sym_tab, bb, is_ptr);

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
        bb: &mut Option<BasicBlock>,
        is_ptr: bool,
    ) -> Value {
        match self {
            AddExp::MulExp(mul_exp) => mul_exp.mul_exp(raw_input, program, function, sym_tab, bb, is_ptr),
            AddExp::Add(exp, more, pos) |
            AddExp::Sub(exp, more, pos) => {
                let lexp = exp.add_exp(raw_input, program, function, sym_tab, bb, is_ptr);
                let rexp = more.mul_exp(raw_input, program, function, sym_tab, bb, is_ptr);

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
        bb: &mut Option<BasicBlock>,
        is_ptr: bool,
    ) -> Value {
        match self {
            MulExp::UnaryExp(unary_exp) => unary_exp.unary_exp(raw_input, program, function, sym_tab, bb, is_ptr),
            MulExp::Mul(exp, more, pos) |
            MulExp::Div(exp, more, pos) |
            MulExp::Rem(exp, more, pos) => {
                let lexp = exp.mul_exp(raw_input, program, function, sym_tab, bb, is_ptr);
                let rexp = more.unary_exp(raw_input, program, function, sym_tab, bb, is_ptr);

                let (op, is_div) = match self {
                    MulExp::Mul(_,_,_) => (BinaryOp::Mul, false),
                    MulExp::Div(_,_,_) => (BinaryOp::Div, true),
                    MulExp::Rem(_,_,_) => (BinaryOp::Mod, true),
                    _ => unreachable!(),
                };
                if is_div {
                    let rexp_val = if let Some(func) = function {
                        to_i32(program.func(func).dfg().value(rexp))
                    }
                    else {
                        to_i32(&program.borrow_value(rexp))
                    };
                    if let Some((i, _)) = rexp_val {
                        let pos = more.token_pos();
                        if i == 0 {
                            semantic_error(raw_input, pos.0, pos.1, 
                                &SemanticError::DividedByZero)
                        }
                    }
                }

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
        bb: &mut Option<BasicBlock>,
        is_ptr: bool,
    ) -> Value {
        // program.func_mut(function.unwrap()).dfg_mut().new_value().integer(2)
        match self {
            UnaryExp::PrimaryExp(pexp, _) => pexp.primary_exp(raw_input, program, function, sym_tab, bb, is_ptr),
            UnaryExp::Op(op, more, pos) => {
                match op {
                    UnaryOp::Plus => more.unary_exp(raw_input, program, function, sym_tab, bb, is_ptr),
                    UnaryOp::Minus => {
                        let lexp = if let Some(func) = function {
                            program.func_mut(func).dfg_mut().new_value().integer(0)
                        }
                        else {
                            program.new_value().integer(0)
                        };
                        let rexp = more.unary_exp(raw_input, program, function, sym_tab, bb, is_ptr);
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
                        let rexp = more.unary_exp(raw_input, program, function, sym_tab, bb, is_ptr);
                        let args = (BinaryOp::Eq, lexp, rexp, (0, 0), more.token_pos(), *pos);
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
        bb: &mut Option<BasicBlock>,
        is_ptr: bool,
    ) -> Value {
        match self {
            PrimaryExp::Bracketed(exp) => exp.exp(raw_input, program, function, sym_tab, bb, is_ptr),
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
            PrimaryExp::LVal(lval) => lval.lval(raw_input, program, function, sym_tab, bb, false, is_ptr),
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

    /// TODO: this is way too complicated, beautify this.
    fn lval (
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>, 
        sym_tab: &mut SymbolTable,
        bb: &mut Option<BasicBlock>,
        is_left: bool,
        is_ptr: bool, // True if the value should be interpreted as a pointer
    ) -> Value {

        // Check symbol table
        let sym = sym_tab.get_ident(&self.ident);
        if let None = sym {
            semantic_error(raw_input, self.ident.token_pos.0, self.ident.token_pos.1, &SemanticError::UndefinedIdent)
        }

        let (sym, sym_id) = sym.unwrap();
        match sym {
            Symbol::Function(_) => semantic_error(raw_input, self.ident.token_pos.0, self.ident.token_pos.1, &SemanticError::MisuseOfFuncIdent(sym_id)),
            Symbol::Value(lval, mut is_const, is_init, is_global) => {

                // Then check the indices over the type
                if let Some(func) = function {
                    
                    // Inside a function
                    
                    let mut insts = Vec::new();
                    let mut ty = if is_global {
                        program.borrow_value(lval).ty().clone()
                    }
                    else {
                        program.func(func).dfg().value(lval).ty().clone()
                    };
                    let mut indexed_lval = lval.clone();

                    if is_left {
                        if is_const {
                            semantic_error(raw_input, self.ident.token_pos.0, self.ident.token_pos.1, 
                                &SemanticError::ConstAssignment(sym_id.token_pos))
                        }
                        // Do not attempt to constantly evaluate
                        is_const = false;
                    }
                    if is_ptr {
                        is_const = false;
                    }

                    // Amend type and lval
                    if let TypeKind::Pointer(btype) = ty.kind() {
                        let btype = btype.clone();
                        ty = btype.clone();
                        
                        if is_const {
                            assert!(is_global);
                            match program.borrow_value(indexed_lval).kind() {
                                ValueKind::GlobalAlloc(alloc) => indexed_lval = alloc.init().clone(),
                                _ => panic!("[AST] Local const Lavl should always be a GlobalAlloc"),
                            }
                        }
                        if let TypeKind::Pointer(_) = btype.kind() {
                            let load = program.func_mut(func).dfg_mut().new_value()
                                    .load(lval);
                            insts.push(load);
                        }
                    }
                    else {
                        panic!("[AST] Type of any ident should always be pointer")
                    }
                    let btype = ty.clone();

                    // TODO: This is hard to handle
                    // if !is_init && self.indices.is_empty() {
                    //     warning(raw_input, self.token_pos.0, self.token_pos.1, &Warning::VarUninit(sym_id.token_pos))
                    // }

                    for exp in self.indices.iter() {
                        let index = exp.exp(raw_input, program, function, sym_tab, bb, is_ptr);
                        let index_data = program.func(func).dfg().value(index);
                        check_value_type(raw_input, &Type::get_i32(), index_data.ty(), &exp.token_pos());

                        match ty.kind() {
                            TypeKind::Pointer(btype) => {
                                assert!(!is_const);
                                let mut is_zero = false;
                                if let Some((index, _)) = to_i32(index_data) {
                                    is_zero = index == 0;
                                }
                                ty = btype.clone();
                                if !is_zero {
                                    let locate = program.func_mut(func).dfg_mut().new_value()
                                        .get_ptr(*insts.last().unwrap_or(&lval), index);
                                    insts.push(locate);
                                };
                            },
                            TypeKind::Array(btype, bound) => {
                                if let Some((index, _)) = to_i32(index_data) {
                                    let pos = exp.token_pos();
                                    if index >= *bound as i32 || index < 0 {
                                        warning(raw_input, pos.0, pos.1, 
                                            &Warning::IndexOutOfBound(index, sym_id.token_pos, *bound))
                                    }
                                    if is_const {
                                        match program.borrow_value(indexed_lval).kind() {
                                            ValueKind::Aggregate(agg) => 
                                                indexed_lval = agg.elems().get(index as usize).unwrap().clone(),
                                            _ => {}
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
                                let extra_pos = exp.token_pos();
                                semantic_error(raw_input, extra_pos.0, extra_pos.1,
                                    &SemanticError::ExtraIndex(sym_id.token_pos, btype))
                            }
                        }
                    }

                    if is_const {
                        // Can replace with a constant value
                        assert!(is_global);
                        assert!(!is_ptr);
                        let indexed_lval_data = program.borrow_value(indexed_lval);
                        assert!(indexed_lval_data.kind().is_const());
                        return if let Some((i, undef)) = to_i32(&indexed_lval_data) {
                            drop(indexed_lval_data);
                            if undef {
                                warning(raw_input, self.token_pos.0, self.token_pos.1, &Warning::ValueUndef)
                            }
                            program.func_mut(func).dfg_mut().new_value().integer(i)
                        }
                        else {
                            let pos = self.token_pos;
                            semantic_error(raw_input, pos.0, pos.1,
                                &SemanticError::ArrayPartialDeref(btype))
                        }
                    }
                    else {
                        if !is_ptr {
                            match ty.kind() {
                                TypeKind::Array(_,_) | TypeKind::Pointer(_) => {
                                    let pos = self.token_pos;
                                    semantic_error(raw_input, pos.0, pos.1,
                                        &SemanticError::ArrayPartialDeref(btype))
                                }
                                _ => {}
                            }
                        }
                        
                        // Must calculate with instructions
                        let last = insts.last();
                        let mut need_load = !is_left && !is_ptr;
                        need_load = need_load && 
                            (last.is_none() 
                            || !matches!(program.func(func).dfg().value(*last.unwrap()).kind(), ValueKind::Load(_)));
                        if need_load {
                            let load = program.func_mut(func).dfg_mut().new_value().load(*last.unwrap_or(&lval));
                            insts.push(load);
                        }
                        let last = *insts.last().unwrap_or(&lval);
                        program.func_mut(func).layout_mut().bb_mut(bb.unwrap()).insts_mut().extend(insts);
                        last
                    }
                }
                else {

                    // Global scope

                    let lval_data = program.borrow_value(lval);
                    let mut ty = lval_data.ty().clone();
                    let mut indexed_lval = lval.clone();
                    if let TypeKind::Pointer(btype) = ty.kind() {
                        ty = btype.clone();
                        match program.borrow_value(indexed_lval).kind() {
                            ValueKind::GlobalAlloc(alloc) => indexed_lval = alloc.init().clone(),
                            _ => panic!("[AST] Global LVal should always be a GlobalAlloc"),
                        }
                    }
                    else {
                        panic!("[AST] Type of any ident should always be pointer")
                    }
                    let btype = ty.clone();
                    if !is_init && self.indices.is_empty() {
                        warning(raw_input, self.token_pos.0, self.token_pos.1, &Warning::ValueUndef)
                    }

                    drop(lval_data);

                    for exp in self.indices.iter() {
                        let index = exp.exp(raw_input, program, function, sym_tab, bb, is_ptr);
                        let index_data = program.borrow_value(index);
                        check_value_type(raw_input, &Type::get_i32(), index_data.ty(), &exp.token_pos());

                        match ty.kind() {
                            TypeKind::Pointer(_) => unreachable!(),
                            TypeKind::Array(btype, bound) => {
                                if let Some((index, _)) = to_i32(&index_data) {
                                    let pos = exp.token_pos();
                                    if index >= *bound as i32 || index < 0 {
                                        warning(raw_input, pos.0, pos.1, 
                                            &Warning::IndexOutOfBound(index, sym_id.token_pos, *bound))
                                    }
                                    if let ValueKind::Aggregate(agg) 
                                        = program.borrow_value(indexed_lval).kind() 
                                    {
                                        indexed_lval = agg.elems().get(index as usize).unwrap().clone();
                                    }
                                    else {
                                        unreachable!()
                                    }
                                }
                                else {
                                    let pos = exp.token_pos();
                                    semantic_error(raw_input, pos.0, pos.1,
                                        &SemanticError::ConstExpected)
                                }
                                ty = btype.clone();
                            },
                            _ => {
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
                        let pos = self.token_pos;
                        semantic_error(raw_input, pos.0, pos.1,
                            &SemanticError::ArrayPartialDeref(btype))
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

    fn token_pos(&self) -> TokenPos {
        self.pos
    }

    fn const_exp(
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>, 
        sym_tab: &mut SymbolTable,
        bb: &mut Option<BasicBlock>, 
    ) -> Value {
        let (exp, pos) = (self.exp.exp(raw_input, program, function, sym_tab, bb, false), self.pos);
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
