//! Implements Koopa IR generation from SYSY AST.
use koopa::ir::{*, builder_traits::*};
use koopa::ir::entities::ValueData;
use crate::frontend::ast::*;
use crate::middleend::{SymbolTable, Symbol, opt};
use crate::error::*;
use std::rc::Rc;

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

    pub fn elim_unused_global(program: &mut Program) {
        let unused: Vec<Value> = program.inst_layout()
            .iter()
            .filter(|v| program.borrow_value(**v).used_by().is_empty())
            .cloned()
            .collect();
        for u in unused {
            program.remove_value(u);
        }
    }
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

impl ConstDecl {

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

impl ConstDef {

    fn const_def(
        &self,
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

impl ConstInitVal {
    fn const_init_val(
        &self,
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>,
        sym_tab: &mut SymbolTable,
        bb: &mut Option<BasicBlock>,
        args: (Value, &Type, usize, &[usize]),
    ) -> (Option<Value>, usize) {
        let (alloc, ty, idx, _) = args;
        match self {
            ConstInitVal::ConstExp(init_exp) => {
                let exp = init_exp.const_exp(raw_input, program, function, sym_tab, bb);
                let args = (exp, ty, &init_exp.token_pos());
                (init_single(args, raw_input, program, function, bb, alloc, true), idx+1)
            },
            ConstInitVal::ValList(val_list, pos) => {
                init_aggregate(
                    &val_list.iter().map(|v| v.clone() as Rc<dyn IsInitVal>).collect()
                    ,pos, raw_input, program, function, sym_tab, bb, args, true
                )
            },
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
        args: (Value, &Type, usize, &[usize]),
        _is_const: bool,
    ) -> (Option<Value>, usize) {
        self.const_init_val(raw_input, program, function, sym_tab, bb, args)
    }

    fn token_pos(&self) -> TokenPos {
        match self {
            ConstInitVal::ConstExp(exp) => exp.token_pos(),
            ConstInitVal::ValList(_,pos) => *pos,
        }
    }

    fn is_aggregate(&self) -> bool {
        match self {
            ConstInitVal::ConstExp(_) => false,
            ConstInitVal::ValList(_, _) => true,
        }
    }
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

impl VarDef {

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
        args: (Value, &Type, usize, &[usize]),
        is_const: bool,
    ) -> (Option<Value>, usize) /* (Const Value?, next_idx) */ {
        let (alloc, btype, idx, _) = args;
        match self {
            InitVal::Exp(init_exp) => {
                let exp = init_exp.exp(raw_input, program, function, sym_tab, bb, false);
                let args = (exp, btype, &init_exp.token_pos());
                (init_single(args, raw_input, program, function, bb, alloc, is_const), idx+1)
            },
            InitVal::ValList(val_list, pos) => {
                init_aggregate(
                    &val_list.iter().map(|v| v.clone() as Rc<dyn IsInitVal>).collect()
                    ,pos, raw_input, program, function, sym_tab, bb, args, is_const
                )
            }
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
        args: (Value, &Type, usize, &[usize]),
        is_const: bool,
    ) -> (Option<Value>, usize) {
        self.init_val(raw_input, program, function, sym_tab, bb, args, is_const)
    }

    fn token_pos(&self) -> TokenPos {
        match self {
            InitVal::Exp(exp) => exp.token_pos(),
            InitVal::ValList(_, pos) => *pos,
        }
    }

    fn is_aggregate(&self) -> bool {
        match self {
            InitVal::Exp(_) => false,
            InitVal::ValList(_, _) => true,
        }
    }
}

impl FuncDef {

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
            
            if bb.is_some() {
                let bb = bb.unwrap();
                if !ret_ty.is_unit() { 
                    warning(raw_input, 0, 0,
                        &Warning::MissingRet(self.ident.token_pos));

                    let undef = program.func_mut(func).dfg_mut().new_value().undef(ret_ty);
                    let ret = program.func_mut(func).dfg_mut().new_value()
                        .ret(Some(undef));
                    program.func_mut(func).layout_mut().bb_mut(bb).insts_mut()
                        .push_key_back(ret).unwrap();
                }
                else {
                    // No warning, just add the missing return.
                    let ret = program.func_mut(func).dfg_mut().new_value()
                        .ret(None);
                    program.func_mut(func).layout_mut().bb_mut(bb).insts_mut()
                        .push_key_back(ret).unwrap();
                }
            }

            opt::sanity_check(program.func_mut(func));
            
            // Out of the scope
            sym_tab.pop();
        }
        else {
            unreachable!()
        }
    }
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
                            &&SemanticError::NonPositiveArrayBound(i))
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
        let mut unreached = false;
        for item in self.block_item.iter() {
            if unreached {
                let pos = item.token_pos();
                warning(raw_input, pos.0, pos.1,
                    &Warning::UnreachedStmt);
                break;
            }
            item.block_item(raw_input, program, function, sym_tab, bb, ret_ty);
            if bb.is_none() {
                // Terminating block item 
                unreached = true;
            }
        }
        sym_tab.pop();
    }
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
            BlockItem::Decl(decl, _) => decl.decl(raw_input, program, Some(function), sym_tab, bb),
            BlockItem::Stmt(stmt, _) => stmt.stmt(raw_input, program, function, sym_tab, bb, ret_ty),
        }
    }

    fn token_pos(&self) -> TokenPos {
        match self {
            BlockItem::Decl(_, pos) | BlockItem::Stmt(_, pos) => *pos,
        }
    }
}



impl Stmt {

    fn stmt(
        &self, 
        raw_input: &[u8],
        program: &mut Program,
        function: Function,
        sym_tab: &mut SymbolTable,
        bb: &mut Option<BasicBlock>,
        ret_ty: &Type,
    ) {
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

                let mut lty = match sym_tab.get(&lval.ident).expect("ident must be defined") {
                    Symbol::Function(_) => unreachable!(),
                    Symbol::Value(_,_,_,_) => {
                        if let Some(data) = program.func(function).dfg().values().get(&lvalue) {
                            data.ty().clone()
                        }
                        else {
                            program.borrow_value(lvalue).ty().clone()
                        }
                    }
                };

                let rdata = program.func(function).dfg().value(rexp);
                let rty = rdata.ty().clone();
                // `lty` should be a pointer
                match lty.kind() {
                    TypeKind::Array(btype,_) | TypeKind::Pointer(btype) => lty = btype.clone(),
                    _ => unreachable!(),
                }          
                check_value_type(raw_input, &lty, &rty, &exp.token_pos());

                // All clear; generate a store
                let store = program.func_mut(function).dfg_mut()
                    .new_value().store(rexp, lvalue);
                program.func_mut(function).layout_mut().bb_mut(bb.unwrap())
                    .insts_mut().push_key_back(store).unwrap();
            },

            Stmt::If(cond, if_stmt, else_stmt, pos) => {
                
                let func = function;

                // Expand cond
                let cond_exp = cond.exp(raw_input, program, Some(func), sym_tab, bb, false);
                check_value_type(raw_input, &Type::get_i32(),
                    program.func(func).dfg().value(cond_exp).ty(), &cond.token_pos());

                let if_bb = program.func_mut(func).dfg_mut().new_bb()
                    .basic_block(Some("%if".into()));
                let else_bb = program.func_mut(func).dfg_mut().new_bb()
                    .basic_block(Some("%else".into()));
                let done_bb = program.func_mut(func).dfg_mut().new_bb()
                    .basic_block(Some("%if_done".into()));

                // Expand if
                program.func_mut(func).layout_mut().bbs_mut().push_key_back(if_bb).unwrap();
                let prev_bb = bb.replace(if_bb).unwrap();
                if_stmt.stmt(raw_input, program, function, sym_tab, bb, ret_ty);
                
                // Expand else
                program.func_mut(func).layout_mut().bbs_mut().extend([else_bb, done_bb]);
                let (if_end_bb, else_end_bb) = match else_stmt {
                    Some(else_stmt) => {
                        let if_end_bb = bb.replace(else_bb);
                        else_stmt.stmt(raw_input, program, function, sym_tab, bb, ret_ty);
                        (if_end_bb, bb.take())
                    },
                    None => (bb.take(), Some(else_bb)),
                };

                // Set up jumps properly
                let br = program.func_mut(func).dfg_mut().new_value()
                    .branch(cond_exp, if_bb, else_bb);
                program.func_mut(func).layout_mut().bb_mut(prev_bb)
                    .insts_mut().push_key_back(br).unwrap();
                if let Some(if_end_bb) = if_end_bb {
                    // May be None, if the if-branch terminates
                    let if_jump = program.func_mut(func).dfg_mut().new_value()
                        .jump(done_bb);
                    program.func_mut(func).layout_mut().bb_mut(if_end_bb)
                        .insts_mut().push_key_back(if_jump).unwrap();
                }
                if let Some(else_end_bb) = else_end_bb {
                    // May be None, if the else-branch terminates
                    let else_jump = program.func_mut(func).dfg_mut().new_value()
                        .jump(done_bb);
                    program.func_mut(func).layout_mut().bb_mut(else_end_bb)
                        .insts_mut().push_key_back(else_jump).unwrap();
                }

                // Both branches are expanded for semantic checks,
                // but they may actually be dead.
                // We save removing these for later.
                let cond_i32 = to_i32(program.func(func).dfg().value(cond_exp))
                    .map(|(i,_)| i);
                let terminate = if_end_bb.is_none() && else_end_bb.is_none();
                match cond_i32 {
                    Some(0) => {
                        warning(raw_input, pos.0, pos.1,
                            &Warning::RedunantIf(0));
                    },
                    Some(i) if i != 0 => {
                        warning(raw_input, pos.0, pos.1,
                            &Warning::RedunantIf(i));
                    },
                    None => {},
                    _ => unreachable!(),
                };

                // Finally, a `if` terminates if both branches terminate
                if terminate {
                    *bb = None;
                }
                else {
                    *bb = Some(done_bb);
                }
            },

            Stmt::While(cond, body, pos) => {

                let func = function;

                let cond_bb = program.func_mut(func).dfg_mut().new_bb()
                    .basic_block(Some("%while_cond".into()));
                let entry_bb = program.func_mut(func).dfg_mut().new_bb()
                    .basic_block(Some("%while_entry".into()));
                let exit_bb = program.func_mut(func).dfg_mut().new_bb()
                    .basic_block(Some("%while_exit".into()));

                // Terminates the old BB
                let old_bb = bb.replace(cond_bb);
                let jump = program.func_mut(func).dfg_mut().new_value()
                    .jump(cond_bb);
                program.func_mut(func).layout_mut().bb_mut(old_bb.unwrap()).insts_mut()
                    .push_key_back(jump).unwrap();

                // Expand cond
                program.func_mut(func).layout_mut().bbs_mut().push_key_back(cond_bb).unwrap();
                let cond_exp = cond.exp(raw_input, program, Some(func), sym_tab, bb, false);
                check_value_type(raw_input, &Type::get_i32(),
                    program.func(func).dfg().value(cond_exp).ty(), &cond.token_pos());
                let mut _may_be_dead = false;
                if let Some((i, _)) = to_i32(program.func(func).dfg().value(cond_exp)) {
                    if i == 0 {
                        warning(raw_input, pos.0, pos.1,
                            &Warning::RedunantWhile)
                    }
                    else {
                        _may_be_dead = true;
                    }
                }

                program.func_mut(func).layout_mut().bbs_mut().push_key_back(entry_bb).unwrap();
                // Generate branch
                let cond_end_bb = bb.take().unwrap();
                let br = program.func_mut(func).dfg_mut().new_value()
                    .branch(cond_exp, entry_bb, exit_bb);
                program.func_mut(func).layout_mut().bb_mut(cond_end_bb).insts_mut()
                    .push_key_back(br).unwrap();
                
                // Expand body
                _ = bb.insert(entry_bb);
                sym_tab.push_resume_bb((cond_bb, exit_bb));
                body.stmt(raw_input, program, function, sym_tab, bb, ret_ty);
                _ = sym_tab.pop_resume_bb();

                program.func_mut(func).layout_mut().bbs_mut().push_key_back(exit_bb).unwrap();

                // Generate loopback
                if let Some(body_end_bb) = bb {
                    let jump = program.func_mut(func).dfg_mut().new_value()
                        .jump(cond_bb);
                    program.func_mut(func).layout_mut().bb_mut(*body_end_bb).insts_mut()
                        .push_key_back(jump).unwrap();
                }
                
                // Finally, a `while` really terminates if 
                // - `may_be_dead` is true
                // - The body is not always continued
                // This is hard to handle for now, so we assume it
                // never terminates.
                *bb = Some(exit_bb);
            },

            Stmt::Break(pos) => {
                if let Some((_, break_bb)) = sym_tab.get_resume_bb() {
                    let jump = program.func_mut(function).dfg_mut().new_value()
                        .jump(break_bb);
                    program.func_mut(function).layout_mut().bb_mut(bb.unwrap())
                        .insts_mut().push_key_back(jump).unwrap();
                    // Break terminates current BB
                    *bb = None;
                }
                else {
                    semantic_error(raw_input, pos.0, pos.1,
                        &SemanticError::ExtraBreak)
                }
            },

            Stmt::Continue(pos) => {
                if let Some((cont_bb, _)) = sym_tab.get_resume_bb() {
                    let jump = program.func_mut(function).dfg_mut().new_value()
                        .jump(cont_bb);
                    program.func_mut(function).layout_mut().bb_mut(bb.unwrap())
                        .insts_mut().push_key_back(jump).unwrap();
                    // Continue terminates current BB
                    *bb = None;
                }
                else {
                    semantic_error(raw_input, pos.0, pos.1,
                        &SemanticError::ExtraCont)
                }
            },
        }
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

impl LVal {

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
            Symbol::Value(lval, mut is_const, _, is_global) => {

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

                    let mut ptr_san_check = false;

                    for exp in self.indices.iter() {

                        let index = exp.exp(raw_input, program, function, sym_tab, bb, 
                            false /* The index is never a pointer */);
                        let index_data = program.func(func).dfg().value(index);
                        check_value_type(raw_input, &Type::get_i32(), index_data.ty(), &exp.token_pos());

                        match ty.kind() {
                            TypeKind::Pointer(btype) => {
                                assert!(!is_const);
                                assert!(!ptr_san_check, "[lval] base type of double pointer should not occur");
                                ty = btype.clone();
                                let locate = program.func_mut(func).dfg_mut().new_value()
                                    .get_ptr(*insts.last().unwrap_or(&lval), index);
                                insts.push(locate);
                                ptr_san_check = true;
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
                                &SemanticError::ArrayPartialDeref(ty))
                        }
                    }
                    else {
                        if !is_ptr {
                            match ty.kind() {
                                TypeKind::Array(_,_) | TypeKind::Pointer(_) => {
                                    let pos = self.token_pos;
                                    semantic_error(raw_input, pos.0, pos.1,
                                        &SemanticError::ArrayPartialDeref(ty))
                                }
                                _ => {}
                            }
                        }
                        
                        // Must calculate with instructions
                        let last = insts.last();

                        let need_load = !is_left && !is_ptr;
                        let need_gep = is_ptr && matches!(ty.kind(), TypeKind::Array(_,_));
                        // println!("{} {}, isleft:{}, isptr:{}, load:{}, gep:{}", self.ident.to_string(), need_load, is_left, is_ptr, need_load, need_gep);
                        // println!("{:?}", lval);
                        if need_load {
                            let load = program.func_mut(func).dfg_mut().new_value().load(*last.unwrap_or(&lval));
                            insts.push(load);
                        }
                        else if need_gep {
                            // Need to add a `GEP` here
                            let zero = program.func_mut(func).dfg_mut().new_value()
                                .integer(0);
                            let gep = program.func_mut(func).dfg_mut().new_value()
                                .get_elem_ptr(
                                    *last.unwrap_or(&lval),
                                    zero);
                            insts.push(gep);
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
                            &SemanticError::ArrayPartialDeref(ty))
                    }
                }
            },
        }
    }
}

impl FuncRParams {
    fn func_rparams(
        &self,
        raw_input: &[u8],
        program: &mut Program,
        function: Function,
        sym_tab: &mut SymbolTable,
        bb: &mut Option<BasicBlock>, 
        is_ptr_vec: &Vec<bool>,
    ) -> Vec<Value> {
        self.params.iter().zip(is_ptr_vec.iter())
            .map(|(exp, is_ptr)| {
                let arg = exp.exp(raw_input, program, Some(function), sym_tab, bb, *is_ptr);
                arg
            })
            .collect()
    }
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
                    let or_l_end = bb.replace(or_r).unwrap();
                    let rexp = more.land_exp(raw_input, program, function, sym_tab, bb, is_ptr);
                    let rexp = boolean_exp(&(rexp, exp.token_pos()), raw_input, program, function, bb);
                    let or_r_end = bb.expect("exp should never terminate BB");

                    // Some of the blocks may be stripped.
                    // But we leave that for later passes.

                    let mut lval = None;
                    let mut rval = None;

                    if let Some((lvalue, _)) = to_i32(program.func(func).dfg().value(lexp)) {
                        // If `lexp` is constant?
                        lval = Some(lvalue);
                    }
                    if let Some ((rvalue, _)) = to_i32(program.func(func).dfg().value(rexp)) {
                        // If `rexp` is constant?
                        rval = Some(rvalue);
                    }

                    // Use a parameterized basic block
                    let or_done = program.func_mut(func).dfg_mut().new_bb()
                        .basic_block_with_param_names(
                            Some("%or_done".into()), 
                            vec![(Some("%or".into()), Type::get_i32())]
                        );

                    let cond = lexp;
                    let br = program.func_mut(func).dfg_mut().new_value()
                        .branch_with_args(cond, or_done, or_r, 
                            vec![lexp], 
                            vec![]);
                    program.func_mut(func).layout_mut().bb_mut(or_l_end).insts_mut()
                        .push_key_back(br).unwrap();
                    
                    let jump = program.func_mut(func).dfg_mut().new_value()
                        .jump_with_args(or_done, vec![rexp]);
                    program.func_mut(func).layout_mut().bb_mut(or_r_end).insts_mut()
                        .push_key_back(jump).unwrap();
                    let or_calc = program.func(func).dfg().bb(or_done).params()[0];
                    
                    let or = match (lval, rval) {
                        (Some(1), _) | (_, Some(1)) => {
                            program.func_mut(func).dfg_mut().new_value()
                                .integer(1)
                        },
                        (Some(0), Some(0)) => {
                            program.func_mut(func).dfg_mut().new_value()
                                .integer(0)
                        },
                        (None, Some(0)) | (Some(0), None) | (None, None) => {
                            or_calc
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
                    let lexp = exp.lor_exp(raw_input, program, function, sym_tab, bb, is_ptr);
                    let lexp = boolean_exp(&(lexp, exp.token_pos()), raw_input, program, function, bb);
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
                    let and_l_end = bb.replace(and_r).unwrap();
                    let rexp = more.eq_exp(raw_input, program, function, sym_tab, bb, is_ptr);
                    let rexp = boolean_exp(&(rexp, exp.token_pos()), raw_input, program, function, bb);
                    let and_r_end = bb.take().expect("exp should never terminates BB");

                    // Some of the blocks may be stripped.

                    let mut lval = None;
                    let mut rval = None;

                    if let Some((lvalue, _)) = to_i32(program.func(func).dfg().value(lexp)) {
                        // If `lexp` is constant?
                        lval = Some(lvalue);
                    }
                    if let Some ((rvalue, _)) = to_i32(program.func(func).dfg().value(rexp)) {
                        // If `rexp` is constant?
                        rval = Some(rvalue);
                    }

                    // Use a parameterized basic block
                    let and_done = program.func_mut(func).dfg_mut().new_bb()
                        .basic_block_with_param_names(
                            Some("%and_done".into()), 
                            vec![(Some("%and".into()), Type::get_i32())]
                        );

                    let cond = lexp;
                    let br = program.func_mut(func).dfg_mut().new_value()
                        .branch_with_args(cond, and_r, and_done, 
                            vec![], 
                            vec![lexp]);
                    program.func_mut(func).layout_mut().bb_mut(and_l_end).insts_mut()
                        .push_key_back(br).unwrap();
                    
                    let jump = program.func_mut(func).dfg_mut().new_value()
                        .jump_with_args(and_done, vec![rexp]);
                    program.func_mut(func).layout_mut().bb_mut(and_r_end).insts_mut()
                        .push_key_back(jump).unwrap();
                    let and_calc = program.func(func).dfg().bb(and_done).params()[0];

                    let and = match (lval, rval) {
                        (Some(0), _) | (_, Some(0)) => {
                            program.func_mut(func).dfg_mut().new_value()
                                .integer(0)
                        },
                        (Some(1), Some(1)) => {
                            program.func_mut(func).dfg_mut().new_value()
                                .integer(1)
                        },
                        (None, Some(1)) | (Some(1), None) | (None, None) => {
                            and_calc
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
                    let lexp = exp.land_exp(raw_input, program, function, sym_tab, bb, is_ptr);
                    let lexp = boolean_exp(&(lexp, exp.token_pos()), raw_input, program, function, bb);
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
            UnaryExp::FuncCall(fid, args, pos) => {
                // Check symbol table
                let func_sym = sym_tab.get_ident(fid);
                if func_sym.is_none() {
                    semantic_error(raw_input, fid.token_pos.0, fid.token_pos.1,
                        &SemanticError::UndefinedIdent)
                }
                let (func_sym, func_id) = func_sym.unwrap();
                match func_sym {
                    Symbol::Value(_,_,_,_) =>
                        semantic_error(raw_input, fid.token_pos.0, fid.token_pos.1,
                            &SemanticError::FunctionExpected(func_id)),
                    Symbol::Function(func) => {
                        // Must be in function scope
                        if function.is_none() {
                            semantic_error(raw_input, pos.0, pos.1,
                                &SemanticError::ConstExpected)
                        }

                        // Check argument count
                        let expected_count = program.func(func).params().len();
                        let found_count = match args {
                            Some(args) => args.params.len(),
                            None => 0,
                        };
                        if expected_count != found_count {
                            semantic_error(raw_input, pos.0, pos.1,
                                &SemanticError::ArgMismatch(
                                    expected_count, 
                                    found_count,
                                    func_id.token_pos
                                ))
                        }

                        let is_ptr_vec: Vec<bool> = program.func(func).params()
                            .iter()
                            .map(|exp| match program.func(func).dfg().value(*exp).ty().kind() {
                                TypeKind::Array(_,_) | TypeKind::Pointer(_) => true,
                                _ => false,
                            })
                            .collect();
                        
                        let positions = match args {
                            Some(args) => args.params.iter().map(
                                |exp| exp.token_pos()
                            ).collect(),
                            None => vec![], 
                        };

                        let cur_func = function.unwrap();
                        
                        // Expand arguments
                        let arg_exps = match args {
                            Some(args) => args.func_rparams(
                                raw_input, 
                                program,
                                cur_func,
                                sym_tab, 
                                bb,
                                &is_ptr_vec
                            ),
                            None => vec![],
                        };

                        // Check argument types
                        let func_data = program.func(func);
                        for i in 0..arg_exps.len() {
                            let actual_exp = arg_exps.get(i).unwrap();
                            let expect_exp = func_data.params().get(i).unwrap();
                            let actual_ty = match program.func(cur_func).dfg().values().get(actual_exp) {
                                Some(data) => data.ty().clone(),
                                None => program.borrow_value(*actual_exp).ty().clone()
                            };
                            let expect_ty = func_data.dfg().value(*expect_exp).ty();
                            check_value_type(raw_input,
                                expect_ty, &actual_ty, positions.get(i).unwrap());
                        }

                        // Generate function call
                        let call = program.func_mut(cur_func).dfg_mut().new_value()
                            .call(func, arg_exps);
                        program.func_mut(cur_func).layout_mut().bb_mut(bb.unwrap())
                            .insts_mut().push_key_back(call).unwrap();
                        call
                    }
                }
            },
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



trait IsInitVal {
    fn init(
        &self,
        raw_input: &[u8],
        program: &mut Program,
        function: Option<Function>,
        sym_tab: &mut SymbolTable,
        bb: &mut Option<BasicBlock>,
        args: (Value, &Type, usize, &[usize]),
        is_const: bool,
    ) -> (Option<Value>, usize);

    fn token_pos(&self) -> TokenPos;

    fn is_aggregate(&self) -> bool;
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
    let mut dims = Vec::new();
    let mut is_global = false;
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
            if bound <= 0 {
                let pos = exp.token_pos();
                semantic_error(raw_input, pos.0, pos.1,
                    &SemanticError::NonPositiveArrayBound(bound))
            }
            ty = Type::get_array(ty, bound as usize);
            dims.push(bound as usize);
        }

        if is_const {
            // Local const is handled as a global alloc.
            // For now, init as undef.
            let undef = program.new_value().undef(ty);
            let alloc = program.new_value().global_alloc(undef);
            is_global = true;
            alloc
        }
        else {
            // Do a local alloc
            let alloc = program.func_mut(func).dfg_mut().new_value().alloc(ty);
            program.func_mut(func).layout_mut().bb_mut(bb.unwrap()).insts_mut().push_key_back(alloc).unwrap();
            if init_val.is_some() {
                // Can start with zeroinit, if initval is present.
                let mut zero = program.func_mut(func).dfg_mut().new_value()
                    .integer(0);
                for dim in dims.iter() {
                    zero = program.func_mut(func).dfg_mut().new_value()
                        .aggregate(vec![zero; *dim]);
                }
                let store = program.func_mut(func).dfg_mut().new_value()
                    .store(zero, alloc);
                program.func_mut(func).layout_mut().bb_mut(bb.unwrap())
                    .insts_mut().push_key_back(store).unwrap();
            }
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
            dims.push(bound as usize);
        }

        // For now, init as undef.
        let init = if init_val.is_some() {
            program.new_value().zero_init(ty)
        }
        else {
            program.new_value().undef(ty)
        };
        let alloc = program.new_value().global_alloc(init);
        is_global = true;
        alloc
    };

    // Add to symbol table; may err
    sym_tab.add(raw_input, ident, Symbol::Value(def, is_const, init_val.is_some(), is_global));
    let scope_name = if let Some(func) = function {
        String::from(&program.func(func).name()[1..])
    }
    else {
        String::new()
    };
    if is_global {
        program.set_value_name(def, sym_tab.get_mangled(&scope_name, ident, is_const));
    }

    // Then handle the init value; this order is consistent with C.
    // For example:
    // const int x = 1;
    // {
    //    const int x[x+1] = {x};
    // }
    // the array will be initialized with the address of itself, rather than 1.
    if let Some(init_val) = init_val {

        if !dims.is_empty() && !init_val.is_aggregate() {
            let pos = init_val.token_pos();
            semantic_error(raw_input, pos.0, pos.1,
                &SemanticError::ValListExpected)
        }

        let btype = match btype {
            BType::Int => Type::get_i32(),
        };

        let (init, _) = if !(is_const || function.is_none()) {
            // For local initialization, need to prepare
            // `def` into a pointer that points to the raw
            // index 0.
            let mut gep = def;
            for _ in dims.iter() {
                let zero = program.func_mut(function.unwrap()).dfg_mut().new_value()
                    .integer(0);
                let locate = program.func_mut(function.unwrap()).dfg_mut().new_value()
                    .get_elem_ptr(gep, zero);
                program.func_mut(function.unwrap()).layout_mut().bb_mut(bb.unwrap())
                    .insts_mut().push_key_back(locate).unwrap();
                gep = locate;
            }
            init_val.init(raw_input, program, function, sym_tab, bb, 
            (gep, &btype, 0, dims.as_slice()), 
                false
            )
        }
        else {
            init_val.init(raw_input, program, function, sym_tab, bb, 
                (def, &btype, 0, dims.as_slice()), 
                true
            )
        };
        if let Some(mut init) = init {

            // In case of an aggregate, it is returned flattened.
            // Must construct a properly leveled one.
            let data = program.borrow_value(init);
            let kind = data.kind().clone();
            drop(data);
            match kind {
                ValueKind::Aggregate(agg) => {
                    if dims.len() >= 1 {
                        let mut exps = Vec::new();
                        let mut reduce_points = Vec::new();
                        let mut prod = 1usize;
                        for dim in dims.iter() {
                            prod *= *dim;
                            reduce_points.push(prod);
                        }
                        for i in 1..agg.elems().len()+1 {
                            exps.push(*agg.elems().get(i-1).unwrap());
                            for (red, dim) in reduce_points.iter().zip(dims.iter()) {
                                if i % *red == 0 {
                                    // println!("reduce at {} by {}, stack has {}", i, *red, exps.len());
                                    let new_agg = program.new_value().aggregate(
                                        exps.drain(exps.len()-*dim..).collect()
                                    );
                                    exps.push(new_agg);
                                }
                            }
                        }
                        assert!(exps.len() == 1);
                        init = *exps.last().unwrap();
                    }
                },
                _ => {},
            }

            // Replace the `init` value of `def`
            program.remove_value(def);
            let alloc = program.new_value().global_alloc(init);
            _ = sym_tab.replace(ident, Symbol::Value(alloc, is_const, true, true));
            if is_global {
                program.set_value_name(alloc, sym_tab.get_mangled(&scope_name, ident, is_const));
            }
        }
        // Otherwise, `init()` already generated instructions for the initialization.
    }
    // Otherwise the value is left as undef.
}

/// Common code for initializing a single value.
fn init_single(
    args: (Value, &Type, &TokenPos),
    raw_input: &[u8],
    program: &mut Program,
    function: Option<Function>,
    bb: &mut Option<BasicBlock>,
    alloc: Value,
    is_const: bool,
) -> Option<Value> {
    let (exp, btype, pos) = args;

    let exp_ty = if let Some(func) = function { 
        program.func(func).dfg().value(exp).ty().clone()
    }
    else {
        program.borrow_value(exp).ty().clone()
    };

    check_value_type(raw_input, btype, &exp_ty, pos);

    if is_const {
        if let Some(func) = function {
            // A local const value is still stored in global
            // data section, and needs an initializer.
            let exp_data = program.func(func).dfg().value(exp);
            // Add to global scope because eventually it is stored there.
            if let Some((i, _)) = to_i32(exp_data) {
                return Some(program.new_value().integer(i));
            }
            else {
                unreachable!()
            }
        }
        else {
            let exp_data = program.borrow_value(exp);
            if let Some((i, _)) = to_i32(&exp_data) {
                drop(exp_data);
                return Some(program.new_value().integer(i));
            }
            else {
                unreachable!()
            }
        }
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

fn init_aggregate(
    val_list: &Vec<Rc<dyn IsInitVal>>,
    pos: &TokenPos,
    raw_input: &[u8],
    program: &mut Program,
    function: Option<Function>,
    sym_tab: &mut SymbolTable,
    bb: &mut Option<BasicBlock>,
    args: (Value, &Type, usize, &[usize]),
    is_const: bool,
) -> (Option<Value>, usize) {
    let (mut alloc, btype, mut idx, dims) = args;
    // Determine how many elements this ValList is
    // initializing.
    if dims.is_empty() {
        semantic_error(raw_input, pos.0, pos.1,
            &SemanticError::ValListTooDeep(idx))
    }
    
    if idx % dims[0] != 0 {
        semantic_error(raw_input, pos.0, pos.1,
            &SemanticError::ValListMisalign(idx, dims[0]))
    }
    let mut bound = 1usize;
    for dim in dims {
        if idx % (bound * *dim) == 0 {
            bound *= *dim;
        }
        else {
            break
        }
    }
    let old_idx = idx;

    let mut exps: Vec<Value> = Vec::new();
    for init_exp in val_list.iter() {
        if idx >= old_idx + bound {
            let exp_pos = init_exp.token_pos();
            semantic_error(raw_input, exp_pos.0, exp_pos.1,
                &SemanticError::ValListTooWide(idx, *pos, old_idx, bound))
        }
        let args = (alloc, btype, idx, &dims[0..dims.len()-1]);
        let (eval_exp, next_idx) = init_exp.init(
            raw_input, 
            program, 
            function, 
            sym_tab, 
            bb, args, is_const);
        
        if is_const {

            idx = next_idx;
            // Must be constantly evaluated
            let eval_exp = eval_exp.unwrap();
            match program.borrow_value(eval_exp).kind() {
                ValueKind::Aggregate(agg) => {
                    exps.extend(agg.elems())
                }
                _ => exps.push(eval_exp),
            }
        }
        else {
            // Must be calculated
            // In this case, `alloc` is assumed to be a pointer instruction
            // that points to the raw index.
            let gp = alloc;
            let func = function.unwrap();
            
            assert!(eval_exp.is_none());
            let index = program.func_mut(func).dfg_mut().new_value()
                .integer((next_idx - idx) as i32);
            let locate = program.func_mut(func).dfg_mut().new_value()
                .get_ptr(gp, index);
            program.func_mut(func).layout_mut().bb_mut(bb.unwrap())
                .insts_mut().push_key_back(locate).unwrap();
            alloc = locate;
            idx = next_idx;
        }
        
    }
    assert!(idx <= old_idx + bound);
    for _ in idx..old_idx + bound {
        // The val-list is shorter than array, patch up with zeroes
        if is_const {
            idx += 1;
            exps.push(program.new_value().integer(0));
        }
        else {
            // let gp = alloc;
            // let func = function.unwrap();
            
            // let zero = program.func_mut(func).dfg_mut().new_value()
            //     .integer(0);
            // let store = program.func_mut(func).dfg_mut().new_value()
            //     .store(zero, gp);
            // let index = program.func_mut(func).dfg_mut().new_value()
            //     .integer(1);
            // let locate = program.func_mut(func).dfg_mut().new_value()
            //     .get_ptr(gp, index);
            // program.func_mut(func).layout_mut().bb_mut(bb.unwrap())
            //     .insts_mut().extend([store, locate]);
            // alloc = locate;
            // idx += 1;

            idx = old_idx + bound;
            break;
        }
    }
    assert!(idx == old_idx + bound);
    if is_const {
        assert!(exps.len() == bound);
        let agg = program.new_value().aggregate(exps);
        (Some(agg), idx)
    }
    else {
        (None, idx)
    }
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
pub(crate) fn to_i32(data: &ValueData) -> Option<(i32, bool)> {
    match data.kind() {
        ValueKind::Integer(i) => Some((i.value(), false)),
        ValueKind::ZeroInit(_) => Some((0, false)),
        ValueKind::Undef(_) => Some((0, true)),
        _ => None
    }
}

pub(crate) fn binary_op_eval(op: &BinaryOp, lvalue: i32, rvalue: i32) -> (i32, bool) {
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