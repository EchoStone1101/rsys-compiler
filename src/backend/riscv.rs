//! Implements RISCV assembly generation from Koopa IR.
//! 
//! The core complexity here is the register allocation scheme,
//! so here is a rundown of the whole picture:
//! - Each referenced Value is associated with *one* `ValueInfo`
//! - A `ValueInfo` describes how the `Value` is to be represented
//!   in the object code. 
//! - IR instructions each requires `ValueInfo` in a certain way.
//!   For example, an IR ADD requires the operands to be concrete
//!   registers, perhaps causing a load from spill locations. An
//!   IR GEP expects an offset as index, producing an offset or
//!   a concrete, the latter when the index is not constant.
//! - IR basic blocks each constraints `ValueInfo`. This happens when
//!   a BB *uses* a `Value` that is defined outside the block,
//!   which includes two case:
//!     - (a) `Value` is a block argument. This implementation
//!       arbitrarily rules that the first 7 block arguments reside
//!       in t0~t6, and the rest is considered spilled.
//!     - (b) `Value` is simply defined elsewhere. In that case, at
//!       the end of any other blocks that directly precedes this block,
//!       they must align the `ValueInfo` to the first use in this block.
//!       For example, if in current block B1 the first use of `%10` 
//!       was in register `s2`, then later in B2 that jumps to B1,
//!       must emit code that put the value of `%10` back to `s2`.
//! - Finally, liveliness analysis is performed on each `Value` to
//!   improve the performance of register allocation. A `Value` that
//!   is no longer used will have its `ValueInfo` killed.
//! 
//! All these combined allows for a register allocation and
//! object code emission algorithm almost within one pass. 
//! The tasks after this pass:
//! - Collect the stack positions actually used, to decide
//!   the frame size.
//! - Collect the callee-saved register used. 
//! - Generate function prologue.
//! - Generate function epilogue.

#[allow(unused)]
use std::io::{Write, Result};
use std::collections::{HashMap, HashSet, VecDeque};
use koopa::ir::{*, entities::*, values::*, layout::*};
use koopa::back::{Generator, NameManager};
use crate::backend::program_info::*;
use crate::middleend::ir::to_i32;

const REG_COUNT: usize = 32;

/* This implementation also uses `ra` for holding temporary values. */
const TEMP_REG: Reg = Reg::Ra;


macro_rules! is_imm12 {
    ($v:expr) => {
        (($v) as i32 >= -2048 && ($v) as i32 <= 2047)
    };
}

macro_rules! stack_aligned {
    ($v:expr) => {
        if ($v) & 0xF != 0 {
            ((($v) | 0xF) + 1)
        }
        else {
            ($v)
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Reg {
    Zero = 0,
    Ra,
    Sp,
    Gp,
    Tp,
    T0,
    T1,
    T2,
    Fp,
    S1,
    A0,
    A1,
    A2,
    A3,
    A4,
    A5,
    A6,
    A7,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8,
    S9,
    S10,
    S11,
    T3,
    T4,
    T5,
    T6,
}

impl Reg {

    /// Whether this register is reserved for special use.
    fn is_reserved(&self) -> bool {
        match self {
            Reg::Zero | Reg::Ra | Reg::Sp |
            Reg::Gp | Reg::Fp | Reg::Tp 
                => true,
            _ => false,
        }
    }

    /// Whether this register is callee-saved.
    #[allow(unused)]
    fn is_callee_saved(&self) -> bool {
        match self {
            Reg::Sp | Reg::Fp | Reg::S1 |
            Reg::S2 | Reg::S3 | Reg::S4 |
            Reg::S5 | Reg::S6 | Reg::S7 |
            Reg::S8 | Reg::S9 | Reg::S10 |
            Reg::S11
                => true,
            _ => false,
        }
    }

    /// Whether this register is caller-saved.
    fn is_caller_saved(&self) -> bool {
        match self {
            Reg::Ra | Reg::T0 | Reg::T1 |
            Reg::T2 | Reg::A0 | Reg::A1 |
            Reg::A2 | Reg::A3 | Reg::A4 |
            Reg::A5 | Reg::A6 | Reg::A7 |
            Reg::T3 | Reg::T4 | Reg::T5 |
            Reg::T6
                => true,
            _ => false,
        }
    }
}

impl std::fmt::Display for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Reg::Zero => write!(f, "zero"),
            Reg::A0 => write!(f, "a0"),
            Reg::A1 => write!(f, "a1"),
            Reg::A2 => write!(f, "a2"),
            Reg::A3 => write!(f, "a3"),
            Reg::A4 => write!(f, "a4"),
            Reg::A5 => write!(f, "a5"),
            Reg::A6 => write!(f, "a6"),
            Reg::A7 => write!(f, "a7"),
            Reg::Sp => write!(f, "sp"),
            Reg::Fp => write!(f, "fp"),
            Reg::Gp => write!(f, "gp"),
            Reg::Tp => write!(f, "tp"),
            Reg::Ra => write!(f, "ra"),
            Reg::S1 => write!(f, "s1"),
            Reg::S2 => write!(f, "s2"),
            Reg::S3 => write!(f, "s3"),
            Reg::S4 => write!(f, "s4"),
            Reg::S5 => write!(f, "s5"),
            Reg::S6 => write!(f, "s6"),
            Reg::S7 => write!(f, "s7"),
            Reg::S8 => write!(f, "s8"),
            Reg::S9 => write!(f, "s9"),
            Reg::S10 => write!(f, "s10"),
            Reg::S11 => write!(f, "s11"),
            Reg::T0 => write!(f, "t0"),
            Reg::T1 => write!(f, "t1"),
            Reg::T2 => write!(f, "t2"),
            Reg::T3 => write!(f, "t3"),
            Reg::T4 => write!(f, "t4"),
            Reg::T5 => write!(f, "t5"),
            Reg::T6 => write!(f, "t6"),
        }
    }
}

impl From<i32> for Reg {
    fn from(value: i32) -> Self {
        match value {
            0 => Reg::Zero,
            1 => Reg::Ra,
            2 => Reg::Sp,
            3 => Reg::Gp,
            4 => Reg::Tp,
            5 => Reg::T0,
            6 => Reg::T1,
            7 => Reg::T2,
            8 => Reg::Fp,
            9 => Reg::S1,
            10 => Reg::A0,
            11 => Reg::A1,
            12 => Reg::A2,
            13 => Reg::A3,
            14 => Reg::A4,
            15 => Reg::A5,
            16 => Reg::A6,
            17 => Reg::A7,
            18 => Reg::S2,
            19 => Reg::S3,
            20 => Reg::S4,
            21 => Reg::S5,
            22 => Reg::S6,
            23 => Reg::S7,
            24 => Reg::S8,
            25 => Reg::S9,
            26 => Reg::S10,
            27 => Reg::S11,
            28 => Reg::T3,
            29 => Reg::T4,
            30 => Reg::T5,
            31 => Reg::T6,
            _ => panic!("[RISCV] bad register index"),
        }
    }
}

/// Manages the value information of `Value`'s. The on-the-fly
/// generator queries this manager to decide where to fetch
/// and put the values, in order to emit object code.
/// This manager complies to the RISCV ABI, especially the
/// calling conventions.
#[derive(Debug, Default)]
struct ValueManager<W: Write> {
    /// Map from `Value` into **one** `ValueInfo`.
    value_info: HashMap<Value, ValueInfo>,

    /// Manages all the necessary information for the
    /// current associated `Program`, such as the uses
    /// of `Value`s.
    program_info: ProgramInfo,

    /// Manages the user of all registers.
    register_info: [Option<Value>; REG_COUNT],

    /// Emitted object code by lines
    object_code: Vec<Code>,

    /// Constraints of each basic block on `Value`s.
    /// TODO: Not implemented.
    // value_constraints: HashMap<BasicBlock, Vec<(Value, ValueInfo)>>,

    /// Statistics, including the registers and stack
    /// locations used.
    stat: Statistics,

    phantom: std::marker::PhantomData<W>,
}

/// The value information of a `Value`.
#[derive(Debug, Clone, Copy)]
enum ValueInfo {
    /// In register.
    Reg(Reg),
    /// Spilled.
    Spilled(i32),

    /// The folllowing is used to optimize for ALLOC, GEP and GP 
    /// instructions, and do not take up real storage.
    
    /// Offset from a register value.
    /// Note: this variant does not contain a Reg, because having
    /// an Offset does not "lock" the value in a specific register. 
    /// What this variant really implies is that the base value is 
    /// meant to be loaded into a register.
    RegOffset(Value, i32),
    /// Offset from current FP.
    StackOffset(i32),
    /// Offset from an absolute global position.
    GlobalOffset(Value, i32),
}

impl ValueInfo {

    /// Get the Reg value if is `Reg` variant.
    /// Panic otherwise.
    pub fn to_reg(&self) -> Reg {
        match self {
            ValueInfo::Reg(reg) => *reg,
            _ => panic!("[RISCV] info not a register")
        }
    }

    /// Adjust the offset value by add `ofs`, if is
    /// one of the offset variant.
    /// Panic otherwise.
    pub fn add_offset(&mut self, ofs: i32) {
        match self {
            ValueInfo::RegOffset(_, offset) => *offset += ofs,
            ValueInfo::StackOffset(offset) => *offset += ofs,
            ValueInfo::GlobalOffset(_, offset) => *offset += ofs,
            _ => panic!("[RISCV] info not a offset")
        }
    }
}

/// A section of RISCV object code, which may
/// be substituted.
#[derive(Debug)]
enum Code {
    Literal(String),
    Prologue,
    Epilogue,
    Guard(Option<String>),
}

impl From<String> for Code {
    fn from(value: String) -> Self {
        Code::Literal(value)
    }
}

impl std::fmt::Display for Code {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Code::Literal(s) => write!(f, "{}\n", s),
            Code::Prologue => write!(f, "  [[PROLOGUE]]\n"),
            Code::Epilogue => write!(f, "  [[EPILOGUE]]\n"),
            Code::Guard(_) => write!(f, "  [[GUARD]]\n"),
        }
    }
}

/// Statistics of the used registers and stack positions.
#[derive(Debug, Default)]
struct Statistics {
    regs_used: HashSet<Reg>,
    frame_size: usize,
    nonce: usize,
}

impl Statistics {
    pub fn clear(&mut self) {
        self.regs_used.clear();
        // Save position for storing `ra`.
        self.frame_size = 4;
        self.nonce = 0;
    }

    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    pub fn use_reg(&mut self, reg: Reg) {
        self.regs_used.insert(reg);
    }

    pub fn use_stack(&mut self, _ofs: usize, sz: usize) -> i32 {
        self.frame_size += sz;
        -(self.frame_size as i32)
    }

    pub fn nonce(&mut self) -> usize {
        let nonce = self.nonce;
        self.nonce = usize::wrapping_add(self.nonce, 1);
        nonce
    }

    #[allow(unused)]
    pub fn regs_used(&self) -> Vec<Reg> {
        Vec::from_iter(self.regs_used.iter().cloned())
    }
}

impl<W: Write> ValueManager<W> {

    /// Creates a new manager with the associated `program`.
    pub fn new(program: &Program) -> Self {
        let mut vm: ValueManager<W> = ValueManager {
            phantom: std::marker::PhantomData {},
            value_info: Default::default(),
            program_info: Default::default(),
            register_info: Default::default(),
            object_code: Default::default(),
            stat: Default::default(),
        };
        vm.program_info.init(program);
        vm
    }

    /// TODO: currently naive.
    /// For now at the end of visiting a BB, `value_info` should contain only
    /// local and global `alloc`s, and `register_info` should be all None.
    pub fn sanity_check(&mut self) {
        let values: Vec<Value> = self.value_info.keys().cloned().collect();
        let cur_pos = self.object_code.len();
        for value in values {
            let info = self.value_info.get(&value).cloned().unwrap();

            if self.program_info.is_outliver(&value)
                && !self.program_info.is_alloc(&value) {
                match info {
                    ValueInfo::Spilled(_) => {},
                    _ => {
                        // Outlivers are evicted.
                        let reg = self.value_to_register(value, None).to_reg();
                        self.evict(reg);
                    }
                }
            }
            else {
                assert! (match info {
                    ValueInfo::GlobalOffset(_,_) | ValueInfo::StackOffset(_) |
                    ValueInfo::Spilled(_) => true,
                    _ => {
                        println!("[SAN-CHECK] {:?}: {:?}", value, info);
                        false
                    },
                }, "[SAN-CHECK] Reg/RegOffset outlives BB");
            }
        }

        for usage in self.register_info {
            assert!(usage.is_none(), "[SAN-CHECK] register outlives BB, and is inconsistent");
        }

        let new_pos = self.object_code.len();
        let evict_code: Vec<Code> = self.object_code.drain(cur_pos..new_pos).collect();
        let guard = self.object_code.iter_mut().rev().find(
            |code| matches!(code, Code::Guard(None))
        ).unwrap();
        if new_pos > cur_pos {
            // Move these code to the previous [[GUARD]]
            let mut code = String::new();
            for c in evict_code {
                code += c.to_string().as_str();
            }
            if let Code::Guard(guard) = guard {
                _ = guard.insert(code);
            }
            else {
                unreachable!()
            }
        }
        else {
            if let Code::Guard(guard) = guard {
                _ = guard.insert(String::new());
            }
            else {
                unreachable!()
            }
        }
        
    }

    /// Call when entering a new function.
    /// Resets internal state.
    pub fn enter_func(&mut self, program: &Program, func: Function) {

        let ValueManager {
            ref mut value_info,
            ref mut program_info,
            ref mut register_info,
            ref mut object_code,
            // ref mut value_constraints,
            ref mut stat,
            ..
        } = *self;

        value_info.retain(|v, _| program_info.is_globl(v));
        program_info.new_func(program, func);
        register_info.fill(None);
        object_code.clear();
        // value_constraints.clear();
        stat.clear();
    }

    /// Marks the end of processing instruction `inst`, killing
    /// `Value`s that no longer lives.
    pub fn progress(&mut self, inst: Value) {

        let ValueManager { 
            ref mut value_info, 
            ref mut program_info, 
            ref mut register_info,
            ..
        } = *self;

        let kills = program_info.get_kills(&inst);
        // println!("[KILL] {:?} {:?}", inst, kills);
        if kills.is_none() {
            return
        }

        let kills = kills.unwrap();
        for killed in kills {
            if let Some(info) = value_info.remove(killed) {
                match info {
                    ValueInfo::Reg(reg) => {
                        register_info[reg as usize] = None;
                    },
                    _ => {},
                }
            }
            else {
                if *killed != inst {
                    panic!("[RISCV] killed Value is not registered")
                }
            }
        }
    }

    /// Call when exiting a visited function.
    /// Emits the prologue, body and epilogue of the
    /// generated object code.
    pub fn exit_func(&mut self, w: &mut W) -> Result<()> 
    where W: Write {

        let mut used_callee_saved: Vec<Reg> = self.stat.regs_used()
            .iter()
            .filter(|reg| reg.is_callee_saved())
            .cloned()
            .collect();
        // Frame pointer is always saved
        used_callee_saved.push(Reg::Fp);

        // println!("===> used callee regs:{:?}", used_callee_saved);

        // Final frame size
        let frame_size = stack_aligned!(
            self.stat.frame_size() + 4 * used_callee_saved.len() + 4 /* `ra` */
        );

        for code in self.object_code.iter_mut() {
            match code {
                Code::Literal(s) => writeln!(w, "{}", s)?,
                Code::Prologue => {
                    //| Arg #10
                    //| Arg #9
                    //|---------- < sp [new fp]
                    //| [save ra]
                    //|----------
                    //| locals
                    //|----------
                    //| [save 
                    //|  used
                    //|  callee-
                    //|  saved]
                    //|---------- < [new sp]

                    // Save `ra`
                    // `ra` must not be used as TEMP_REG before this,
                    // so we store it first.
                    write!(w, "  sw\t{}, {}({})\n", Reg::Ra, -4, Reg::Sp)?;

                    // Move `sp`
                    if is_imm12!(frame_size){
                        // Within imm12 offset
                        write!(w, "  addi\t{}, {}, {}\n", Reg::Sp, Reg::Sp, -(frame_size as i32))?;
                    }
                    else {
                        write!(w, "  li\t{}, {}\n", TEMP_REG,  -(frame_size as i32))?;
                        write!(w, "  add\t{}, {}, {}\n", Reg::Sp, Reg::Sp, TEMP_REG)?;
                    }

                    // Save callee-saved registers
                    let mut ofs = 0;
                    for reg in used_callee_saved.iter() {
                        if is_imm12!(ofs) {
                            // Within imm12 offset
                            write!(w, "  sw\t{}, {}({})\n", *reg, ofs, Reg::Sp)?;
                        }
                        else {
                            write!(w, "  li\t{}, {}\n", TEMP_REG, ofs)?;
                            write!(w, "  add\t{}, {}, {}\n", TEMP_REG, TEMP_REG, Reg::Sp)?;
                            write!(w, "  sw\t{}, 0({})\n", *reg, TEMP_REG)?;
                        }
                        ofs += 4;
                    }
                    
                    // Move `fp`
                    if is_imm12!(frame_size){
                        // Within imm12 offset
                        write!(w, "  addi\t{}, {}, {}\n", Reg::Fp, Reg::Sp, frame_size)?;
                    }
                    else {
                        write!(w, "  li\t{}, {}\n", TEMP_REG, frame_size)?;
                        write!(w, "  add\t{}, {}, {}\n", Reg::Fp, Reg::Sp, TEMP_REG)?;
                    }
                },
                Code::Epilogue => {

                    // Restore callee-saved registers
                    let mut ofs = 0;
                    for reg in used_callee_saved.iter() {
                        if is_imm12!(ofs) {
                            // Within imm12 offset
                            write!(w, "  lw\t{}, {}({})\n", *reg, ofs, Reg::Sp)?;
                        }
                        else {
                            write!(w, "  li\t{}, {}\n", TEMP_REG, ofs)?;
                            write!(w, "  add\t{}, {}, {}\n", TEMP_REG, TEMP_REG, Reg::Sp)?;
                            write!(w, "  lw\t{}, 0({})\n", *reg, TEMP_REG)?;
                        }
                        ofs += 4;
                    }

                    // Move `sp`
                    if is_imm12!(frame_size){
                        // Within imm12 offset
                        write!(w, "  addi\t{}, {}, {}\n", Reg::Sp, Reg::Sp, frame_size)?;
                    }
                    else {
                        write!(w, "  li\t{}, {}\n", TEMP_REG, frame_size)?;
                        write!(w, "  add\t{}, {}, {}\n", Reg::Sp, Reg::Sp, TEMP_REG)?;
                    }

                    // Restore `ra`
                    write!(w, "  lw\t{}, {}({})\n", Reg::Ra, -4, Reg::Sp)?;

                    // `ret`
                    write!(w, "  ret\n")?;
                },
                Code::Guard(s) => write!(w, "{}", s.as_ref().unwrap())?,
            }
            
        }
        Ok(())
    }

    /// Whether `reg` is used after instruction `inst`.
    /// To qualify as still in-use, a `reg` must have an entry
    /// in `register_info`, and the corresponding `Value` is 
    /// not killed by `inst`.
    pub fn register_used_after(&self, reg: Reg, inst: Value) -> bool {
        self.register_info[reg as usize].is_some() 
        && !self.program_info.get_kills(&inst).map_or(
            false, 
            |kills| kills.contains(&self.register_info[reg as usize].unwrap())
        )
    }

    /// Map `reg` into the entry in `register_info`.
    pub fn register_to_value(&self, reg: Reg) -> Option<Value> {
        self.register_info[reg as usize]
    }

    /// Insert `info` for `value`.
    pub fn insert_info(
        &mut self, 
        value: Value, 
        info: ValueInfo
    ) {
        _ = self.value_info.insert(value, info);

        match info {
            ValueInfo::Reg(reg) => {
                // Resides in `reg`.
                // May trigger an eviction.
                self.evict(reg);
                self.register_info[reg as usize] = Some(value);
            },
            ValueInfo::Spilled(ofs) => {
                // Resides on a spilled stack location.
                assert!(ofs >= -(self.stat.frame_size() as i32));
            },
            _ => {/* Offsets do not take up real storage. */}
        }
    }

    /// Allocates a new position on stack for `value`, with size `sz`,
    /// returning the offset.
    /// For now we follow a simple approach: increment the frame size 
    /// each time, and returning bottom of the frame.
    /// TODO: reap the allocation for values that no longer lives.
    pub fn alloc_stack(&mut self, _: Value, sz: usize) -> i32 {
        self.stat.use_stack(0, sz)
    }

    /// Get the register info for `value`.
    /// If `value` is currently not in register, a load is
    /// generated. Further, this may cause eviction and even
    /// spilling.
    /// If `dest_reg` is not `None`, the destination register
    /// will be `dest_reg`, which may cause register movement.
    pub fn value_to_register(
        &mut self, 
        value: Value,
        dest_reg: Option<(Reg, bool)>,
    ) -> ValueInfo {

        // println!("[INFO] {:?} to {:?}, outliver: {}", value, dest_reg, self.program_info.is_outliver(&value));

        let info = self.value_info.get(&value)
            .expect("[RISCV] value not registered")
            .clone();
        let new_info = match info {
            ValueInfo::Reg(reg) => {
                if dest_reg.is_none() {
                    //println!("[INFO] {}", reg);
                    return info;
                }
                let (dest_reg, force) = dest_reg.unwrap();
                if dest_reg == reg || !force {
                    //println!("[INFO] {}", reg);
                    return info;
                }

                // Move
                if self.register_info[dest_reg as usize].is_none() {
                    _ = self.value_info.insert(value, ValueInfo::Reg(dest_reg));
                    self.register_info[dest_reg as usize] = Some(value);
                    self.register_info[reg as usize] = None;
                    self.emit_mv(reg, dest_reg);
                }
                else {
                    // Swap 
                    let other = self.register_info[dest_reg as usize].expect("[RISCV] inconsistent");
                    _ = self.value_info.insert(value, ValueInfo::Reg(dest_reg));
                    _ = self.value_info.insert(other, ValueInfo::Reg(reg));
                    self.register_info[dest_reg as usize] = Some(value);
                    self.register_info[reg as usize] = Some(other);
                    self.emit_swap(reg, dest_reg);
                }
                ValueInfo::Reg(dest_reg)
            },
            ValueInfo::Spilled(ofs) => {
                let (reg, _) = dest_reg.unwrap_or((self.alloc_register(None), true));

                let new_info = ValueInfo::Reg(reg);
                self.insert_info(value, new_info);
                self.emit_stack_load(ofs, reg);
                new_info
            },
            ValueInfo::StackOffset(ofs) => {
                // Instantiate the offset value
                let (reg, _) = dest_reg.unwrap_or((self.alloc_register(None), true));

                let new_info = ValueInfo::Reg(reg);
                // Outliver allocs must not be Reg
                if !self.program_info.is_outliver(&value) {
                    self.insert_info(value, new_info);
                }
                self.emit_stack_offset(ofs, reg);

                new_info
            },
            ValueInfo::GlobalOffset(globl, ofs) => {
                // Instantiate the offset value
                let (reg, _) = dest_reg.unwrap_or((self.alloc_register(None), true));

                let new_info = ValueInfo::Reg(reg);
                // Outliver allocs must not be Reg
                if !self.program_info.is_outliver(&value) {
                    self.insert_info(value, new_info);
                }
                self.emit_globl_offset(globl, ofs, reg);

                new_info
            },
            ValueInfo::RegOffset(reg_val, ofs) => {
                // Get the base value to register first
                match self.value_to_register(reg_val, dest_reg) {
                    ValueInfo::Reg(reg) => {
                        let new_info = ValueInfo::Reg(reg);
                        self.insert_info(value, new_info);
                        self.emit_reg_offset(ofs, reg, reg);

                        new_info
                    },
                    _ => unreachable!(),
                }
            }
        };
        //println!("[INFO] {}", new_info.to_reg());
        new_info
    }

    /// Get the current ValueInfo for `value` as a
    /// offset.
    pub fn value_to_offset(&mut self, value: Value) -> ValueInfo {
        let offset = self.value_info.get(&value)
            .expect("[RISCV] value not registered");

        match offset {
            ValueInfo::RegOffset(_,_) | ValueInfo::GlobalOffset(_,_) |
            ValueInfo::StackOffset(_)
                => *offset,
            ValueInfo::Reg(reg) => {
                ValueInfo::RegOffset(
                    self.register_info[*reg as usize].expect("[RISCV] inconsistent"),
                    0
                )
            },
            ValueInfo::Spilled(_) => {
                self.value_to_register(value, None);
                self.value_to_offset(value)
            },
        }
    }

    /// Allocates a register storage.
    /// Implements a simple register allocation scheme:
    /// - Any register that is not currently used can be allocated
    /// - When no such register is available, pick with clockhand
    ///   algorithm.
    pub fn alloc_register(&mut self, skip: Option<&[Reg]>) -> Reg {

        let nonce = self.stat.nonce() % REG_COUNT;

        let reg = self.register_info.iter().enumerate()
            .filter(|(i, _)| {
                let reg = Reg::from(*i as i32);
                !reg.is_reserved() && !(skip.is_some() && skip.unwrap().contains(&reg))
            })
            .map(|(i, user)| {
                // Approximate cost for using each register
                let reg = Reg::from(i as i32);
                if user.is_none() {
                    // Unoccupied register has the lowest cost
                    return (i32::MIN, reg);
                }
                ((nonce + i) as i32, reg)
            })
            .min_by_key(|(cost, _)| *cost)
            .expect("[RISCV] no available register").1;
        self.stat.use_reg(reg);
        reg
    }

    /// Checks for and performs register value eviction.
    pub fn evict(&mut self, reg: Reg) {
        let usage = self.register_info[reg as usize];
        if usage.is_none() {
            return
        }

        self.register_info[reg as usize] = None;
        let value = usage.unwrap();
        _ = self.value_info.remove(&value);

        self.spill(reg, value);
    }

    /// Find and store `value` in `reg` to a spill location on the stack.
    fn spill(&mut self, reg: Reg, value: Value) {
        let ofs = self.alloc_stack(value, 4); // Register is always 4-byte
        let new_info = ValueInfo::Spilled(ofs);
        
        self.insert_info(value, new_info);

        self.emit_stack_store(ofs, reg);
    }

    pub fn emit_code(&mut self, code: Code) {
        self.object_code.push(code);
    }

    /// Emits a load of global value `value` into `reg`.
    pub fn emit_globl_load(&mut self, v: Value, ofs: i32, reg: Reg) {
        let name = self.program_info.get_globl(&v).expect("[RISCV] no such global name");

        // self.object_code.push(format!("  lui\t{}, %hi({})", TEMP_REG, name).into());
        // self.object_code.push(format!("  addi\t{}, %lo({})", TEMP_REG, name).into());
        self.object_code.push(format!("  la\t{}, {}", reg, name).into());

        self.emit_reg_load(ofs, reg, reg);
    }

    /// Emits a store of global value `value` with `reg`.
    pub fn emit_globl_store(&mut self, v: Value, ofs: i32, reg: Reg) {
        let name = self.program_info.get_globl(&v).expect("[RISCV] no such global name");

        // self.object_code.push(format!("  lui\t{}, %hi({})", TEMP_REG, name).into());
        // self.object_code.push(format!("  addi\t{}, %lo({})", TEMP_REG, name).into());

        // This unfortunately takes two temporary registers.
        self.object_code.push(format!("  la\t{}, {}", Reg::Tp, name).into());

        self.emit_reg_store(ofs, Reg::Tp, reg);
    }

    /// Emits a load of stack value at offset `ofs` into `reg`.
    pub fn emit_stack_load(&mut self, ofs: i32, reg: Reg) {
        self.emit_reg_load(ofs, Reg::Fp, reg)
    }

    /// Emits a store of stack value at offset `ofs` with `reg`.
    pub fn emit_stack_store(&mut self, ofs: i32, reg: Reg) {
        self.emit_reg_store(ofs, Reg::Fp, reg)
    }

    /// Emits a load of register-relative value at offset `ofs` 
    /// into `dest_reg`.
    pub fn emit_reg_load(&mut self, ofs: i32, reg: Reg, dest_reg: Reg) {
        if is_imm12!(ofs) {
            // Within imm12 offset
            self.object_code.push(format!("  lw\t{}, {}({})", dest_reg, ofs, reg).into());
        }
        else {
            self.object_code.push(format!("  li\t{}, {}", TEMP_REG, ofs).into());
            self.object_code.push(format!("  add\t{}, {}, {}", TEMP_REG, TEMP_REG, reg).into());
            self.object_code.push(format!("  lw\t{}, 0({})", dest_reg, TEMP_REG).into());
        }
    }

    /// Emits a store of register-relative value at offset `ofs`
    /// with `src_reg`.
    pub fn emit_reg_store(&mut self, ofs: i32, reg: Reg, src_reg: Reg) {
        
        if is_imm12!(ofs) {
            // Within imm12 offset
            self.object_code.push(format!("  sw\t{}, {}({})", src_reg, ofs, reg).into());
        }
        else {
            self.object_code.push(format!("  li\t{}, {}", TEMP_REG, ofs).into());
            self.object_code.push(format!("  add\t{}, {}, {}", TEMP_REG, TEMP_REG, reg).into());
            self.object_code.push(format!("  sw\t{}, 0({})", src_reg, TEMP_REG).into());
        }
    }

    /// Emits a swap of `reg1` and `reg2`.
    pub fn emit_swap(&mut self, reg1: Reg, reg2: Reg) {
        self.object_code.push(format!("  mv\t{}, {}", TEMP_REG, reg1).into());
        self.object_code.push(format!("  mv\t{}, {}", reg1, reg2).into());
        self.object_code.push(format!("  mv\t{}, {}", reg2, TEMP_REG).into());
    }

    /// Emits a global offset calculation.
    pub fn emit_globl_offset(&mut self, v: Value, ofs: i32, reg: Reg) {
        let name = self.program_info.get_globl(&v).expect("[RISCV] no such global name");

        // self.object_code.push(format!("  lui\t{}, %hi({})", reg, name).into());
        // self.object_code.push(format!("  addi\t{}, {}, %lo({})", reg, reg, name).into());
        self.object_code.push(format!("  la\t{}, {}", reg, name).into());
        self.emit_reg_offset(ofs, reg, reg);
    }

    /// Emits a stack offset calculation.
    pub fn emit_stack_offset(&mut self, ofs: i32, reg: Reg) {
        self.emit_reg_offset(ofs, Reg::Fp, reg)
    }

    /// Emits register offset calculation.
    pub fn emit_reg_offset(&mut self, ofs: i32, reg: Reg, dest_reg: Reg) {
        if is_imm12!(ofs) {
            // Within imm12 offset
            self.object_code.push(format!("  addi\t{}, {}, {}", dest_reg, reg, ofs).into());
        }
        else {
            self.object_code.push(format!("  li\t{}, {}", TEMP_REG, ofs).into());
            self.object_code.push(format!("  add\t{}, {}, {}", dest_reg, reg, TEMP_REG).into());
        }
    }

    /// Emits a immediate load.
    pub fn emit_li(&mut self, imm: i32, dest_reg: Reg) {
        self.object_code.push(format!("  li\t{}, {}", dest_reg, imm).into());
    }

    /// Emits a move.
    pub fn emit_mv(&mut self, reg: Reg, dest_reg: Reg) {
        if reg != dest_reg {
            self.object_code.push(format!("  mv\t{}, {}", dest_reg, reg).into());
        }
    }

    /// Emits an add;
    pub fn emit_add(&mut self, reg: Reg, adder: Reg, dest_reg: Reg) {
        self.object_code.push(format!("  add\t{}, {}, {}", dest_reg, reg, adder).into());
    }

    /// Emits an immediate add.
    pub fn emit_addi(&mut self, reg: Reg, imm: i32, dest_reg: Reg) {
        if is_imm12!(imm){
            // Within imm12 offset
            self.object_code.push(format!("  addi\t{}, {}, {}", dest_reg, reg, imm).into());
        }
        else {
            self.object_code.push(format!("  li\t{}, {}", TEMP_REG, imm).into());
            self.object_code.push(format!("  add\t{}, {}, {}", dest_reg, reg, TEMP_REG).into());
        }
    }

    /// Emits an multiplication;
    pub fn emit_mul(&mut self, reg: Reg, mult: Reg, dest_reg: Reg) {
        self.object_code.push(format!("  mul\t{}, {}, {}", dest_reg, reg, mult).into());
    }

    /// Emits an immediate multiplication;
    pub fn emit_muli(&mut self, reg: Reg, imm: i32, dest_reg: Reg) {

        let mut imm = imm as u32;
        if imm == 0 {
            self.emit_mv(Reg::Zero, dest_reg);
            return;
        }
        
        let popcnt = imm.count_ones();
        // The following trick works only for `reg != dest_reg`
        if popcnt <= 4 && reg != dest_reg {
            let mut generated = false;
            if imm & 1 != 0 {
                self.emit_mv(reg, dest_reg);
                generated = true;
                imm &= 0xFFFFFFFE;
            }

            let mut mask = 2;
            for shamt in 1..32 {
                if mask & imm != 0 {
                    if generated {
                        self.object_code.push(format!("  slli\t{}, {}, {}", TEMP_REG, reg, shamt).into());
                        self.object_code.push(format!("  add\t{}, {}, {}", dest_reg, dest_reg, TEMP_REG).into());
                    }
                    else {
                        self.object_code.push(format!("  slli\t{}, {}, {}", dest_reg, reg, shamt).into());
                        generated = true;
                    }
                }
                mask = mask.wrapping_shl(1);
            }
        }
        else {
            self.emit_li(imm as i32, TEMP_REG);
            self.emit_mul(reg, TEMP_REG, dest_reg);
        }
    }

    /// Emits an immediate SLT.
    pub fn emit_slti(&mut self, reg: Reg, imm: i32, dest_reg: Reg) {
        if is_imm12!(imm){
            // Within imm12 offset
            self.object_code.push(format!("  slti\t{}, {}, {}", dest_reg, reg, imm).into());
        }
        else {
            self.object_code.push(format!("  li\t{}, {}", TEMP_REG, imm).into());
            self.object_code.push(format!("  slt\t{}, {}, {}", dest_reg, reg, TEMP_REG).into());
        }
    }
}

/// Visitor for generating the in-memeory form Koopa IR program into
/// RISCV object code.
#[derive(Default)]
pub struct Visitor;

impl<W: Write> koopa::back::Visitor<W> for Visitor {

    type Output = ();

    fn visit(&mut self, w: &mut W, _: &mut NameManager, program: &Program) -> Result<()> {
        let mut visitor = VisitorImpl {
            w,
            vm: &mut ValueManager::new(program),
            program,
            func: None,
        };
        visitor.visit()
    }
}

pub type RiscvGenerator<W> = Generator<W, Visitor>;

/// The implementation of text form Koopa IR generator.
struct VisitorImpl<'a, W: Write> {
    w: &'a mut W,
    vm: &'a mut ValueManager<W>,
    program: &'a Program,
    func: Option<&'a FunctionData>,
}
  
/// Returns a reference to the current function.
macro_rules! func {
    ($self:ident) => {
        $self.func.unwrap()
    };
}

/// Returns a reference to the given value in the current function.
macro_rules! value {
    ($self:ident, $value:expr) => {
        func!($self).dfg().value($value)
    };
}



/// Visitor implementation for RISCV.
impl<'a, W: Write> VisitorImpl<'a, W> {

    /// Visits the program.
    fn visit(&mut self) -> Result<()> {

        Type::set_ptr_size(4);

        // Data section
        writeln!(self.w, ".data")?;
        // TODO: Global pointer optimization

        let mut const_global_insts = Vec::new();
        for inst in self.program.inst_layout() {

            let inst_data = self.program.borrow_value(*inst);
            let raw_name = inst_data.name().as_ref().expect("[RISCV] global alloc should be named");
            if name_is_const(raw_name) {
                const_global_insts.push(*inst);
                continue
            }
            self.visit_global_inst(&inst_data)?;
            self.vm.insert_info(*inst, ValueInfo::GlobalOffset(*inst, 0));
        }

        // Code section
        writeln!(self.w, ".text")?;
        for inst in const_global_insts {
            self.visit_global_inst(&self.program.borrow_value(inst))?;
            self.vm.insert_info(inst, ValueInfo::GlobalOffset(inst, 0));
        }
        for function in self.program.func_layout().iter() {
            let func = self.program.func(*function);
            self.func = Some(func);

            self.vm.enter_func(self.program, *function);
            self.visit_func(func)?;
            self.vm.exit_func(self.w)?;
        }
        Ok(())
    }
  
    /// Generates the given function.
    fn visit_func(&mut self, func: &FunctionData) -> Result<()> {

        let func_name = &func.name()[1..];

        // Header
        let is_decl = func.dfg().bbs().is_empty();
        if is_decl {
            // Declaration need not be written in RISCV
            return Ok(())
        } else {
            writeln!(self.w, "\n  .globl {}", func_name)?
        }

        // Function name
        writeln!(self.w, "{}:", func_name)?;
        // println!("{}", func_name);

        // Function arguments
        // According to RISCV calling convention, the first 8 arguments
        // reside in a0~a7; then *(sp+0), *(sp+4)...
        for (i, p) in func.params().iter().enumerate() {
            if i < 8 {
                self.vm.insert_info(*p, ValueInfo::Reg((Reg::A0 as i32 + i as i32).into()))
            }
            else {
                self.vm.insert_info(*p, ValueInfo::Spilled( 4*(i as i32-8) ))
            }
        }

        self.vm.emit_code(Code::Prologue);

        // Function body
        // Visit in BFS order
        let mut visited = HashSet::new();
        let mut to_visit = VecDeque::new();

        to_visit.push_back(func.layout().entry_bb().expect("[RISCV] empty function"));
        while let Some(bb) = to_visit.pop_front() {
            if !visited.insert(bb) {
                continue;
            }
            if bb != func.layout().entry_bb().unwrap() {
                // Basic block name
                let bb_name = self.vm.program_info.get_bb(&bb).unwrap();
                // println!("\n[BB] {}", bb_name);
                self.vm.emit_code(format!("{}:", bb_name).into());
            }
            let bb_data = func.layout().bbs().node(&bb).unwrap();
            self.visit_bb(bb, bb_data)?;

            let last_inst = bb_data.insts().back_key().unwrap();
            match value!(self, *last_inst).kind() {
                ValueKind::Jump(jump) => {
                    to_visit.push_back(jump.target());
                },
                ValueKind::Branch(br) => {
                    to_visit.push_back(br.true_bb());
                    to_visit.push_back(br.false_bb());
                },
                ValueKind::Return(_) => {},
                _ => unreachable!(),
            }
        }
        Ok(())
    }
  
    /// Generates the given basic block.
    fn visit_bb(&mut self, bb: BasicBlock, node: &BasicBlockNode) -> Result<()> {
        // Prepares BB arguments
        let params = func!(self).dfg().bb(bb).params();
        if params.len() > 7 {
            todo!()
        }
        for (i, arg) in func!(self).dfg().bb(bb).params().iter().enumerate() {
            let reg = match i {
                0 => Reg::T0,
                1 => Reg::T1,
                2 => Reg::T2,
                3 => Reg::T3,
                4 => Reg::T4,
                5 => Reg::T5,
                6 => Reg::T6,
                _ => unreachable!(),
            };
            self.vm.insert_info(*arg, ValueInfo::Reg(reg));
        }

        // Instrustions in BB
        let insts: Vec<&Value> = node.insts().keys().collect();
        for inst in insts[..insts.len()-1].iter() {
            self.visit_local_inst(**inst, value!(self, **inst))?;
        }
        self.vm.emit_code(Code::Guard(None));
        self.visit_local_inst(**insts.last().unwrap(), value!(self, **insts.last().unwrap()))?;

        // Check for outlivers referenced;
        // they must be evicted.
        self.vm.sanity_check();

        Ok(())
    }
  
    /// Generates the given global instruction.
    fn visit_global_inst(&mut self, inst: &ValueData) -> Result<()> {
        let alloc = match inst.kind() {
            ValueKind::GlobalAlloc(alloc) => alloc,
            _ => panic!("[RISCV] invalid global instruction"),
        };
        let name = name_stripped(
            inst.name().as_ref().expect("[RISCV] global alloc should be named")
        );

        let init = self.program.borrow_value(alloc.init());
        writeln!(self.w, "  .globl {}", name)?;
        writeln!(self.w, "{}:", name)?;

        match inst.ty().kind() {
            TypeKind::Pointer(btype) => self.visit_global_const(btype, &init)?,
            _ => unreachable!(),
        }
        
        Ok(())
    }
  
    /// Generates the given local instruction.
    fn visit_local_inst(&mut self, inst: Value, data: &ValueData) -> Result<()> {

        // println!("[VISIT]: {:?}\n{:?}", inst, data);

        match data.kind() {
            ValueKind::Alloc(_) => self.visit_alloc(inst, data.ty()),
            ValueKind::Load(v) => self.visit_load(inst, v),
            ValueKind::Store(v) => self.visit_store(inst, v),
            ValueKind::GetPtr(v) => self.visit_getptr(inst, v),
            ValueKind::GetElemPtr(v) => self.visit_getelemptr(inst, v),
            ValueKind::Binary(v) => self.visit_binary(inst, v),
            ValueKind::Branch(v) => self.visit_branch(v),
            ValueKind::Jump(v) => self.visit_jump(v),
            ValueKind::Call(v) => self.visit_call(inst, v),
            ValueKind::Return(v) => self.visit_return(v),
            _ => panic!("[RISCV] invalid local instruction"),
        }?;
        self.vm.progress(inst);
        Ok(())
    }
  
    /// Generates for allocation.
    /// Allocation associates a stack position with a value.
    fn visit_alloc(&mut self, inst: Value, ty: &Type) -> Result<()> {
        let base = match ty.kind() {
            TypeKind::Pointer(base) => base,
            _ => panic!("[RISCV] invalid pointer type"),
        };
        let sz = base.size();
        let ofs = self.vm.alloc_stack(inst, sz);
        let info = ValueInfo::StackOffset(ofs);
        
        self.vm.insert_info(inst, info);
        Ok(())
    }
  
    /// Generates memory load.
    fn visit_load(&mut self, inst: Value, load: &Load) -> Result<()> {

        let offset = self.vm.value_to_offset(load.src());
        // Mind the order in which register are allocated.
        match offset {
            ValueInfo::RegOffset(reg_val, ofs) => {
                let base_reg = self.vm.value_to_register(reg_val, None).to_reg();
                let dest_reg = self.vm.alloc_register(Some(&[base_reg]));
                self.vm.emit_reg_load(ofs, base_reg, dest_reg);

                self.vm.insert_info(inst, ValueInfo::Reg(dest_reg));
            },
            ValueInfo::StackOffset(ofs) => {
                let dest_reg = self.vm.alloc_register(None);
                self.vm.emit_stack_load(ofs, dest_reg);

                self.vm.insert_info(inst, ValueInfo::Reg(dest_reg));
            },
            ValueInfo::GlobalOffset(globl, ofs) => {
                let dest_reg = self.vm.alloc_register(None);
                self.vm.emit_globl_load(globl, ofs, dest_reg);

                self.vm.insert_info(inst, ValueInfo::Reg(dest_reg));
            },
            _ => unreachable!(),
        }
        Ok(())
    }

    /// Generates aggregate store. This only happens for a
    /// StackOffset
    fn visit_agg_store(&mut self, temp_reg: Reg, mut ofs: i32, values: &[Value]) -> Result<()> {
        for value in values {
            match value!(self, *value).kind() {
                ValueKind::Aggregate(agg) => {
                    self.visit_agg_store(temp_reg, ofs, agg.elems())?;
                },
                ValueKind::Integer(i) => {
                    self.vm.emit_li(i.value(), temp_reg);
                    self.vm.emit_stack_store(ofs, temp_reg);
                },
                _ => unreachable!()
            }
            ofs += value!(self, *value).ty().size() as i32;
        }
        Ok(())
    }
  
    /// Generates memory store.
    fn visit_store(&mut self, _: Value, store: &Store) -> Result<()> {

        let offset = self.vm.value_to_offset(store.dest());

        // Handle pseudo aggregate store
        match value!(self, store.value()).kind() {
            ValueKind::Aggregate(agg) => {
                if let ValueInfo::StackOffset(ofs) = offset {
                    let temp_reg = self.vm.alloc_register(None);
                    self.vm.evict(temp_reg);
                    self.visit_agg_store(temp_reg, ofs, agg.elems())?;
                }
                else {
                    panic!("[RISCV] aggregate store must be targeted at StackOffset")
                }
                return Ok(())
            },
            _ => {},
        }

        let src_reg = if let Some((imm, _)) = to_i32(value!(self, store.value())) {
            let src_reg = self.vm.alloc_register(None);
            self.vm.evict(src_reg);
            self.vm.emit_li(imm, src_reg);
            src_reg
        }
        else {
            self.vm.value_to_register(store.value(), None).to_reg()
        };
        match offset {
            ValueInfo::RegOffset(reg_val, ofs) => {
                let base_reg = self.vm.alloc_register(Some(&[src_reg]));
                let base_reg = self.vm.value_to_register(reg_val, Some((base_reg, false))).to_reg();
                self.vm.emit_reg_store(ofs, base_reg, src_reg);
            },
            ValueInfo::StackOffset(ofs) => {
                self.vm.emit_stack_store(ofs, src_reg);
            },
            ValueInfo::GlobalOffset(globl, ofs) => {
                self.vm.emit_globl_store(globl, ofs, src_reg);
            },
            _ => unreachable!(),
        }
        
        Ok(())
    }
  
    /// Generates pointer calculation.
    fn visit_getptr(&mut self, inst: Value, gp: &GetPtr) -> Result<()> {
        let values = self.program.borrow_values();
        let kind = if let Some(data) = values.get(&gp.src()) {
            data.ty().kind()
        }
        else {
            value!(self, gp.src()).ty().kind() 
        };

        let sz: usize = match kind {
            TypeKind::Pointer(btype) => {
                btype.size()
            },
            _ => unreachable!(),
        };
        self.visit_get_offset(inst, gp.src(), gp.index(), sz)
    }
  
    /// Generates element pointer calculation.
    fn visit_getelemptr(&mut self, inst: Value, gep: &GetElemPtr) -> Result<()> {
        let values = self.program.borrow_values();
        let kind = if let Some(data) = values.get(&gep.src()) {
            data.ty().kind()
        }
        else {
            value!(self, gep.src()).ty().kind() 
        };

        let sz: usize = match kind {
            TypeKind::Pointer(btype) => {
                match btype.kind() {
                    TypeKind::Array(btype, _) => {
                        btype.size()
                    },
                    _ => unreachable!(),
                }
            },
            _ => unreachable!(),
        };
        self.visit_get_offset(inst, gep.src(), gep.index(), sz)
    }

    /// Generates pointer offset calculation.
    fn visit_get_offset(&mut self, inst: Value, src: Value, index: Value, step: usize)  -> Result<()> {
        if let Some((index, _)) = to_i32(value!(self, index)) {
            let mut base = self.vm.value_to_offset(src);
            base.add_offset(index * step as i32);
            self.vm.insert_info(inst, base);
            Ok(())
        }
        else {
            // Instantiate calculation
            let index = self.vm.value_to_register(index, None).to_reg();

            let res = self.vm.alloc_register(Some(&[index]));
            self.vm.insert_info(inst, ValueInfo::Reg(res));
            self.vm.emit_muli(index, step as i32, res);

            let src_reg = self.vm.alloc_register(Some(&[res, index]));
            let src_reg = self.vm.value_to_register(src, Some((src_reg, false))).to_reg();
            self.vm.emit_add(src_reg, res, res);
            
            Ok(())
        }
    }
  
    /// Generates binary operation.
    fn visit_binary(&mut self, inst: Value, bin: &Binary) -> Result<()> {
        let op = bin.op();
        let vlhs = bin.lhs();
        let vrhs = bin.rhs();

        match (to_i32(value!(self, vlhs)), to_i32(value!(self, vrhs))) {
            (Some((_, _)), Some((_, _))) => panic!("[RISCV] binary with both constant operand"),
            (Some((l, _)), None) => {
                self.visit_binary_limm(inst, op, l, vrhs)
            },
            (None, Some((r, _))) => {
                self.visit_binary_rimm(inst, op, vlhs, r)
            },
            (None, None) => {
                let lhs = self.vm.value_to_register(vlhs, None).to_reg();
                let rhs = self.vm.alloc_register(Some(&[lhs]));
                let rhs = self.vm.value_to_register(vrhs, Some((rhs, false))).to_reg();

                let res = self.vm.alloc_register(Some(&[lhs, rhs]));
                self.vm.insert_info(inst, ValueInfo::Reg(res));
                self.visit_binary_no_imm(op, lhs, rhs, res)
            }
        }
    }

    fn visit_binary_limm(&mut self, inst: Value, mut op: BinaryOp, imm: i32, rhs: Value) -> Result<()> {
        match op {
            BinaryOp::Div | BinaryOp::Mod | BinaryOp::Sub => {
                // These must be handled as-is
                let rhs = self.vm.value_to_register(rhs, None).to_reg();
                let lhs = self.vm.alloc_register(Some(&[rhs]));
                self.vm.evict(lhs);
                self.vm.emit_li(imm, lhs);

                let res = self.vm.alloc_register(Some(&[lhs, rhs]));
                self.vm.insert_info(inst, ValueInfo::Reg(res));
                return self.visit_binary_no_imm(op, lhs, rhs, res)
            },
            BinaryOp::Ge => op = BinaryOp::Le,
            BinaryOp::Le => op = BinaryOp::Ge,
            BinaryOp::Gt => op = BinaryOp::Lt,
            BinaryOp::Lt => op = BinaryOp::Gt,
            BinaryOp::Add | BinaryOp::Eq |BinaryOp::NotEq | BinaryOp::Mul => {},
            BinaryOp::Sar | BinaryOp::Shl | BinaryOp::Shr |
            BinaryOp::And | BinaryOp::Or | BinaryOp::Xor 
                => unreachable!(),
        }
        self.visit_binary_rimm(inst, op, rhs, imm)
    }

    fn visit_binary_rimm(&mut self, inst: Value, op: BinaryOp, lhs: Value, imm: i32) -> Result<()> {
        let lhs = self.vm.value_to_register(lhs, None).to_reg();
        let res = self.vm.alloc_register(Some(&[lhs]));
        self.vm.insert_info(inst, ValueInfo::Reg(res));

        match op {
            BinaryOp::Add => self.vm.emit_addi(lhs, imm, res),
            BinaryOp::Sub => self.vm.emit_addi(lhs, -imm, res),
            BinaryOp::Mul => self.vm.emit_muli(lhs, imm, res),
            BinaryOp::Div | BinaryOp::Mod => {
                if imm.count_ones() == 1 && imm > 0 {
                    // Power of 2 optimization
                    let mut shamt: u32 = 0;
                    while (1 << shamt) as i32 != imm {
                        shamt += 1;
                    }
                    if matches!(op, BinaryOp::Div) {
                        self.vm.emit_code(format!("  srai {}, {}, {}", res, lhs, shamt).into());
                        self.vm.emit_code(format!("  slt {}, {}, {}", TEMP_REG, res, Reg::Zero).into());
                        self.vm.emit_code(format!("  add {}, {}, {}", res, res, TEMP_REG).into());
                    }
                    else {
                        self.vm.emit_code(format!("  andi {}, {}, {}", res, lhs, ((1u32 << shamt)-1) as i32).into());
                    }
                }
                else if imm == 998244353 {
                    // Optimize for non-power-of-two
                    // TODO: generalize
                    self.vm.emit_code(format!("  li {}, {}", TEMP_REG, 0x1135c811).into());
                    self.vm.emit_code(format!("  mulh {}, {}, {}", res, lhs, TEMP_REG).into());
                    self.vm.emit_code(format!("  srai {}, {}, {}", res, res, 0x1a).into());

                    self.vm.emit_code(format!("  srai {}, {}, {}", TEMP_REG, lhs, 0x1f).into());
                    self.vm.emit_code(format!("  sub {}, {}, {}", res, res, TEMP_REG).into());

                    if matches!(op, BinaryOp::Mod) {
                        self.vm.emit_code(format!("  li {}, {}", TEMP_REG, 998244353).into());
                        self.vm.emit_code(format!("  mul {}, {}, {}", res, res, TEMP_REG).into());
                        self.vm.emit_code(format!("  sub {}, {}, {}", res, lhs, res).into());
                    }
                }
                else {
                    let rhs = self.vm.alloc_register(Some(&[lhs, res]));
                    self.vm.evict(rhs);
                    self.vm.emit_li(imm, rhs);

                    return self.visit_binary_no_imm(op, lhs, rhs, res)
                }
            },
            BinaryOp::Gt | BinaryOp::Le => {
                let rhs = self.vm.alloc_register(Some(&[lhs, res]));
                self.vm.evict(rhs);
                self.vm.emit_li(imm, rhs);

                return self.visit_binary_no_imm(op, lhs, rhs, res)
            },
            BinaryOp::Eq => {
                if imm != 0 {
                    self.vm.emit_addi(lhs, -imm, TEMP_REG);
                    self.vm.emit_code(format!("  seqz {}, {}", res, TEMP_REG).into());
                }
                else {
                    self.vm.emit_code(format!("  seqz {}, {}", res, lhs).into());
                }
            },
            BinaryOp::NotEq => {
                if imm != 0 {
                    self.vm.emit_addi(lhs, -imm, TEMP_REG);
                    self.vm.emit_code(format!("  snez {}, {}", res, TEMP_REG).into());
                }
                else {
                    self.vm.emit_code(format!("  snez {}, {}", res, lhs).into());
                }
            },
            BinaryOp::Lt => {
                self.vm.emit_slti(lhs, imm, res);
            },
            BinaryOp::Ge => {
                self.vm.emit_slti(lhs, imm, res);
                self.vm.emit_code(format!("  xori {}, {}, {}", res, res, 1).into());
            },
            BinaryOp::Sar | BinaryOp::Shl | BinaryOp::Shr |
            BinaryOp::And | BinaryOp::Or | BinaryOp::Xor 
                => unreachable!(),
        }
        Ok(())
    }

    fn visit_binary_no_imm(
        &mut self, 
        op: BinaryOp, 
        lhs: Reg, 
        rhs: Reg,
        res: Reg,
    ) -> Result<()> {
        match op {
            BinaryOp::Add => self.vm.emit_add(lhs, rhs, res),
            BinaryOp::Sub => self.vm.emit_code(format!("  sub {}, {}, {}", res, lhs, rhs).into()),
            BinaryOp::Div => self.vm.emit_code(format!("  div {}, {}, {}", res, lhs, rhs).into()),
            BinaryOp::Mod => self.vm.emit_code(format!("  rem {}, {}, {}", res, lhs, rhs).into()),
            BinaryOp::Mul => self.vm.emit_code(format!("  mul {}, {}, {}", res, lhs, rhs).into()),
            BinaryOp::Eq => {
                self.vm.emit_code(format!("  sub {}, {}, {}", TEMP_REG, lhs, rhs).into());
                self.vm.emit_code(format!("  seqz {}, {}", res, TEMP_REG).into());
            },
            BinaryOp::NotEq => {
                self.vm.emit_code(format!("  sub {}, {}, {}", TEMP_REG, lhs, rhs).into());
                self.vm.emit_code(format!("  snez {}, {}", res, TEMP_REG).into());
            },
            BinaryOp::Lt => {
                self.vm.emit_code(format!("  slt {}, {}, {}", res, lhs, rhs).into());
            },
            BinaryOp::Gt => {
                self.vm.emit_code(format!("  sgt {}, {}, {}", res, lhs, rhs).into());
            },
            BinaryOp::Le => {
                // Inverse Gt
                self.vm.emit_code(format!("  sgt {}, {}, {}", res, lhs, rhs).into());
                self.vm.emit_code(format!("  xori {}, {}, {}", res, res, 1).into());
            },
            BinaryOp::Ge => {
                // Inverse Lt
                self.vm.emit_code(format!("  slt {}, {}, {}", res, lhs, rhs).into());
                self.vm.emit_code(format!("  xori {}, {}, {}", res, res, 1).into());
            },
            BinaryOp::Sar | BinaryOp::Shl | BinaryOp::Shr |
            BinaryOp::And | BinaryOp::Or | BinaryOp::Xor 
                => unreachable!(),
        }
        Ok(())
     }
  
    /// Generates branch.
    fn visit_branch(&mut self, br: &Branch) -> Result<()> {

        let cond = br.cond();
        let true_bb = br.true_bb();
        let false_bb = br.false_bb();

        let reg = if let Some((imm, _)) = to_i32(value!(self, cond)) {
            self.vm.emit_li(imm, TEMP_REG);
            TEMP_REG
        }
        else {
            self.vm.value_to_register(cond, None).to_reg()
        };

        let true_name = if br.true_args().is_empty() {
            self.vm.program_info.get_bb(&true_bb).unwrap().clone()
        }
        else {
            self.vm.program_info.get_bb(&true_bb).unwrap().clone() + "_pro"
        };

        self.vm.emit_code(format!("  bnez {}, {}", 
            reg, 
            true_name,
        ).into());

        // The False branch first
        _ = self.visit_jump_argument(false_bb, br.false_args());
       
        // The True branch, if arguments exist
        if !br.true_args().is_empty() {
            self.vm.emit_code(format!("{}:", true_name).into());
            _ = self.visit_jump_argument(true_bb, br.true_args());
        }

        Ok(())
    }
  
    /// Generates jump.
    fn visit_jump(&mut self, jump: &Jump) -> Result<()> {
        self.visit_jump_argument(jump.target(), jump.args())
    }

    fn visit_jump_argument(
        &mut self,
        target: BasicBlock,
        args: &[Value],
    ) -> Result<()> {
        if args.len() > 7 {
            todo!()
        }

        for (i, arg) in args[..std::cmp::min(7, args.len())].as_ref().iter().enumerate() {
            let reg = match i {
                0 => Reg::T0,
                1 => Reg::T1,
                2 => Reg::T2,
                3 => Reg::T3,
                4 => Reg::T4,
                5 => Reg::T5,
                6 => Reg::T6,
                _ => unreachable!(),
            };
            if let Some((imm, _)) = to_i32(value!(self, *arg)) {
                self.vm.emit_li(imm, reg);
            }
            else {
                self.vm.value_to_register(*arg, Some((reg, true)));
            }
        }
        self.vm.emit_code(format!("  j {}", self.vm.program_info.get_bb(&target).unwrap()).into());
        Ok(())
    }
  
    /// Generates function call.
    fn visit_call(&mut self, inst: Value, call: &Call) -> Result<()> {

        let ret_is_unit = match self.program.func(call.callee()).ty().kind() {
            TypeKind::Function(_, ret_ty) => ret_ty.is_unit(),
            _ => unreachable!(),
        };
        
        // Stack movement
        let amt = if call.args().len() > 8 { 
            stack_aligned!(4 * (call.args().len() - 8))
        }
        else {
            0
        };
        if amt > 0 {
            self.vm.emit_addi(Reg::Sp, -(amt as i32), Reg::Sp)
        }

        // Sets up arguments
        for (i, arg) in call.args().iter().enumerate() {
            if i < 8 {
                let reg = Reg::from(Reg::A0 as i32 + i as i32);
                if let Some((imm, _)) = to_i32(value!(self, *arg)) {
                    self.vm.evict(reg);
                    self.vm.emit_li(imm, reg);
                }
                else {
                    self.vm.value_to_register(*arg, Some((reg, true)));
                }
            }
            else {
                // Must not disturb register arguments
                let reg = self.vm.alloc_register(Some(&[
                    Reg::A0, Reg::A1, Reg::A2, Reg::A3,
                    Reg::A4, Reg::A5, Reg::A6, Reg::A7,
                ]));
                if let Some((imm, _)) = to_i32(value!(self, *arg)) {
                    self.vm.emit_li(imm, reg);
                    self.vm.emit_reg_store(4 * (i as i32 - 8), Reg::Sp, reg);
                }
                else {
                    let reg = self.vm.value_to_register(*arg, Some((reg, false))).to_reg();
                    self.vm.emit_reg_store(4 * (i as i32 - 8), Reg::Sp, reg);
                }
            }
        }

        // Saves all caller-saved registers that will still be in-use.
        if !ret_is_unit {
            // A0 must be evicted because it is replaced by return value.
            // TODO: this can be optimized away
            self.vm.evict(Reg::A0);
        }
        for reg in (0..32).map(|i| Reg::from(i)) {
            if reg.is_caller_saved() && self.vm.register_used_after(reg, inst) {
                self.vm.evict(reg);
            }
        }

        // Ready to call
        let name = &self.program.func(call.callee()).name()[1..];
        self.vm.emit_code(format!("  call {}", name).into());

        // Bind return value
        if !ret_is_unit {
            assert!(self.vm.register_to_value(Reg::A0).is_none());
            self.vm.insert_info(inst, ValueInfo::Reg(Reg::A0));
        }

        // Clean-up; restore the stack.
        if amt > 0 {
            self.vm.emit_addi(Reg::Sp, amt as i32, Reg::Sp)
        }
        
        Ok(())
    }
  
    /// Generates function return.
    fn visit_return(&mut self, ret: &Return) -> Result<()> {
        
        if ret.value().is_none() {
            // No return value
            // The epilogue code takes care of restoring the
            // stack pointer and frame pointer, as well as
            // getting the return address into `ra` and the
            // final `ret`.
            self.vm.emit_code(Code::Epilogue);
            return Ok(())
        }

        // Has return value; prepare the return value
        // into `a0`.
        let ret_val = ret.value().unwrap();
        if let Some((imm, _)) = to_i32(value!(self, ret_val)) {
            self.vm.emit_li(imm, Reg::A0);
        }
        else {
            self.vm.value_to_register(ret_val, Some((Reg::A0, true)));
        }
        self.vm.emit_code(Code::Epilogue);
        Ok(())
    }
  
  
    /// Generates the given global constant.
    fn visit_global_const(&mut self, ty: &Type, value: &ValueData) -> Result<()> {

        match ty.kind() {
            TypeKind::Array(btype, bound) => {
                // Recurse
                match value.kind() {
                    ValueKind::Aggregate(agg) => {
                        for v in agg.elems() {
                            let data = self.program.borrow_value(*v);
                            self.visit_global_const(btype, &data)?
                        }
                        Ok(())
                    },
                    ValueKind::ZeroInit(_) | ValueKind::Undef(_) => {
                        for _ in 0..*bound {
                            self.visit_global_const(btype, value)?
                        }
                        Ok(())
                    }
                    _ => unreachable!(),
                }
            },
            TypeKind::Int32 => {
                match value.kind() {
                    ValueKind::Integer(v) => 
                        writeln!(self.w, "  .word {}", v.value()),
                    ValueKind::ZeroInit(_) | ValueKind::Undef(_)
                        => writeln!(self.w, "  .word 0"),
                    _ => unreachable!(),
                }
            },
            _ => unreachable!(),
        }
    }
  
}
