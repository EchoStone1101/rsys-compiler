//! Implements all utility methods that a backend
//! visitor may want to have precalculated, without
//! tediously querying a `Program` itself.

#[allow(unused)]
use koopa::ir::*;
use std::collections::{HashMap, HashSet};
use crate::opt;

#[derive(Debug, Default)]
pub(in crate::backend) struct ProgramInfo {

    globls: HashMap<Value, String>,

    bbs: HashMap<BasicBlock, String>,

    kills: HashMap<Value, Vec<Value>>,

    outlivers: HashSet<Value>,

    allocs: HashSet<Value>,

    load_store_info: HashMap<BasicBlock, HashMap<Value, Value>>,

}

impl ProgramInfo {
    pub fn init(&mut self, program: &Program) {
        self.globls.extend(
            program.inst_layout().iter()
                .map(|v| (*v, name_stripped(
                    program.borrow_value(*v).name().as_ref().expect("[PROGINFO] global must be named")
        ))))
    }

    pub fn new_func(&mut self, program: &Program, func: Function) {

        let mut pass = opt::ElimLoadStore;
        self.load_store_info = pass.run(func, program.func(func));

        self.bbs.clear();
        let func_name = program.func(func).name();
        let mut name_map: HashMap<String, usize> = HashMap::new();
        self.bbs.extend(program.func(func).dfg()
            .bbs()
            .iter()
            .filter_map(|(bb, bb_data)| {
                if let Some(name) = bb_data.name() {
                    let mut new_name = format!("{}_{}", &func_name[1..], &name[1..]);
                    if let Some(rep) = name_map.get_mut(&new_name) {
                        new_name += format!("_{}", rep).as_str();
                        *rep += 1;
                    }
                    else {
                        _ = name_map.insert(new_name.clone(), 0);
                    }
                    Some((*bb, new_name))
                }
                else {
                    None
                }
            })
        );

        self.init_kills(program, func);

    }

    fn init_kills(&mut self, program: &Program, func: Function) {

        self.kills.clear();
        self.outlivers.clear();
        self.allocs.clear();
        // Globals are always outlivers.
        self.outlivers.extend(self.globls.keys());
        self.allocs.extend(self.globls.keys());

        let func = program.func(func);
        let mut index = 0usize;
        let mut inst_indices = HashMap::new();
        for (_, bb) in func.layout().bbs() {
            for (inst, _) in bb.insts() {
                assert!(inst_indices.insert(*inst, index).is_none(),
                    "[PROGINFO] duplicate inst Value");
                index += 1;
            }
        }

        for (bb, bb_data) in func.layout().bbs() {

            let to_check = bb_data.insts().iter()
                .map(|(inst, _)| inst)
                .chain(func.dfg().bb(*bb).params().iter());

            let mut last_use_of: HashMap<Value, Value> = HashMap::new();

            for inst in to_check {
                // println!("{:?}", inst);

                assert!(self.kills.insert(*inst, vec![]).is_none());

                let inst_data = func.dfg().value(*inst);
                if matches!(inst_data.kind(), ValueKind::Alloc(_)) {
                    self.allocs.insert(*inst);
                }

                let mut out_lives = false;
                for user in inst_data.used_by() {
                    let user_bb = func.layout().parent_bb(*user);
                    if user_bb.is_none() {
                        continue
                    }
                    if user_bb.unwrap() != *bb {
                        // TODO: currently naive
                        // panic!("[PROGINFO] value use outlives BB, and is not an alloc")
                        out_lives = true;
                        break;
                    }
                }
                if out_lives {
                    // println!("outliver: {:?} {:?}", inst, inst_data);
                    self.outlivers.insert(*inst);
                    // Outlivers are never killed
                    continue
                }

                let last_use = inst_data.used_by().iter()
                    .filter(|user| func.layout().parent_bb(**user).is_some())
                    .max_by_key(|a| *inst_indices.get(a).unwrap());
                if last_use.is_some() {
                    _ = last_use_of.insert(*inst, *last_use.unwrap());
                }
                else {
                    // Unused value, killed by itself
                    _ = last_use_of.insert(*inst, *inst);
                }
            }

            // A depended `Value`, for simplicity, is killed at the end
            // of the block.
            let last_inst = bb_data.insts().back_key().expect("[PROGINFO] empty bb");
            // println!("[INFO] current shadow at {:?}: {:?}", self.get_bb(bb), self.load_store_info.get(&bb).unwrap());
            for (_, depended) in self.load_store_info.get(&bb).unwrap().iter() {
                if let Some(user) = last_use_of.get_mut(depended) {
                    match func.dfg().value(*depended).kind() {
                        ValueKind::Integer(_) | ValueKind::Store(_) |
                        ValueKind::Undef(_) | ValueKind::ZeroInit(_)
                            => continue,
                        ValueKind::Aggregate(_) => unreachable!(),
                        _ => {},
                    }
                    // println!("[INFO] fixing {:?}, delay kill to {:?}", depended, last_inst);
                    *user = *last_inst;
                }
            }

            for (inst, mut user) in last_use_of.iter() {
                let mut cur_inst = inst;
                loop {
                    // If used by a GEP or GP as source, recursively find its
                    // last use.
                    match func.dfg().value(*user).kind() {
                        ValueKind::GetElemPtr(gep) => {
                            if *cur_inst == gep.src() {
                                let new_user = last_use_of.get(user).unwrap();
                                if *new_user == *user {
                                    break;
                                }
                                cur_inst = user;
                                user = new_user;
                            }
                            else {
                                break;
                            }
                        },
                        ValueKind::GetPtr(gp) => {
                            if *cur_inst == gp.src() {
                                let new_user = last_use_of.get(user).unwrap();
                                if *new_user == *user {
                                    break;
                                }
                                cur_inst = user;
                                user = new_user;
                            }
                            else {
                                break;
                            }
                        },
                        _ => break
                    }
                }
                // println!("{:?} -> {:?}", inst, user);
                self.kills.get_mut(user).unwrap().push(*inst);
            }
        }

        // TODO: currently naive
        for param in func.params() {
            let users = func.dfg().value(*param).used_by();
            assert!(users.len() <= 1, "[PROGINFO] function argument should be used not more than once");
            for user in users {
                if !self.kills.contains_key(&user) {
                    self.kills.insert(*user, vec![*param]);
                }
                else {
                    self.kills.get_mut(user).unwrap().push(*param);
                }
            }
        }

        // println!("{:#?}", self.kills);
        
    }

    pub fn is_globl(&self, v: &Value) -> bool {
        self.globls.contains_key(v)
    }

    pub fn get_globl(&self, v: &Value) -> Option<&String> {
        self.globls.get(v)
    }

    pub fn get_kills(&self, v: &Value) -> Option<&Vec<Value>> {
        self.kills.get(v)
    }

    pub fn get_bb(&self, b: &BasicBlock) -> Option<&String> {
        self.bbs.get(b)
    }

    pub fn is_outliver(&self, inst: &Value) -> bool {
        self.outlivers.contains(inst)
    }

    pub fn is_alloc(&self, inst: &Value) -> bool {
        self.allocs.contains(inst)
    }

    pub fn get_shadowed_by(&self, bb: &BasicBlock, inst: &Value) -> Option<Value> {
        self.load_store_info
            .get(bb)
            .map_or(None, |bb_map| {
                bb_map.get(inst).cloned()
            })
    }

    // Collects value uses
    // self.value_uses.clear();
    // let func = program.func(func);
    
    // for inst in inst_indices.keys() {
    //     let mut uses: Vec<usize> = func.dfg().value(*inst).used_by()
    //         .iter()
    //         .filter_map(|u| inst_indices.get(u).cloned())
    //         .collect();
    //     uses.sort_by(|a, b| b.cmp(a));
    //     _ = self.value_uses.insert(*inst, uses);
    // }
    // for val in program.inst_layout().iter() {
    //     let mut uses: Vec<usize> = program.borrow_value(*val).used_by()
    //         .iter()
    //         .filter_map(|u| inst_indices.get(u).cloned())
    //         .collect();
    //     uses.sort_by(|a, b| b.cmp(a));
    //     _ = self.value_uses.insert(*val, uses);
    // }
    // for (_, bb) in func.dfg().bbs() {
    //     for p in bb.params() {
    //         if let Some(data) = func.dfg().values().get(p) {
    //             let mut uses: Vec<usize> = data.used_by()
    //                 .iter()
    //                 .filter_map(|u| inst_indices.get(u).cloned())
    //                 .collect();
    //             uses.sort_by(|a, b| b.cmp(a));
    //             _ = self.value_uses.insert(*p, uses);
    //         }
    //     }
    // }

}

pub(in crate::backend) fn name_is_const(n: &str) -> bool {
    n.ends_with("_con")
}

pub(in crate::backend) fn name_stripped(n: &str) -> String {
    String::from(&n[1..n.len()-4])
}