//! Implements optimization passes for the Koopa IR.

use koopa::opt::*;
use koopa::ir::{*, builder_traits::*};
use std::collections::{HashMap, HashSet, VecDeque};
use crate::middleend::ir::to_i32;


pub fn sanity_check(data: &mut FunctionData) {
    let empty_bbs: Vec<BasicBlock> = data.layout().bbs().iter()
        .filter(|(_, data)| data.insts().is_empty())
        .map(|(bb,_) | *bb)
        .collect();
    for bb in empty_bbs.iter() {
        data.layout_mut().bbs_mut().remove(bb);
        data.dfg_mut().remove_bb(*bb);
    }

    // Sanity checks
    let entry = data.layout().entry_bb().expect("should have at least one non-empty entry");
    let non_target_bbs: Vec<BasicBlock> = data.dfg().bbs().iter()
        .filter(|(bb, data)| **bb != entry && data.used_by().is_empty())
        .map(|(bb,_) | *bb)
        .collect();
    assert!(non_target_bbs.is_empty(), "[AST] all but the entry BB should be used at least once");
}

/// Perform reachability analysis over BasicBlocks,
/// and remove unreachable blocks in a function.
pub struct ElimUnreachableBlock;

impl FunctionPass for ElimUnreachableBlock {

    fn run_on(&mut self, _func: koopa::ir::Function, data: &mut koopa::ir::FunctionData) {

        let entry_bb = data.layout().entry_bb();
        if entry_bb.is_none() {
            return;
        }
        let entry_bb = entry_bb.unwrap();

        let mut reachable = HashSet::new();
        let mut to_visit = VecDeque::new();
        to_visit.push_back(entry_bb);

        // Mark all reachable blocks
        while let Some(bb) = to_visit.pop_front() {
            if !reachable.insert(bb) {
                // Visisted
                continue;
            }
            let last_inst = *data.layout_mut().bb_mut(bb).insts().back_key()
                .expect("[OPT] encountered empty BB");
            let mut new_target = None;

            match data.dfg().value(last_inst).kind() {
                ValueKind::Branch(br) => {
                    let cond = br.cond();
                    let true_bb = br.true_bb();
                    let false_bb = br.false_bb();

                    let (true_reach, false_reach) = 
                        if let Some((i, _)) = to_i32(data.dfg().value(cond)) {
                            // Rewrite this branch into a jump
                            if i == 0 {
                                new_target = Some((false_bb, br.false_args()));
                                (false, true)
                            }
                            else {
                                new_target= Some((true_bb, br.true_args()));
                                (true, false)
                            }
                        }
                        else {
                            (true, true)
                        };
                    
                    if true_reach && !reachable.contains(&true_bb) {
                        to_visit.push_back(true_bb);
                    }
                    if false_reach && !reachable.contains(&false_bb) {
                        to_visit.push_back(false_bb);
                    }
                },
                ValueKind::Jump(jmp) => {
                    // Definite next
                    let next_bb = jmp.target();
                    if !reachable.contains(&next_bb) {
                        to_visit.push_back(next_bb);
                    }
                },
                ValueKind::Return(_) => {/* No next */},
                _ => panic!("[OPT] non SSA IR"),
            }

            if let Some((target, args)) = new_target {
                // Replace with jump
                let args = Vec::from(args);
                let jump =  data.dfg_mut().new_value()
                    .jump_with_args(target, args);
                let (inst, _) = data.layout_mut().bb_mut(bb).insts_mut().pop_back().unwrap();
                data.dfg_mut().remove_value(inst);
                data.layout_mut().bb_mut(bb).insts_mut().push_key_back(jump).unwrap();
            }
        }

        // Remove unreachable BBs
        // Note: must first clear all BBs, then remove them.
        let unreachable: Vec<BasicBlock> = data.layout().bbs()
            .iter()
            .filter_map(|(bb, _)| 
                if !reachable.contains(bb) {
                    Some(bb)
                }
                else {
                    None
                }
            )
            .cloned()
            .collect();
        
        for bb in unreachable.iter() {
            let last_inst = *data.layout_mut().bb_mut(*bb).insts_mut().back_key().unwrap();
            data.dfg_mut().remove_value(last_inst);
            data.layout_mut().bb_mut(*bb).insts_mut().clear();
            _ = data.layout_mut().bbs_mut().remove(bb);
        }
        for bb in unreachable.iter() {
            data.dfg_mut().remove_bb(*bb);
        }

        sanity_check(data);
    }
}


/// Remove useless blocks, which are blocks that
/// contains a single *no-param* jump instruction 
/// only, and the target is not itself.
pub struct ElimUselessBlock;

fn is_useless(data: &mut FunctionData, bb: BasicBlock) -> Option<BasicBlock> {
    if data.layout_mut().bb_mut(bb).insts().len() != 1 {
        return None
    }
    let last_inst = *data.layout_mut().bb_mut(bb).insts().back_key()
        .expect("[OPT] encountered empty BB");
    match data.dfg().value(last_inst).kind() {
        ValueKind::Jump(jump) => {
            if jump.args().is_empty()
                && jump.target() != bb {
                    return Some(jump.target())
                }
        }
        _ => {},
    }
    return None
}

fn find_resolved_bb(
    data: &mut FunctionData, 
    bb: BasicBlock,
    resolved_bb: &mut HashMap<BasicBlock, Option<BasicBlock>>,
) -> Option<BasicBlock> {

    if let Some(res_bb) = resolved_bb.get(&bb) {
        return *res_bb
    }
    else {
        if let Some(target) = is_useless(data, bb) {
            // To avoid loops; in case they do occur, the last one
            // will be viewed as not useless.
            resolved_bb.insert(bb, None);
            let mut res_bb = find_resolved_bb(data, target, resolved_bb);
            if res_bb.is_none() {
                res_bb = Some(target);
            }
            else {
                if res_bb.unwrap() == bb {
                    res_bb = None;
                }
            }
            resolved_bb.insert(bb, res_bb);
            return res_bb
        }
        else {
            resolved_bb.insert(bb, None);
            return None
        }
    }
}

impl FunctionPass for ElimUselessBlock {
    fn run_on(&mut self, _func: Function, data: &mut FunctionData) {

        let entry_bb = data.layout().entry_bb();
        if entry_bb.is_none() {
            return;
        }
        let entry_bb = entry_bb.unwrap();

        // Map from a useless block to the furthest non-useless block
        // down the line, or None if key is not a useless block.
        let mut resolved_bb = HashMap::<BasicBlock, Option<BasicBlock>>::new();
        // The entry BB is never useless.
        resolved_bb.insert(entry_bb, None);
        let all_bbs: Vec<BasicBlock> = data.layout().bbs().keys().cloned().collect();
        for bb in all_bbs.iter() {
            find_resolved_bb(data, *bb, &mut resolved_bb);
        }

        // for (bb, target) in resolved_bb.iter() {
        //     println!("{} -> {:?}",
        //         data.dfg().bb(*bb).name().as_ref().unwrap(),
        //         target.map(|bb| {
        //             data.dfg().bb(bb).name().as_ref().unwrap()
        //         })
        //     );
        // }

        // Reposition jumps in non-useless blocks
        for bb in all_bbs.iter() {

            if resolved_bb.get(bb).unwrap().is_some() {
                // Useless blocks are skipped
                continue;
            }

            let last_inst = *data.layout_mut().bb_mut(*bb).insts().back_key()
                .expect("[OPT] encountered empty BB");

            match data.dfg().value(last_inst).kind().clone() {
                ValueKind::Branch(br) => {
                    let new_true_bb = *resolved_bb.get(&br.true_bb())
                        .expect("[OPT] target should be resolved");
                    let new_false_bb = *resolved_bb.get(&br.false_bb())
                        .expect("[OPT] target should be resolved");
                    match (new_true_bb, new_false_bb) {
                        (Some(new_true_bb), Some(new_false_bb)) => {
                            data.dfg_mut().replace_value_with(last_inst)
                                .branch_with_args(
                                    br.cond(),
                                    new_true_bb, new_false_bb,
                                    vec![], vec![]
                                );
                        },
                        (Some(new_true_bb), None) => {
                            data.dfg_mut().replace_value_with(last_inst)
                                .branch_with_args(
                                    br.cond(),
                                    new_true_bb, br.false_bb(),
                                    vec![], br.false_args().into()
                                );
                        },
                        (None, Some(new_false_bb)) => {
                            data.dfg_mut().replace_value_with(last_inst)
                                .branch_with_args(
                                    br.cond(),
                                    br.true_bb(), new_false_bb,
                                    br.true_args().into(), vec![]
                                );
                        },
                        (None, None) => {},
                    }
                },
                ValueKind::Jump(jmp) => {
                    let new_target = *resolved_bb.get(&jmp.target())
                        .expect("[OPT] target should be resolved");
                    if let Some(new_target) = new_target {
                        data.dfg_mut().replace_value_with(last_inst)
                            .jump(new_target);
                    }
                },
                ValueKind::Return(_) => {/* No next */},
                _ => panic!("[OPT] non SSA IR"),
            }
        }

        // Finally remove the useless blocks
        let useless_bbs: Vec<BasicBlock> = all_bbs
            .into_iter()
            .filter_map(|bb| {
                if resolved_bb.get(&bb).unwrap().is_some() {
                    Some(bb)
                }
                else {
                    None
                }
            })
            .collect();
        for bb in useless_bbs.iter() {
            let last_inst = *data.layout_mut().bb_mut(*bb).insts_mut().back_key().unwrap();
            data.dfg_mut().remove_value(last_inst);
            data.layout_mut().bb_mut(*bb).insts_mut().clear();
            _ = data.layout_mut().bbs_mut().remove(bb);
        }

        for bb in useless_bbs.iter() {
            data.dfg_mut().remove_bb(*bb);
        }

        sanity_check(data);
    }
}

/// Perform naive def-use analysis over values, and
/// remove all unused values in a function.
/// This pass is meant to only get rid of unnecessary 
/// instructions generated by the rigid AST-to-IR process.
pub struct ElimUnusedValue;

impl FunctionPass for ElimUnusedValue {
    fn run_on(&mut self, _: Function, data: &mut FunctionData) {

        let entry_bb = data.layout().entry_bb();
        if entry_bb.is_none() {
            return;
        }

        // Start from all unreferenced values and do a topological sort.
        // Unreferenced values are unused, except for:
        // - Control-flow instructions
        // - Function calls
        // - Stores. The effectiveness of stores is out of scope for this
        // pass. I.e., storing to the same location consecutively is not
        // considered unused in this pass.

        let mut unused: VecDeque<Value> = VecDeque::new();
        unused.extend(data.dfg().values()
            .iter()
            .filter(|(_, vdata)| {
                vdata.used_by().is_empty()
                && match vdata.kind() {
                    ValueKind::Branch(_) | ValueKind::Call(_) |
                    ValueKind::Jump(_) | ValueKind::Return(_) |
                    ValueKind::Store(_) => false,
                    _ => true,
                }
            })
            .map(|(v,_)| *v)
        );

        while let Some(u) = unused.pop_front() {

            if !data.dfg().value(u).used_by().is_empty() {
                continue
            }
            if let Some(bb) = data.layout().parent_bb(u) {
                
                _ = data.layout_mut().bb_mut(bb).insts_mut().remove(&u);
            }
            let udata = data.dfg().value(u);
            match udata.kind() {
                ValueKind::Aggregate(_) | ValueKind::Alloc(_) |
                ValueKind::FuncArgRef(_) | ValueKind::GlobalAlloc(_) |
                ValueKind::Integer(_) | ValueKind::Undef(_) |
                ValueKind::ZeroInit(_) => {/* Used no more values. */},
                ValueKind::BlockArgRef(_bar) => {
                    let _bb = data.layout().parent_bb(u)
                        .expect("[OPT] BlockArgRef must be in a BB");
                    // TODO: for predecessor blocks, the corresponding parameter
                    // in the jump/branch instruction should be marked as unused. 
                    continue
                },
                ValueKind::Call(_) => {
                    // A function call is never unused.
                    continue
                },
                ValueKind::Branch(_) | ValueKind::Jump(_) |
                ValueKind::Return(_) => unreachable!(),
                ValueKind::Binary(b) => {
                    unused.push_back(b.lhs());
                    unused.push_back(b.lhs());
                },
                ValueKind::GetElemPtr(gep) => {
                    unused.push_back(gep.src());
                    unused.push_back(gep.index());
                },
                ValueKind::GetPtr(gp) => {
                    unused.push_back(gp.src());
                    unused.push_back(gp.index());
                },
                ValueKind::Load(ld) => {
                    unused.push_back(ld.src());
                },
                ValueKind::Store(st) => {
                    unused.push_back(st.dest());
                    unused.push_back(st.value());
                }
            }
            data.dfg_mut().remove_value(u);
        }

        sanity_check(data);

    }
}
