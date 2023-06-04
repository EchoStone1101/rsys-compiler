//! Implements IR generation.
use koopa::ir::*;
use koopa::back::KoopaGenerator;

pub trait IRGen {
    fn gen(&self, stream: &mut String);
}

impl IRGen for Program {
    fn gen(&self, out: &mut String) {
        // TODO: global variables

        // Functions
        for (_, data) in self.funcs().iter() {
            gen_function(data, out);
        }
    }
}

fn gen_function(func: &FunctionData, out: &mut String) {
    let signature = format!("{}({}): {}", 
        func.name(), 
        func.params()
            .iter()
            .map(|v| {
                let vdata = func.dfg().value(*v);
                // TODO: types
                format!("{}: {}", vdata.name().clone().unwrap(), "i32")
            })
            .collect::<Vec<String>>()
            .join(", "),
        // TODO: types
        // func.ty()
        "i32",
    );
    let mut body = String::new();
    gen_function_body(&mut body);
    out.push_str(&format!("fun {} {{\n{}}}\n", signature, body));
}

fn gen_function_body(_out: &mut String) {

}