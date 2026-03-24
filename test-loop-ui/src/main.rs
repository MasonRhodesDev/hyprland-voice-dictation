use std::process;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let recording = args.iter().position(|a| a == "--recording").and_then(|i| args.get(i + 1)).map(|s| s.as_str());

    if let Err(e) = test_loop_ui::run(recording) {
        eprintln!("test-loop-ui error: {:#}", e);
        process::exit(1);
    }
}
