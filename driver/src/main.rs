extern crate regex;

#[macro_use]
extern crate log;
extern crate env_logger;

mod timings;

use std::time::Instant;

fn main() {
    // Set up the logger
    env_logger::init().unwrap();

    // Set the start time. This will be used to timestamp all measurements.
    let baseline = Instant::now();

    println!("{:?}", timings::benchmark(baseline, "alexnet".to_string(), 512, 1).unwrap());
}
