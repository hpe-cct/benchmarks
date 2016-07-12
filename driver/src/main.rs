/*
 * (c) Copyright 2016 Hewlett Packard Enterprise Development LP
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

extern crate regex;
extern crate rand;
extern crate csv;
extern crate rustc_serialize;

#[macro_use]
extern crate log;
extern crate env_logger;

mod timings;
use timings::BenchmarkEvent;

use rand::Rng;
use std::time::{Instant, Duration};

#[derive(Clone, Debug)]
struct RunConfig {
    container: String,
    net: String,
    batch_size: u32,
    gpu: u32
}

#[derive(Debug)]
struct RunResult {
    config: RunConfig,
    result: Result<Vec<timings::BenchmarkEvent>, String>
}

#[derive(Debug, RustcEncodable)]
struct RunErrorRecord {
    container: String,
    net: String,
    batch_size: u32,
    gpu: u32,
    error: String
}

#[derive(Debug, RustcEncodable)]
struct RunAverageRecord {
    container: String,
    net: String,
    batch_size: u32,
    gpu: u32,
    forward: f64,
    backward: f64,
    total: f64
}

/*#[derive(Debug, RustcEncodable)]
struct PerLayerRecord {
    net: String,
    batch_size: u32,
    gpu: u32,
    direction: String, // "forward" or "backward"
    time: f64
}

#[derive(Debug, RustcEncodable)]
struct EventRecord {
    net: String,
    batch_size: u32,
    gpu: u32,
    timestamp: f64,
    iteration: u32,
    time: f64
}*/

fn duration_to_sec(d: Duration) -> f64 {
    let s = d.as_secs() as f64;
    let n = d.subsec_nanos() as f64 / 1e9f64;
    return s + n
}

fn main() {
    // Set up the logger
    env_logger::init().unwrap();

    let containers = vec!["benchmark-caffe", "benchmark-cct"];
    let gpus = vec![0, 1];
    let batch_sizes = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8096, 16384, 32768, 65536];
    let nets = vec!["alexnet", "cifar10_quick"];

    //let gpus = vec![1];
    //let batch_sizes = vec![1, 4096];
    //let nets = vec!["alexnet"];

    let mut configurations: Vec<RunConfig> = vec![];

    for c in &containers {
        for g in &gpus {
            for n in &nets {
                for b in &batch_sizes {
                    let config = RunConfig {
                        container: c.to_string(),
                        net: n.to_string(),
                        batch_size: *b,
                        gpu: *g
                    };

                    configurations.push(config);
                }
            }
        }
    }

    // Evaluate the configurations in a random order
    rand::thread_rng().shuffle(&mut configurations);

    // Set the start time. This will be used to timestamp all measurements.
    let baseline = Instant::now();

    // Evaluate all the points in the configuration space
    let mut results: Vec<RunResult> = vec![];
    for c in configurations {
        let result = RunResult {
            config: c.clone(),
            result: timings::benchmark(baseline, c.container, c.net, c.batch_size, c.gpu)
        };
        results.push(result);
    }

    // Write the first output table: failed points in the configuration space
    {
        let mut writer = csv::Writer::from_file("failed.csv").unwrap();
        writer.write(vec!["container", "net", "batch_size", "gpu", "error"].into_iter()).unwrap();
        for r in &results {
            match &r.result {
                &Err(ref s) => {
                    let line = RunErrorRecord {
                        container: r.config.container.clone(),
                        net: r.config.net.clone(),
                        batch_size: r.config.batch_size,
                        gpu: r.config.gpu,
                        error: s.clone()
                    };

                    writer.encode(line).unwrap();
                },

                &Ok(_) => ()
            }
        }
    }

    // Write the second output table: average performance at valid points in the configuration space
    {
        let mut writer = csv::Writer::from_file("valid.csv").unwrap();
        writer.write(vec!["container", "net", "batch_size", "gpu", "forward", "backward", "total"].into_iter()).unwrap();
        for r in &results {
            match &r.result {
                &Err(_) => (),

                &Ok(ref d) => {
                    let mut forward: Option<f64> = None;
                    let mut backward: Option<f64> = None;
                    let mut total: Option<f64> = None;

                    for event in d {
                        match event {
                            &BenchmarkEvent::AvgForward {timestamp: _, time: d} => forward = Some(duration_to_sec(d)),
                            &BenchmarkEvent::AvgBackward {timestamp: _, time: d} => backward = Some(duration_to_sec(d)),
                            &BenchmarkEvent::AvgForwardBackward {timestamp: _, time: d} => total = Some(duration_to_sec(d)),
                            _ => ()
                        }
                    }

                    let line = RunAverageRecord {
                        container: r.config.container.clone(),
                        net: r.config.net.clone(),
                        batch_size: r.config.batch_size,
                        gpu: r.config.gpu,
                        forward: forward.unwrap(),
                        backward: backward.unwrap(),
                        total: total.unwrap()
                    };

                    writer.encode(line).unwrap();
                }
            }
        }
    }
}
