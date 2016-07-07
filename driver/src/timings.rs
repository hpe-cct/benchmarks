use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Instant, Duration};
use regex::Regex;

#[derive(Debug)]
pub enum BenchmarkEvent {
    IterForwardBackward {timestamp: Duration, iteration: u32, time: Duration},
    LayerForward {timestamp: Duration, layer: String, time: Duration},
    LayerBackward {timestamp: Duration, layer: String, time: Duration},
    AvgForward {timestamp: Duration, time: Duration},
    AvgBackward {timestamp: Duration, time: Duration},
    AvgForwardBackward {timestamp: Duration, time: Duration}
}

fn ms_to_duration(ms: f64) -> Duration {
    assert!(ms > 0f64);

    let seconds = (ms / 1000f64).trunc();
    let nanos = (ms / 1000f64).fract() * 1e9f64;

    Duration::new(seconds as u64, nanos as u32)
}

pub fn benchmark(baseline: Instant, net: String, batch_size: u32, gpu: u32) -> Result<Vec<BenchmarkEvent>, String> {
    info!("launching Caffe process (net: {}, batch_size: {}, gpu: {})", net, batch_size, gpu);

    let mut child = match Command::new("nvidia-docker")
        .arg("run")
        .arg("--rm")
        .arg("-i")
        .arg("benchmark")
        .arg(net)
        .arg(batch_size.to_string())
        .env("NV_GPU", gpu.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .spawn() {
            Ok(c) => c,
            Err(e) => return Err(format!("failed to execute process nvidia-docker: {}", e))
        };

    // "Iteration:" <int> "forward-backward time:" <f32> "ms." -> IterForwardBackward(iteration, time)
    // <string> "forward:" <f32> "ms." -> LayerForward(layer, time)
    // <string> "backward:" <f32> "ms." -> LayerBackward(layer, time)
    // "Average Forward pass:" <f32> "ms." -> AvgForward(time)
    // "Average Backward pass:" <f32> "ms." -> AvgBackward(time)
    // "Average Forward-Backward:" <f32> "ms." -> AvgForwardBackward(time)
    let iter_forward_backward = Regex::new(r"Iteration: (?P<iter>\d+) forward-backward time: (?P<time>\d*[.]\d*) ms[.]").unwrap();
    let layer_forward = Regex::new(r"(?P<layer>[:alnum:]+)\s+forward: (?P<time>\d*[.]\d*) ms[.]").unwrap();
    let layer_backward = Regex::new(r"(?P<layer>[:alnum:]+)\s+backward: (?P<time>\d*[.]\d*) ms[.]").unwrap();
    let avg_forward = Regex::new(r"Average Forward pass: (?P<time>\d*[.]\d*) ms[.]").unwrap();
    let avg_backward = Regex::new(r"Average Backward pass: (?P<time>\d*[.]\d*) ms[.]").unwrap();
    let avg_forward_backward = Regex::new(r"Average Forward-Backward: (?P<time>\d*[.]\d*) ms[.]").unwrap();

    info!("process launched, parsing output");

    let mut events: Vec<BenchmarkEvent> = vec!();

    let stderr = child.stderr.take().unwrap();
    let mut reader = BufReader::new(stderr);
    let mut buffer = String::new();

    while reader.read_line(&mut buffer).unwrap() > 0 {
        iter_forward_backward.captures(&buffer).map(|i| {
            info!("{:?}", i);
            let it = i.name("iter").unwrap().parse().unwrap();
            let v = ms_to_duration(i.name("time").unwrap().parse().unwrap());
            let e = BenchmarkEvent::IterForwardBackward {timestamp: baseline.elapsed(), iteration: it, time: v};
            events.push(e)
        });

        layer_forward.captures(&buffer).map(|i| {
            info!("{:?}", i);
            let l = i.name("layer").unwrap().to_string();
            let v = ms_to_duration(i.name("time").unwrap().parse().unwrap());
            let e = BenchmarkEvent::LayerForward {timestamp: baseline.elapsed(), layer: l, time: v};
            events.push(e)
        });

        layer_backward.captures(&buffer).map(|i| {
            info!("{:?}", i);
            let l = i.name("layer").unwrap().to_string();
            let v = ms_to_duration(i.name("time").unwrap().parse().unwrap());
            let e = BenchmarkEvent::LayerBackward {timestamp: baseline.elapsed(), layer: l, time: v};
            events.push(e)
        });

        avg_forward.captures(&buffer).map(|i| {
            info!("{:?}", i);
            let v = ms_to_duration(i.name("time").unwrap().parse().unwrap());
            let e = BenchmarkEvent::AvgForward {timestamp: baseline.elapsed(), time: v};
            events.push(e)
        });

        avg_backward.captures(&buffer).map(|i| {
            info!("{:?}", i);
            let v = ms_to_duration(i.name("time").unwrap().parse().unwrap());
            let e = BenchmarkEvent::AvgBackward {timestamp: baseline.elapsed(), time: v};
            events.push(e)
        });

        avg_forward_backward.captures(&buffer).map(|i| {
            info!("{:?}", i);
            let v = ms_to_duration(i.name("time").unwrap().parse().unwrap());
            let e = BenchmarkEvent::AvgForwardBackward {timestamp: baseline.elapsed(), time: v};
            events.push(e)
        });

        buffer.clear();
    }

    let ecode = match child.wait() {
        Ok(e) => e,
        Err(e) => return Err(format!("failed to wait on child: {}", e))
    };

    match ecode.code() {
        None => return Err("child process unexpectedly terminated".to_string()),
        Some(c) => if c != 0 {
            return Err(format!("child process terminated with non-zero exit code {}", c))
        }
    };

    info!("finished");

    return Ok(events);
}
