use std::num::Float;

type float = f64;

#[derive(Debug)]
struct Config {
    arp:       float,
    tau_m:     float,
    tau_r:     float,
    weight_r:  float,
    threshold: float, 
}

#[derive(Debug)]
struct Neuron {
    /// End of absolute refractory period 
    arp_end:         float,
    last_spike_time: float,
    mem_pot:         float,
}

#[derive(Debug)]
enum NeuronResult {
    Arp,
    NoFire,
    Fire
}

impl Neuron {
    fn new() -> Neuron {
        Neuron {
            arp_end: 0.0,
            last_spike_time: 0.0,
            mem_pot: 0.0
        }
    }

    fn spike(&mut self, timestamp: float, weight: float, cfg: &Config) -> NeuronResult {
        assert!(timestamp >= 0.0);
        debug_assert!(self.last_spike_time >= 0.0);

        // Time since end of last absolute refractory period (arp)
        let delta = timestamp - self.arp_end;

        // Return early if still in arp
        if delta < 0.0 {
            return NeuronResult::Arp;
        }

        // Calculate dynamic threshold
        let dyn_threshold = cfg.threshold +
                            cfg.weight_r * (-delta / cfg.tau_r).exp(); 

        // Update memory potential
        if cfg.tau_m > 0.0 {
            let d = timestamp - self.last_spike_time;
            let decay = (-d / cfg.tau_m).exp();
            self.mem_pot *= decay;
            self.mem_pot += weight;
        }
        else {
            self.mem_pot = weight;
        }

        // Update last spike time
        self.last_spike_time = timestamp;

        if self.mem_pot >= dyn_threshold {
            self.arp_end = timestamp + cfg.arp;
            NeuronResult::Fire
        }
        else {
            NeuronResult::NoFire
        }
    }
}

fn main() {
    // Koinzidenz neurons
    let cfg_k = Config {
        arp: 0.5,
        tau_m: 0.04,
        tau_r: 0.5,
        weight_r: 0.0,
        threshold: 1.1,
    };

    // Input neurons
    let cfg_i = Config {
        arp: 1.0,
        tau_m: 0.0,
        tau_r: 0.5,
        weight_r: 0.0,
        threshold: 0.6,
    };

    let mut n1 = Neuron::new();

    for i in 1..100 {
        let t = i as float / 10.0;
        println!("-------------------------------------");
        println!("t: {}", t);

        let fire = n1.spike(t, 0.5, &cfg_i);
        println!("{:?}", fire);
        println!("{:?}", n1);
    }
}
