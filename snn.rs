use std::num::Float;
use std::collections::binary_heap::BinaryHeap; // pq
use std::cmp::Ordering;

type float = f64;
type time = u64;

const TIME_RESOLUTION: time = 1000_000; // in micro seconds.

#[derive(Debug)]
struct Config {
    arp:       time,
    tau_m:     float,
    tau_r:     float,
    weight_r:  float,
    threshold: float, 
}

#[derive(Debug)]
struct Neuron {
    /// End of absolute refractory period 
    arp_end:         time,
    last_spike_time: time,
    mem_pot:         float,

    /// An index into a [Config] table.
    config_id:       usize,
}

#[derive(Debug)]
enum NeuronResult {
    InArp,
    NoFire,
    Fire
}

impl Neuron {
    fn new(config_id: usize) -> Neuron {
        Neuron {
            arp_end: 0,
            last_spike_time: 0,
            mem_pot: 0.0,
            config_id: config_id,
        }
    }

    fn spike(&mut self, timestamp: time, weight: float, cfg: &Config) -> NeuronResult {
        assert!(timestamp >= self.last_spike_time);

        // Return early if still in absolute refractory period (arp)
        if timestamp < self.arp_end {
            return NeuronResult::InArp;
        }
         
        // Time since end of last absolute refractory period (arp)
        assert!(timestamp >= self.arp_end);
        let delta = timestamp - self.arp_end;

        // Convert delta to float
        let delta: float = (delta as float) / TIME_RESOLUTION as float;

        // Calculate dynamic threshold
        let dyn_threshold = cfg.threshold +
                            cfg.weight_r * (-delta / cfg.tau_r).exp(); 

        // Update memory potential
        if cfg.tau_m > 0.0 {
          
            let d = timestamp - self.last_spike_time;
            // Convert delta to float
            let d: float = (d as float) / TIME_RESOLUTION as float;

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

#[derive(Debug)]
struct Event {
    time:   time,
    weight: float,
    target: usize
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time
    }
}

impl Eq for Event {}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.time.partial_cmp(&other.time).map(|o| o.reverse())
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        self.time.cmp(&other.time).reverse()
    }
}


fn ms(n: time) -> time { n * 1000 }
fn us(n: time) -> time { n }


struct Net {
    neurons: Vec<Neuron>,
    neuron_configs: Vec<Config>,
    events: BinaryHeap<Event>,
}

impl Net {
    fn new() -> Net {
        Net {
            neurons: vec![],
            neuron_configs: vec![],
            events: BinaryHeap::new(),
        }
    }

    /// Returns the config_id
    fn add_neuron_config(&mut self, config: Config) -> usize {
        self.neuron_configs.push(config);
        self.neuron_configs.len() - 1
    }

    /// Returns the neuron_id
    fn add_neuron(&mut self, config_id: usize) -> usize {
        self.neurons.push(Neuron::new(config_id));
        self.neurons.len() - 1
    }

    fn add_event(&mut self, event: Event) {
        self.events.push(event);
    }

    fn simulate(&mut self) {
	loop {
	    println!("-------------------------------------");

	    if let Some(ev) = self.events.pop() {
		println!("{:?}", ev);
		let neuron = &mut (self.neurons.as_mut_slice())[ev.target];
		let cfg = &(self.neuron_configs.as_slice())[neuron.config_id];
		let fire = neuron.spike(ev.time, ev.weight, cfg);
		println!("{:?}", fire);
		println!("{:?}", neuron);
	    }
	    else {
		break;
	    }
	}
    }
}

fn main() {

    let mut net = Net::new();
   
    // Koinzidenz neurons
    let cfg_k = net.add_neuron_config(Config {
        arp: us(500), // 0.5 ms = 500 us
        tau_m: 0.04,
        tau_r: 0.5,
        weight_r: 0.0,
        threshold: 1.1,
    });

    // Input neurons
    let cfg_i = net.add_neuron_config(Config {
        arp: ms(1),
        tau_m: 0.0,
        tau_r: 0.5,
        weight_r: 0.0,
        threshold: 0.6,
    });

    let n1 = net.add_neuron(cfg_i);

    // Fill event queue with events
    for i in 1..100 {
        let ev = Event {
            time:   ms(i) / 10,   
            weight: 0.6,
            target: n1 
        };

        net.add_event(ev);
        //pq.push(ev);
    }

    net.simulate();
}
