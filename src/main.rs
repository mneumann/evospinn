#![feature(core)]

use std::num::Float;
use std::collections::binary_heap::BinaryHeap; // pq
use std::cmp::Ordering;

type float = f64;
type time = u64;

const TIME_RESOLUTION: time = 1000_000; // in micro seconds.

#[derive(Debug, Copy)]
struct NeuronConfig {
    arp:       time,
    tau_m:     float,
    tau_r:     float,
    weight_r:  float,
    threshold: float,
    record:    Option<&'static str>,
}

#[derive(Debug, Copy)]
struct NeuronConfigId(usize);

#[derive(Debug, Copy)]
struct Neuron {
    /// End of absolute refractory period 
    arp_end:         time,
    last_spike_time: time,
    mem_pot:         float,

    /// An index into a [NeuronConfig] table.
    config_id:       NeuronConfigId,
}

#[derive(Debug, Copy)]
struct NeuronId(usize);

#[derive(Debug, Copy)]
enum NeuronResult {
    InArp,
    NoFire,
    Fire
}

impl Neuron {
    fn new(config_id: NeuronConfigId) -> Neuron {
        Neuron {
            arp_end: 0,
            last_spike_time: 0,
            mem_pot: 0.0,
            config_id: config_id,
        }
    }

    fn spike(&mut self, timestamp: time, weight: float, cfg: &NeuronConfig) -> NeuronResult {
        assert!(timestamp >= self.last_spike_time);
        assert!(cfg.tau_m >= 0.0);

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
    target: NeuronId
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

#[derive(Debug)]
struct Synapse {
    delay:  time,
    weight: float,
    post_neuron: NeuronId,
}

struct Net {
    neurons: Vec<Neuron>,
    neuron_configs: Vec<NeuronConfig>,
    synapses: Vec<Vec<Synapse>>,
    events: BinaryHeap<Event>,
}

impl Net {
    fn new() -> Net {
        Net {
            neurons: vec![],
            neuron_configs: vec![],
            synapses: vec![],
            events: BinaryHeap::new(),
        }
    }

    fn create_neuron_config(&mut self, config: NeuronConfig) -> NeuronConfigId {
        self.neuron_configs.push(config);
        NeuronConfigId(self.neuron_configs.len() - 1)
    }

    fn create_neuron(&mut self, config_id: NeuronConfigId) -> NeuronId {
        self.neurons.push(Neuron::new(config_id));
        self.synapses.push(vec![]);
        NeuronId(self.neurons.len() - 1)
    }

    fn create_synapse(&mut self, from_neuron: NeuronId, synapse: Synapse) {
        let mut syn_arr = &mut (self.synapses.as_mut_slice())[from_neuron.0];
        syn_arr.push(synapse);
    }

    fn add_event(&mut self, event: Event) {
        self.events.push(event);
    }

    // values are in ms
    fn add_spike_train_float_ms(&mut self, target: NeuronId, weight: float, spikes: &[float]) {
        for &t in spikes {
            self.add_event(Event {
                time: (t * 1000.0) as time, // convert to us
                weight: weight,
                target: target
            });
        }
    }

    fn fire(&mut self, timestamp: time, neuron_id: NeuronId) {
        for syn in &self.synapses[neuron_id.0] {
            println!("{:?}", syn);
            self.events.push(Event {
                time: timestamp + syn.delay,
                weight: syn.weight,
                target: syn.post_neuron
            });
        }

        if let Some(ident) = self.get_config_for_neuron_id(neuron_id).record {
            println!("RECORD\t{}\t{}\t{}", ident, neuron_id.0, timestamp);
        }
    }

    fn get_neuron(&self, neuron_id: NeuronId) -> &Neuron {
        &self.neurons[neuron_id.0]
    }

    fn get_config_for_neuron<'a>(&'a self, neuron: &Neuron) -> &'a NeuronConfig {
        &(self.neuron_configs.as_slice())[neuron.config_id.0]
    }

    fn get_config_for_neuron_id(&self, neuron_id: NeuronId) -> &NeuronConfig {
        self.get_config_for_neuron(self.get_neuron(neuron_id))
    }

    fn simulate(&mut self) {
	loop {
	    println!("-------------------------------------");

	    if let Some(ev) = self.events.pop() {
		println!("{:?}", ev);
                let neuron_id = ev.target;
                let fire = {
		    let neuron = &mut (self.neurons.as_mut_slice())[neuron_id.0];
                    let cfg = &(self.neuron_configs.as_slice())[neuron.config_id.0];
		    let fire = neuron.spike(ev.time, ev.weight, cfg);
		    println!("{:?}", fire);
		    println!("{:?}", neuron);
                    fire
                };
                match fire {
                    NeuronResult::Fire => {
                        // Post a new event to all synapses
                        self.fire(ev.time, neuron_id);
                    }
                    _ => {
                    }
                }
	    }
	    else {
		break;
	    }
	}
    }
}

// From Optimierung-1/stimuli.txt

const spikes_input_0: [float; 49] = [
    0.38, 3.38, 6.38, 9.38, 12.38, 15.38, 18.38,
    21.38, 24.38, 27.38, 30.38, 33.38, 36.38, 39.38,
    42.38, 45.38, 48.38, 52.98, 55.98, 58.98,
    61.98, 64.98, 67.98, 70.98, 73.98, 76.98, 79.98,
    82.98, 85.98, 88.98, 91.98, 94.98, 97.98,
    103.59, 106.59, 109.59, 112.59, 115.59, 118.59,
    121.59, 124.59, 127.59, 130.59, 133.59, 136.59, 139.59,
    142.59, 145.59, 148.59
];

const spikes_input_1: [float; 49] = [
    1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0,
    22.0, 25.0, 28.0, 31.0, 34.0, 37.0,
    40.0, 43.0, 46.0, 49.0, 53.0, 56.0, 59.0,
    62.0, 65.0, 68.0, 71.0, 74.0, 77.0,
    80.0, 83.0, 86.0, 89.0, 92.0, 95.0, 98.0,
    103.0, 106.0, 109.0, 112.0, 115.0, 118.0,
    121.0, 124.0, 127.0, 130.0, 133.0, 136.0, 139.0,
    142.0, 145.0, 148.0
];

const correct_output_0: (float, float) = (  0.0,  47.0);
const correct_output_1: (float, float) = ( 47.0, 100.0);
const correct_output_2: (float, float) = (100.0, 170.0);

fn main() {
    let mut net = Net::new();

    let input_neuron_template = NeuronConfig {
        arp: ms(0),
        tau_m: 0.0,
        tau_r: 0.0,
        weight_r: 0.0,
        threshold: 0.0,
        record: Some("input"),
    };
    let output_neuron_template = NeuronConfig {
        arp: ms(0),
        tau_m: 0.0,
        tau_r: 0.0,
        weight_r: 0.0,
        threshold: 0.609375,
        record: Some("output"),
    };

    let cfg_input0 = net.create_neuron_config(NeuronConfig {
        record: Some("input0"),
        ..input_neuron_template
    });
    let cfg_input1 = net.create_neuron_config(NeuronConfig {
        record: Some("input1"),
        ..input_neuron_template
    });
    let cfg_output0 = net.create_neuron_config(NeuronConfig {
        record: Some("output0"),
        ..output_neuron_template
    });
    let cfg_output1 = net.create_neuron_config(NeuronConfig {
        record: Some("output1"),
        ..output_neuron_template
    });
    let cfg_output2 = net.create_neuron_config(NeuronConfig {
        record: Some("output2"),
        ..output_neuron_template
    });
    let cfg_innerinp = net.create_neuron_config(NeuronConfig {
        arp: ms(1),
        tau_m: 0.0,
        tau_r: 0.0,
        weight_r: 0.0,
        threshold: 0.59375,
        record: None,
    });
    // Koinzidenz neurons
    let cfg_k = net.create_neuron_config(NeuronConfig {
        arp: us(500), // 0.5 ms = 500 us
        tau_m: 0.046875,
        tau_r: 0.0,
        weight_r: 0.0,
        threshold: 1.09375,
        record: None,
    });

    let n_input0 = net.create_neuron(cfg_input0);
    let n_input1 = net.create_neuron(cfg_input1);
    let n_output0 = net.create_neuron(cfg_output0);
    let n_output1 = net.create_neuron(cfg_output1);
    let n_output2 = net.create_neuron(cfg_output2);
    let n_innerinp0 = net.create_neuron(cfg_innerinp);
    let n_innerinp1 = net.create_neuron(cfg_innerinp);
    let n_k0 = net.create_neuron(cfg_k);
    let n_k1 = net.create_neuron(cfg_k);
    let n_k2 = net.create_neuron(cfg_k);

    net.create_synapse(n_input0, Synapse {delay: us(0), weight: 1.0, post_neuron: n_innerinp0});
    net.create_synapse(n_input1, Synapse {delay: us(0), weight: 1.0, post_neuron: n_innerinp1});
    net.create_synapse(n_k0, Synapse {delay: us(0), weight: 1.0, post_neuron: n_output0});
    net.create_synapse(n_k1, Synapse {delay: us(0), weight: 1.0, post_neuron: n_output1});
    net.create_synapse(n_k2, Synapse {delay: us(0), weight: 1.0, post_neuron: n_output2});

    net.create_synapse(n_innerinp0, Synapse {delay: us(625), weight: 1.0, post_neuron: n_k0});
    net.create_synapse(n_innerinp0, Synapse {delay: us(15), weight: 1.0, post_neuron: n_k1}); // 0.015625
    net.create_synapse(n_innerinp0, Synapse {delay: us(0), weight: 1.0, post_neuron: n_k2});

    net.create_synapse(n_innerinp1, Synapse {delay: us(0), weight: 1.0, post_neuron: n_k0});
    //net.create_synapse(n_innerinp1, Synapse {delay: us(0), weight: 1.0, post_neuron: n_k1});
    net.create_synapse(n_innerinp1, Synapse {delay: us(15), weight: 1.0, post_neuron: n_k1});
    net.create_synapse(n_innerinp1, Synapse {delay: us(593), weight: 1.0, post_neuron: n_k2});

    net.add_spike_train_float_ms(n_input0, 1.0, &spikes_input_0);
    net.add_spike_train_float_ms(n_input1, 1.0, &spikes_input_1);

    net.simulate();
}
