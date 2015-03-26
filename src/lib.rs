#![feature(core)]

use std::num::Float;
use std::collections::binary_heap::BinaryHeap; // pq
use std::cmp::Ordering;

pub type float = f64;
pub type time = u64;

const TIME_RESOLUTION: time = 1000_000; // in micro seconds.

#[derive(Debug, Copy)]
pub struct NeuronConfig {
    pub arp:       time,
    pub tau_m:     float,
    pub tau_r:     float,
    pub weight_r:  float,
    pub threshold: float,
    pub record:    Option<&'static str>,
}

#[derive(Debug, Copy)]
pub struct NeuronConfigId(usize);

#[derive(Debug, Copy)]
pub struct Neuron {
    /// End of absolute refractory period 
    arp_end:         time,
    last_spike_time: time,
    mem_pot:         float,

    /// An index into a [NeuronConfig] table.
    config_id:       NeuronConfigId,
}

#[derive(Debug, Copy)]
pub struct NeuronId(usize);

#[derive(Debug, Copy)]
pub enum NeuronResult {
    InArp,
    NoFire,
    Fire
}

impl Neuron {
    pub fn new(config_id: NeuronConfigId) -> Neuron {
        Neuron {
            arp_end: 0,
            last_spike_time: 0,
            mem_pot: 0.0,
            config_id: config_id,
        }
    }

    pub fn spike(&mut self, timestamp: time, weight: float, cfg: &NeuronConfig) -> NeuronResult {
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
pub struct Event {
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


pub fn ms(n: time) -> time { n * 1000 }
pub fn us(n: time) -> time { n }

#[derive(Debug)]
pub struct Synapse {
    pub delay:  time,
    pub weight: float,
    pub post_neuron: NeuronId,
}

pub struct Net {
    neurons: Vec<Neuron>,
    neuron_configs: Vec<NeuronConfig>,
    synapses: Vec<Vec<Synapse>>,
    events: BinaryHeap<Event>,
}

impl Net {
    pub fn new() -> Net {
        Net {
            neurons: vec![],
            neuron_configs: vec![],
            synapses: vec![],
            events: BinaryHeap::new(),
        }
    }

    pub fn create_neuron_config(&mut self, config: NeuronConfig) -> NeuronConfigId {
        self.neuron_configs.push(config);
        NeuronConfigId(self.neuron_configs.len() - 1)
    }

    pub fn create_neuron(&mut self, config_id: NeuronConfigId) -> NeuronId {
        self.neurons.push(Neuron::new(config_id));
        self.synapses.push(vec![]);
        NeuronId(self.neurons.len() - 1)
    }

    pub fn create_synapse(&mut self, from_neuron: NeuronId, synapse: Synapse) {
        let mut syn_arr = &mut (self.synapses.as_mut_slice())[from_neuron.0];
        syn_arr.push(synapse);
    }

    pub fn add_event(&mut self, event: Event) {
        self.events.push(event);
    }

    // values are in ms
    pub fn add_spike_train_float_ms(&mut self, target: NeuronId, weight: float, spikes: &[float]) {
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

    pub fn simulate(&mut self) {
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
