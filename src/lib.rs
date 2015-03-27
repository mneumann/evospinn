#![feature(core)]

use std::num::Float;
use std::collections::binary_heap::BinaryHeap; // pq
use std::cmp::Ordering;

pub type float = f64;
pub type time = u64; // in nano seconds

pub mod timestamp;

fn time_to_ms_float(t: time) -> float {
    (t as float) / 1_000_000.0 as float
}

fn ms_float_to_time(t: float) -> time {
    (t * 1_000_000.0) as time
}

pub fn ms(n: time) -> time { n * 1_000_000 }
pub fn us(n: time) -> time { n * 1000 }
pub fn ns(n: time) -> time { n }

#[derive(Debug, Copy)]
pub struct NeuronConfig {
    pub arp:       time,
    pub tau_m:     time,
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

#[derive(Debug, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NeuronId(usize);

#[derive(Debug, Copy, PartialEq, Eq)]
pub enum NeuronResult {
    InArp,
    NoFire,
    Fire
}

fn calc_decay(duration: time, tau_m: time) -> float {
    (duration as f64 / tau_m as f64).exp()
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

        // Return early if still in absolute refractory period (arp)
        if timestamp < self.arp_end {
            return NeuronResult::InArp;
        }
         
        // Time since end of last absolute refractory period (arp)
        assert!(timestamp >= self.arp_end);
        let delta = timestamp - self.arp_end;

        // Convert delta to float (ms)
        let delta = time_to_ms_float(delta);

        // Calculate dynamic threshold
        let dyn_threshold = cfg.threshold +
                            cfg.weight_r * (-delta / cfg.tau_r).exp(); 

        // Update memory potential
        if cfg.tau_m > 0 {
            assert!(timestamp >= self.last_spike_time);
            let decay = calc_decay(timestamp - self.last_spike_time, cfg.tau_m);
            self.mem_pot = weight + (self.mem_pot / decay);
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

macro_rules! assert_in_delta {
    ( $exp:expr, $act:expr, $delta:expr ) => {
        assert!( ($exp - $act).abs() < $delta )
    };
}

#[test]
fn test_decay()
{
    let decay = calc_decay(1, 1);
    assert_in_delta!(2.7182, decay, 0.01);

    let decay = calc_decay(100000000000, 144269504089);
    assert_in_delta!(2.0, decay, 0.01);
}

#[test]
fn test_neuron_firing()
{
    let mut neuron = Neuron::new(NeuronConfigId(0));
    let neuron_cfg = NeuronConfig {
        arp:       0,
        tau_m:     us(0),
        tau_r:     0.0,
        weight_r:  0.0,
        threshold: 1.0,
        record:    None,
    };

    assert_eq!(0, neuron.arp_end);
    assert_eq!(0, neuron.last_spike_time);
    assert_in_delta!(0.0, neuron.mem_pot, 0.0001);

    let res = neuron.spike(ms(1), 0.9, &neuron_cfg); 
    assert_eq!(NeuronResult::NoFire, res);
}

#[derive(Debug)]
pub struct Event {
    time:   time,
    weight: float,
    target: NeuronId
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.target == other.target
    }
}

impl Eq for Event {}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let res = if self.time < other.time {
            Ordering::Less
        }
        else if self.time > other.time {
            Ordering::Greater
        }
        else {
            debug_assert!(self.time == other.time);
            self.target.cmp(&other.target)
        };
        Some(res.reverse())
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

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
                time: ms_float_to_time(t),
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

	    if let Some(mut ev) = self.events.pop() {

                // consume all elements with same timestamp and same target 
                loop {
                    match self.events.peek() {
                        Some(ev2) if ev2.time == ev.time && ev2.target == ev.target => {
                            println!("consume additional event: {:?}", ev2);
                            ev.weight += ev2.weight;
                        }
                        _ => break
                    }
                    let _ = self.events.pop();
                }

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
