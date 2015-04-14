#![feature(collections,core)]
extern crate evospinn;

extern crate rand;

use evospinn::*;
use std::ops::Range;
use std::collections::BitVec;

// From Optimierung-1/stimuli.txt

const SPIKES_INPUT_0: [float; 49] = [
    0.38, 3.38, 6.38, 9.38, 12.38, 15.38, 18.38,
    21.38, 24.38, 27.38, 30.38, 33.38, 36.38, 39.38,
    42.38, 45.38, 48.38, 52.98, 55.98, 58.98,
    61.98, 64.98, 67.98, 70.98, 73.98, 76.98, 79.98,
    82.98, 85.98, 88.98, 91.98, 94.98, 97.98,
    103.59, 106.59, 109.59, 112.59, 115.59, 118.59,
    121.59, 124.59, 127.59, 130.59, 133.59, 136.59, 139.59,
    142.59, 145.59, 148.59
];

const SPIKES_INPUT_1: [float; 49] = [
    1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0,
    22.0, 25.0, 28.0, 31.0, 34.0, 37.0,
    40.0, 43.0, 46.0, 49.0, 53.0, 56.0, 59.0,
    62.0, 65.0, 68.0, 71.0, 74.0, 77.0,
    80.0, 83.0, 86.0, 89.0, 92.0, 95.0, 98.0,
    103.0, 106.0, 109.0, 112.0, 115.0, 118.0,
    121.0, 124.0, 127.0, 130.0, 133.0, 136.0, 139.0,
    142.0, 145.0, 148.0
];

#[derive(Debug)]
struct FitnessRecorder {
    total_fires:    usize,
    correct_fires:  usize,
    correct_ranges: Vec<(NeuronId, Range<time>)>
}

impl FitnessRecorder {
    fn new() -> FitnessRecorder {
        FitnessRecorder {
            total_fires: 0,
            correct_fires: 0,
            correct_ranges: vec!(),
        }
    }

    fn add_correct_range(&mut self, neuron_id: NeuronId, range: Range<time>) {
        self.correct_ranges.push((neuron_id, range));
    }

    fn classification_rate(&self) -> f64 {
        (self.correct_fires as f64) / (self.total_fires as f64)
    }
}

impl Recorder for FitnessRecorder {
    fn record_fire(&mut self, timestamp: time, neuron_id: NeuronId) {
        for &(nid, ref range) in self.correct_ranges.iter() {
            if nid == neuron_id {
                self.total_fires += 1;
                if timestamp >= range.start && timestamp < range.end {
                    self.correct_fires += 1;
                }
            }
        }
    }
}

fn generate_net<R:Recorder>(tau_m_k: time) -> Net<R> {
    let mut net = Net::new();

    let input_neuron_template = NeuronConfig {
        arp: ms(0),
        tau_m: ms(0),
        tau_r: 0.0,
        weight_r: 0.0,
        threshold: 0.0,
        record: Some("input"),
    };
    let output_neuron_template = NeuronConfig {
        arp: ms(0),
        tau_m: ms(0),
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
        tau_m: ms(0),
        tau_r: 0.0,
        weight_r: 0.0,
        threshold: 0.59375,
        record: None,
    });
    // Koinzidenz neurons
    let cfg_k = net.create_neuron_config(NeuronConfig {
        arp: us(500), // 0.5 ms = 500 us
        tau_m: tau_m_k,
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
    net.create_synapse(n_innerinp0, Synapse {delay: ns(15_625), weight: 1.0, post_neuron: n_k1});
    net.create_synapse(n_innerinp0, Synapse {delay: us(0), weight: 1.0, post_neuron: n_k2});

    net.create_synapse(n_innerinp1, Synapse {delay: us(0), weight: 1.0, post_neuron: n_k0});
    net.create_synapse(n_innerinp1, Synapse {delay: us(0), weight: 1.0, post_neuron: n_k1});
    net.create_synapse(n_innerinp1, Synapse {delay: ns(593_750), weight: 1.0, post_neuron: n_k2});

    net.name_neuron(n_output0, "output0");
    net.name_neuron(n_output1, "output1");
    net.name_neuron(n_output2, "output2");
    net.name_neuron(n_input0,  "input0");
    net.name_neuron(n_input1,  "input1");
 
    net
}

/// `nbits` signification number of bits
/*
fn bitvec_from_u64(n: u64, nbits: usize) -> BitVec {
    match nbits {
        1 ... 63 if n < (2u64 << nbits) => {}
        64 => {}
        _  => panic!()
    }

    BitVec::from_fn(nbits, |i| { (n >> i) & 1 == 1 })
}
*/

/// Push the lower `nbits` bits of `value`, msb first.
fn bitvec_push_bits(bv: &mut BitVec, value: usize, nbits: usize) {
    if nbits >= std::usize::BITS { panic!() }
    bv.reserve(nbits);
    for i in (0 .. nbits).rev() {
        bv.push((value >> i) & 1 == 1);
    }
}

trait BooleanLike: Sized {
    fn is_true(self) -> bool;
    fn is_false(self) -> bool { !self.is_true() }
}

impl BooleanLike for bool {
    fn is_true(self) -> bool { self }
}

/// Construct an integer value out of the `nbits` next bits, msb first.
fn bitvec_construct_value<I:Iterator>(iter: &mut I, nbits: usize) -> usize
where I::Item: BooleanLike {
    let mut value = 0usize;

    for _ in (0 .. nbits) {
        value = value << 1;
        if iter.next().unwrap().is_true() {
            value |= 1;
        }
    }

    return value;
}

fn bitvec_random(nbits: usize) -> BitVec {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    BitVec::from_fn(nbits, |_| rng.gen())
}

#[test]
fn test_bitvec_from_u64() {
    let bv = bitvec_from_u64(0b1111, 4);
    assert_eq!(BitVec::from_elem(4, true), bv);
}

trait Genome {
    fn encode_to_bitvec(&self) -> BitVec;
    fn decode_from_bitvec(bv: &BitVec) -> Self;

    /// Higher values are better
    fn fitness(&self) -> f32;
}

/// `flip_prob` is the probablity that we flip a bit.
fn mutate_dna(dna: &BitVec, flip_prob: f32) -> BitVec {
    use rand::{Rng, Open01};

    let mut rng = rand::thread_rng();
    let mut mutant = BitVec::with_capacity(dna.len());

    for bit in dna {
        let Open01(r): Open01<f32> = rng.gen();
        let bit = if r < flip_prob { !bit } else { bit };
        mutant.push(bit);
    }
    mutant
}

#[derive(Debug)]
struct MyGenome {
    tau_m_k: time
}

impl Genome for MyGenome {
    fn encode_to_bitvec(&self) -> BitVec {
        let mut bv = BitVec::new();
        bitvec_push_bits(&mut bv, self.tau_m_k as usize, 20);
        bv
    }

    fn decode_from_bitvec(bv: &BitVec) -> MyGenome {
        let mut it = bv.iter();
        let value = bitvec_construct_value(&mut it, 20); 
        // XXX test that iterator is exhausted
        MyGenome {
            tau_m_k: value as time
        }
    }

    fn fitness(&self) -> f32 {
        //let mut net = generate_net(ns(46_875));
        let mut net = generate_net(ns(self.tau_m_k));

        let mut fitness = Box::new(FitnessRecorder::new());
        fitness.add_correct_range(net.lookup_neuron("output0"), ms(0) .. ms(47));
        fitness.add_correct_range(net.lookup_neuron("output1"), ms(47) .. ms(100));
        fitness.add_correct_range(net.lookup_neuron("output2"), ms(100) .. ms(170));

        let mut fitness = Box::new(FitnessRecorder::new());
        fitness.add_correct_range(net.lookup_neuron("output0"), ms(0) .. ms(47));
        fitness.add_correct_range(net.lookup_neuron("output1"), ms(47) .. ms(100));
        fitness.add_correct_range(net.lookup_neuron("output2"), ms(100) .. ms(170));
        net.set_recorder(Some(fitness));

        let input0 = net.lookup_neuron("input0");
        let input1 = net.lookup_neuron("input1");
        net.add_spike_train_float_ms(input0, 1.0, &SPIKES_INPUT_0);
        net.add_spike_train_float_ms(input1, 1.0, &SPIKES_INPUT_1);

        net.simulate();
        let fitness = net.get_recorder().map(|r| r.classification_rate()).unwrap();
        fitness as f32
     }
}

#[derive(Debug)]
struct Population {
    pool: Vec<BitVec>
}

impl Population {
    fn new(pop_size: usize, dna_len: usize) -> Population {
        let mut pool = Vec::with_capacity(pop_size);

        for _ in 0 .. pop_size {
            let dna = bitvec_random(dna_len);
            pool.push(dna);
        }

        assert!(pool.len() == pop_size);

        Population {
            pool: pool
        }
    }

    // For each member in the population calculate it's fitness.
    fn calc_fitness<T:Genome>(&self) -> Vec<f32> {
        self.pool.iter().by_ref().map(|dna| {
            let genome: T = Genome::decode_from_bitvec(dna);
            genome.fitness()
        }).collect()
    }
}

fn main() {
    let pop = Population::new(10, 20);
    println!("pop:     {:?}", pop);

    let fitness = pop.calc_fitness::<MyGenome>();
    println!("fitness:     {:?}", fitness);

    // select

/*
    let dna = bitvec_random(20);
    let genome: MyGenome = Genome::decode_from_bitvec(&dna);
    let fitness = genome.fitness();

    println!("dna:     {:?}", dna);
    println!("genome:  {:?}", genome);
    println!("fitness: {:?}", fitness);

    let new_dna = mutate_dna(&dna, 0.1);
    println!("new_dna: {:?}", new_dna);
*/
}
