#![feature(collections,core)]
extern crate evospinn;

extern crate rand;

use evospinn::*;
use std::ops::Range;
use std::collections::BitVec;
use rand::distributions::IndependentSample;
use rand::distributions::Range as XRange;
use rand::{Rng, Open01, sample};
use std::marker::PhantomData;
use std::cmp::{PartialOrd, Ordering};

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

fn generate_net<R:Recorder>(tau_m_k: time, delay1: time, delay2: time, delay3: time, delay4: time, delay5: time, delay6: time) -> Net<R> {
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

    // delay1: us(625),
    // delay2: ns(15_625)
    // delay3: us(0)
    // delay4: us(0)
    // delay5: us(0)
    // delay6: ns(593_750)
    net.create_synapse(n_innerinp0, Synapse {delay: delay1, weight: 1.0, post_neuron: n_k0});
    net.create_synapse(n_innerinp0, Synapse {delay: delay2, weight: 1.0, post_neuron: n_k1});
    net.create_synapse(n_innerinp0, Synapse {delay: delay3, weight: 1.0, post_neuron: n_k2});

    net.create_synapse(n_innerinp1, Synapse {delay: delay4, weight: 1.0, post_neuron: n_k0});
    net.create_synapse(n_innerinp1, Synapse {delay: delay5, weight: 1.0, post_neuron: n_k1});
    net.create_synapse(n_innerinp1, Synapse {delay: delay6, weight: 1.0, post_neuron: n_k2});

    net.name_neuron(n_output0, "output0");
    net.name_neuron(n_output1, "output1");
    net.name_neuron(n_output2, "output2");
    net.name_neuron(n_input0,  "input0");
    net.name_neuron(n_input1,  "input1");
 
    net
}

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

#[test]
fn test_bitvec_from_u64() {
    let bv = bitvec_from_u64(0b1111, 4);
    assert_eq!(BitVec::from_elem(4, true), bv);
}

#[derive(Debug,Clone)]
struct Dna {
    bits: BitVec
}

impl Dna {
    fn with_capacity(capa: usize) -> Dna {
        Dna {bits: BitVec::with_capacity(capa)}
    }

    fn new() -> Dna {
        Dna {bits: BitVec::new()}
    }

    fn new_random<R:Rng>(rng: &mut R, nbits: usize) -> Dna {
        Dna {bits: BitVec::from_fn(nbits, |_| rng.gen())}
    }

    /// `flip_prob` is the probablity that we flip a bit.
    fn mutate<R:Rng>(&self, rng: &mut R, flip_prob: f32) -> Dna {
        let mut mutant = BitVec::with_capacity(self.bits.len());

        for bit in self.bits.iter() {
            let Open01(r): Open01<f32> = rng.gen();
            let bit = if r < flip_prob { !bit } else { bit };
            mutant.push(bit);
        }

        Dna {bits: mutant}
    }

    fn push_nbits(&mut self, n: usize, value: usize) {
        bitvec_push_bits(&mut self.bits, value, n);
    }

    fn crossover1<R:Rng>(&self, rng: &mut R, other: &Dna) -> (Dna, Dna) {
         let len = self.bits.len();
         assert!(len == other.bits.len());
         assert!(len >= 2); // XXX: otherwise crossover does not make any sense
         let mut res1 = Dna::with_capacity(len);
         let mut res2 = Dna::with_capacity(len);

         let between = XRange::new(1, len);

         // `n': number of bits that are exchanged between the two dna's
         let n = between.ind_sample(rng);
         assert!(n > 0 && n < len);

         let mut i1 = self.bits.iter();
         let mut i2 = other.bits.iter();

         for _ in 0..n {
             res1.bits.push(i2.next().unwrap());
             res2.bits.push(i1.next().unwrap());
         }
         for _ in n..len {
             res1.bits.push(i1.next().unwrap());
             res2.bits.push(i2.next().unwrap());
         }

         (res1, res2)
    }


}

trait Genome {
    fn to_dna(&self) -> Dna;
    fn from_dna(bv: &Dna) -> Self;

    /// Higher values are better
    fn fitness(&self) -> f32;
}


#[derive(Debug)]
struct MyGenome {
    tau_m_k: time, // 46_875
    delay1: time,
    delay2: time,
    delay3: time,
    delay4: time,
    delay5: time,
    delay6: time,
}

impl Genome for MyGenome {
    fn to_dna(&self) -> Dna {
        let mut dna = Dna::new();
        dna.push_nbits(20, self.tau_m_k as usize);
        dna.push_nbits(20, self.delay1 as usize);
        dna.push_nbits(20, self.delay2 as usize);
        dna.push_nbits(20, self.delay3 as usize);
        dna.push_nbits(20, self.delay4 as usize);
        dna.push_nbits(20, self.delay5 as usize);
        dna.push_nbits(20, self.delay6 as usize);
        dna
    }

    fn from_dna(dna: &Dna) -> MyGenome {
        let mut it = dna.bits.iter();
        // XXX test that iterator is exhausted
        MyGenome {
            tau_m_k: bitvec_construct_value(&mut it, 20) as time,
            delay1: bitvec_construct_value(&mut it, 20) as time,
            delay2: bitvec_construct_value(&mut it, 20) as time,
            delay3: bitvec_construct_value(&mut it, 20) as time,
            delay4: bitvec_construct_value(&mut it, 20) as time,
            delay5: bitvec_construct_value(&mut it, 20) as time,
            delay6: bitvec_construct_value(&mut it, 20) as time,
        }
    }

    fn fitness(&self) -> f32 {
        let mut net = generate_net(
            ns(self.tau_m_k),
            ns(self.delay1),
            ns(self.delay2),
            ns(self.delay3),
            ns(self.delay4),
            ns(self.delay5),
            ns(self.delay6),
        );

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

#[derive(Debug,Clone)]
struct Solution {
    dna: Dna,
    fitness: f32
}

#[derive(Debug)]
struct Generation<G:Genome> {
    solutions: Vec<Solution>,
    max_pop_size: usize,
    _phantom: PhantomData<G>
}

impl<G:Genome> Generation<G> {
    pub fn new(max_pop_size: usize) -> Generation<G> {
        assert!(max_pop_size > 0);
        Generation {
            solutions: Vec::with_capacity(max_pop_size),
            max_pop_size: max_pop_size,
            _phantom: PhantomData,
        }
    }

    pub fn fill<F:FnMut() -> Dna>(&mut self, mut f: F) {
        while self.solutions.len() < self.max_pop_size {
            assert!(self.add(f()));
        }
    }

    pub fn sort(&mut self) {
        (&mut self.solutions[..]).sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal).reverse());
    }

    pub fn best(&mut self) -> &Solution {
        self.sort();
        &self.solutions[0]
    }

    /// Adds a solution to the pool. Returns false if maximum population size is reached,
    /// in which case the solution is not added.
    ///
    /// TODO: keep the best solution
    pub fn add(&mut self, dna: Dna) -> bool {
        if self.solutions.len() < self.max_pop_size {
             let genome: G = Genome::from_dna(&dna);
             let fitness = genome.fitness();
             self.solutions.push(Solution{dna: dna, fitness: fitness});
             true
         } else {
             false
         }
    }

    fn add_solution(&mut self, solution: Solution) -> bool {
        if self.solutions.len() < self.max_pop_size {
             self.solutions.push(solution);
             true
         } else {
             false
         }
    }

    /// Select one solution out of k. Returns index.
    fn tournament_selection<R:Rng>(&mut self, rng: &mut R, k: usize) -> usize {
        assert!(!self.solutions.is_empty());

        let mut best: Option<(usize, f32)> = None;

        let sample = sample(rng, 0..self.solutions.len(), k);

        for i in sample {
            let sol = &mut self.solutions[i];
            let fitness = sol.fitness;
            best = match best {
                Some((j, f)) if f > fitness => Some((j, f)),
                _ => Some((i, fitness))
            };
        }
        return best.unwrap().0;
    }

    /// Creata a new generation of size `pop_size` based on the current generation.
    pub fn reproduce<R:Rng>(&mut self, rng: &mut R, pop_size: usize, tournament_size: usize, mutate_prob: f32) -> Generation<G> {
        assert!(!self.solutions.is_empty());
        self.sort();
        let mut new_gen: Generation<G> = Generation::new(pop_size);
        let _ = new_gen.add_solution(self.solutions[0].clone()); // add best solution

        loop {
            let parent1 = self.tournament_selection(rng, tournament_size);
            let parent1 = self.solutions[parent1].clone();
            let parent2 = self.tournament_selection(rng, tournament_size);
            let parent2 = self.solutions[parent2].clone();

            let (child1, child2) = parent1.dna.crossover1(rng, &parent2.dna);

            let child1 = child1.mutate(rng, mutate_prob);
            let child2 = child2.mutate(rng, mutate_prob);

            if !new_gen.add_solution(parent1) { break }
            if !new_gen.add_solution(parent2) { break }
            if !new_gen.add(child1) { break }
            if !new_gen.add(child2) { break }
        }

        new_gen
    }
}

const POP_SIZE: usize = 100;

fn main() {
    use rand::SeedableRng;
    use rand::isaac::Isaac64Rng;
    let mut rng: Isaac64Rng = SeedableRng::from_seed(&[0, 1, 2, 3][..]);

    // min/max
    let mut pop: Generation<MyGenome> = Generation::new(POP_SIZE);
    pop.fill(|| Dna::new_random(&mut rng, 140));
    println!("best:     {:?}", pop.best());

    for gen in 0..10 {
        println!("--------------------------");
        println!("Generation {}", gen);
        pop = pop.reproduce(&mut rng, POP_SIZE, 3, 0.05);
        println!("best:     {:?}", pop.best());
        let genome = MyGenome::from_dna(&pop.best().dna);
        println!("genome:   {:?}", genome);
    }
}
