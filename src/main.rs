extern crate evospinn;

use evospinn::*;
use std::ops::Range;

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

fn main() {
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
        tau_m: ns(46_875),
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

    let mut fitness = Box::new(FitnessRecorder::new());
    fitness.add_correct_range(n_output0, ms(0) .. ms(47));
    fitness.add_correct_range(n_output1, ms(47) .. ms(100));
    fitness.add_correct_range(n_output2, ms(100) .. ms(170));

    net.set_recorder(Some(fitness));

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

    net.add_spike_train_float_ms(n_input0, 1.0, &SPIKES_INPUT_0);
    net.add_spike_train_float_ms(n_input1, 1.0, &SPIKES_INPUT_1);

    net.simulate();

    println!("{:?}", net.get_recorder());
}
