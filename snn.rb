# Zustand (state) + Configuration (params)
#SRM_KernelBased_Config = Struct.new(:tau_m, :tau_ref, :ref_weight, :const_threshold, :arp)

class SRM_KernelBased_Config
  attr_reader :tau_m, :tau_ref, :ref_weight, :const_threshold, :arp

  def initialize(tau_m: 0.0, tau_ref: 0.0, ref_weight: 0.0, const_threshold: 0.0, arp: 0.0)
    @tau_m = tau_m
    @tau_ref = tau_ref
    @ref_weight = ref_weight
    @const_threshold = const_threshold
    @arp = arp
  end

  def create_neuron
    Neuron_SRM_KernelBased.new(self)
  end
end


class Neuron_SRM_KernelBased
  attr_reader :mem_pot

  def initialize(cfg)
    @cfg = cfg
    @end_of_refr_period = 0.0
    @last_spike_time = 0.0
    @mem_pot = 0.0
  end

  # Returns true if neuron fires, otherwise false.
  def spike(timestamp, weight)
    # Time since end of last absolute refractory period  
    delta = timestamp - @end_of_refr_period

    # Abort if still in absolute refractory period
    return false if delta < 0.0

    # Calculate dynamic threshold
    # Border condition: threshold = const_threshold (delta = inf)
    # ignore this border case.
    threshold = @cfg.const_threshold +
                @cfg.ref_weight * Math.exp(-delta / @cfg.tau_ref)

    # We don't have to care about the border case,
    # because @mem_pot is 0.0 at the beginning, so 
    # whatever value @last_spike_time has, it does not 
    # matter.
    d = timestamp - @last_spike_time
    if @cfg.tau_m > 0.0
      decay = Math.exp(-d / @cfg.tau_m)
      @mem_pot *= decay
      @mem_pot += weight
    else
      @mem_pot = weight
    end

    # Update last spike time
    @last_spike_time = timestamp

    if @mem_pot >= threshold
      @end_of_refr_period = timestamp + @cfg.arp
      true
    else
      false
    end
  end
end

# tau_ref?
K = SRM_KernelBased_Config.new(arp: 0.5, tau_m: 0.04, tau_ref: 0.5, const_threshold: 1.1)


# Filter all spikes below 0.6 out
Input = SRM_KernelBased_Config.new(arp: 1.0, tau_m: 0.0, tau_ref: 0.5, ref_weight: 0.0, const_threshold: 0.6)

p K, Input

n1 = Input.create_neuron
p n1

for i in 1..100
  t = i / 10.0
  puts "------------------------------"
  p t
  fire = n1.spike(t, 0.6)
  p fire
  p n1
end
