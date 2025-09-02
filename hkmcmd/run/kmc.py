#!/usr/bin/env python3
# Author
#    Dylan Gilley
#    dylan.gilley@gmail.com


import sys, datetime
import numpy as np
from hkmcmd import io, system


# Main argument
def main(argv):
    """Driver for conducting pure kMC.
    """

    # Parse the command line arguments
    parser = io.HkmcmdArgumentParser()
    parser.add_argument(dest="starting_counts", type=str, help="Starting counts for each species.")
    parser.add_argument("-steps", dest="steps", type=int, default=100_000, help="Number of steps to run.")
    parser.add_argument("-dump_every", dest="dump_every", type=int, default=1_000, help="Number of steps to dump every.")
    args = parser.parse_args()
    starting_counts = args.starting_counts.split()
    starting_counts = {starting_counts[i]:int(starting_counts[i+1]) for i in range(0, len(starting_counts), 2)}
    steps = int(args.steps)
    dump_every = int(args.dump_every)

    # Read system data
    system_data = system.SystemData(args.system, args.prefix)
    system_data.read_json()

    # Prepare for kMC
    species_names = sorted(starting_counts.keys())
    species_name2idx = {name:idx for idx,name in enumerate(species_names)}
    reactants, products, rates = [], [], []
    for rxn in system_data.reactions:
        rxn.calculate_raw_rate(system_data.hkmcmd["temperature_rxn"])
        rates.append(rxn.rawrate)
        reactants.append([species_name2idx[sp.kind] for sp in rxn.reactant_molecules])
        products.append([species_name2idx[sp.kind] for sp in rxn.product_molecules])
    rates = np.array(rates).flatten()
    change = np.zeros((len(rates), len(species_names)))
    for rxn_idx,rxn in enumerate(system_data.reactions):
        for sp in rxn.reactant_molecules:
            change[rxn_idx, species_name2idx[sp.kind]] -= 1
        for sp in rxn.product_molecules:
            change[rxn_idx, species_name2idx[sp.kind]] += 1
    counts = np.array([starting_counts[sp] for sp in species_names]).reshape(1,-1)
    times = np.zeros((1,1))
    compound = np.concatenate((counts, times), axis=1)

    # Create output file
    filename_output = f"{args.prefix}.kmc.txt"
    with open(filename_output, "w") as f:
        f.write(f"# file written {datetime.datetime.now()}\n")
        f.write(f"# steps: {steps}\n")
        f.write(f"# dump_every: {dump_every}\n")
        f.write("########################################################################\n")
        f.write(f"# {species_names[0]:>6s}") # 14.8e
        f.write(f"{''.join([f'{_:>8s}' for _ in species_names[1:]])}           times\n  ") # 14.8e
        np.savetxt(f, compound, fmt=["%6d"]*len(species_names) + ["%14.8e"], delimiter="  ", newline="\n  ")

    # Loop over steps, running kMC and periodically dumping to output file
    for step_idx in range(0, steps, dump_every):
        if step_idx + dump_every > steps:
            dump_every = steps - step_idx
        counts, times = run_kmc(counts, rates, change, dump_every, times)
        compound = np.concatenate((counts[1:], times[1:]), axis=1)
        compound = compound[-1,:].reshape(1,-1)
        with open(filename_output, "ab") as f:
            np.savetxt(f, compound, fmt=["%6d"]*len(species_names) + ["%14.8e"], delimiter="  ", newline="\n  ")
        counts = counts[-1,:].reshape(1,-1)
        times = times[-1,:].reshape(1,-1)

    return


def run_kmc(
    counts,
    rates,
    change,
    steps,
    times,
):

    # set counts array
    if counts.ndim == 1:
        counts.reshape(1,-1)
    counts = np.concatenate((counts, np.zeros((steps, counts.shape[1]))), axis=0)

    # set times array
    times = np.concatenate((times, np.zeros((steps, 1))), axis=0)

    # set rate_idx to reactant_idx map
    rate_idx2reactant_idx = {idx:np.argwhere(change[idx] == -1)[0][0] for idx in range(len(rates)) }

    # do kmc
    for step in range(0, steps):
        total_rates = np.array([k*counts[step,rate_idx2reactant_idx[idx]] for idx,k in enumerate(rates)])
        u = 0.0
        while u == 0.0:
            u = np.random.uniform(0, 1.0) * np.sum(total_rates)
        reaction = np.where(np.cumsum(total_rates) >= u)[0][0]
        counts[step+1] = counts[step] + change[reaction]
        u = 0.0
        while u == 0.0:
            u = np.random.uniform(0, 1.0)
        times[step+1,0] = times[step,0] + np.log(1.0/u) / np.sum(total_rates)
    return counts, times
        

if __name__ == "__main__":
    main(sys.argv[1:])
