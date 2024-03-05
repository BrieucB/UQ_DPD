#!/usr/bin/env python
import sys
import pickle

def create_param(nd, L, a, gamma, kBT, power, shear_rate):
    simu_param={'m':1, 'nd':nd, 'rc':1, 'L':L, 'shear_rate':shear_rate, 't_dump_every':0.01}
    dpd_param={'a':a, 'gamma':gamma, 'kBT':kBT, 'power':power}

    p={'simu':simu_param, 'dpd':dpd_param}

    with open('parameters', 'wb') as f:
        pickle.dump(p, f)

def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nd', type=float, default=False)
    parser.add_argument('--L', type=float, default=False)
    parser.add_argument('--a', type=float, default=False)
    parser.add_argument('--gamma', type=float, default=False)
    parser.add_argument('--kBT', type=float, default=False)
    parser.add_argument('--power', type=float, default=0.5)
    parser.add_argument('--shear_rate', type=float,  default=1)

    args = parser.parse_args(argv)

    create_param(nd=args.nd, 
                 L=args.L, 
                 a=args.a, 
                 gamma=args.gamma, 
                 kBT=args.kBT, 
                 power=args.power, 
                 shear_rate=args.shear_rate)


if __name__ == '__main__':
    main(sys.argv[1:])