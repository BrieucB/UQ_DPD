#!/usr/bin/env python
import sys
import pickle

def create_param(nd, L, a, gamma, kBT):
    simu_param={'m':1, 'nd':nd, 'rc':1, 'L':L}
    dpd_param={'a':a, 'gamma':gamma, 'kBT':kBT, 'power':0.5}

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

    args = parser.parse_args(argv)

    create_param(nd=args.nd, 
                 L=args.L, 
                 a=args.a, 
                 gamma=args.gamma, 
                 kBT=args.kBT)


if __name__ == '__main__':
    main(sys.argv[1:])