#! /usr/bin/env python3

import os
#import numpy as np
from argparse import ArgumentParser
import yaml


def ParsingFunc( text ):
    pass

###########################################################

if __name__ == '__main__':

    filename_default = '../../parameter/parameters-default00001eq.yaml'
    with open(filename_default, 'rb') as f:
        parameters_default = yaml.load(f, Loader = yaml.CLoader)

    default_count = parameters_default["numObjects"]
    
    parser = ArgumentParser()       
    parser.add_argument('-s', '--bubble-size', dest = 'Size', default = None )
    parser.add_argument('-c', '--bubble-count', dest = 'Count', default = default_count )
    parser.add_argument('-D', '--input-directory-location', dest = 'Dir', default = '.' )
    parser.add_argument('-O', '--output-directory-location', dest = 'OutDir', default = None )
    parser.add_argument('-m', '--mark', dest = 'ids', action = 'append', nargs = 2, default = None)

    args = parser.parse_args()
    
    if(args.ids):
        ids = [int(x[0]) for x in args.ids] #int(args.ids)
        types = [int(x[1]) for x in args.ids]

    if not ( args.Count or args.Size ):
        exit( "Provide either size or number of microbubbles in the system." )
    elif args.Count:
        NCount = int( args.Count )
    elif args.Size:
        exit( "Size specification not implemented yet." )
    
    FileDir = os.listdir( args.Dir )
    FileDirXYZ = list( filter( lambda FD: FD[-4:] == ".xyz", FileDir ) )

    if args.OutDir:
        OutDir = args.OutDir
        if not os.path.isdir( OutDir ):
            os.mkdir( OutDir )
        
    else:
        OutDir = args.Dir 

    print( "Output will be written to directory:", OutDir )
        
        
    for F in FileDirXYZ:
        with open( os.path.join( args.Dir, F ), "r" ) as FL:
            FLines = FL.readlines()
            
            Header = FLines[:2]

            Coors = FLines[2:]
            
            if(args.ids):
                it = 0
                for idt in ids:
                    list_tmp = list(Coors[idt])
                    list_tmp[0] = str(types[it])
                    Coors[idt] = ''.join(list_tmp)
                    print(Coors[idt])
                    it += 1

            if len( Coors ) % NCount:
                exit( "Check specified number of microbubbles (-c)." )

            MBSize = int( len( Coors ) / NCount )

                
        for MBC in range( NCount ):

            Name = "MB_" + str( MBC ) + "_" + F

            MBHeader = Header[:]
            MBHeader[0] = f"{MBSize}\n"
            MBHeader[1] = "# Separate trajectory file for a single microbubble " + str( MBC ) + "\n"  
            
            
            with open( os.path.join( OutDir, Name ), "w" ) as SepMB:
                SepMB.writelines( MBHeader )
                SepMB.writelines( Coors[ MBC * MBSize : ( MBC + 1 ) * MBSize ] ) 
