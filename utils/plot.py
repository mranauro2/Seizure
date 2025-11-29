from constants_main import METRICS_EXTENTION
from metric import Metrics

import argparse
import re
import os

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=f"Plot learning trend from a {METRICS_EXTENTION} file", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('filename',                  type=str,                   help=f"Path to the {METRICS_EXTENTION} file to plot")
    
    parser.add_argument('--marker',                  action='store_true',        help="Use a marker to highlight the points")
    parser.add_argument('--show',                    type=str, default='none',   help="If 'none' the train will not be show, if 'full' show full train, if 'points' show test metric for best K min/max validation values, if 'both' show both 'full' and 'points'")
    parser.add_argument('--best_k',                  type=int, default=3,        help="Show K train point")
    parser.add_argument('--start_check',             type=int|float, default=0.2,help="If show is 'full' or 'both' start checking for the best value from a defined epoch. If it is int then it is the epoch num, if it is float then it is the percentage of the total")
    parser.add_argument('--higher_is_better', '-hb', action='store_true',        help="If True, higher metric values are better (e.g., accuracy). If False, lower metric values are better (e.g., loss)")
    
    args= parser.parse_args()
    
    filename:str=           args.filename
    marker:bool=            args.marker
    show:str=               args.show
    start_check:int|float=  args.start_check
    best_k:int=             args.best_k
    higher_is_better:bool=  args.higher_is_better
    
    if not filename.endswith(METRICS_EXTENTION):
        raise ValueError(f"Input does not end with '{METRICS_EXTENTION}' extention")
    
    match = re.search(fr'^(.+)_(\d+)\.{METRICS_EXTENTION}$', os.path.basename(filename))
    
    metric_name= match.group(1) if match else "Metric"
    
    Metrics.plot(*Metrics.load(filename), metric_name=match.group(1), marker=marker, show=show, start_check=start_check, best_k=best_k, higher_is_better=higher_is_better)