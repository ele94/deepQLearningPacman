#!/bin/bash
python pacman.py -p MyApproximateQAgent -x 10000 -n 10010 -l smallGrid --frameTime 0.0
mv approxQweights.pkl newtables/gridapprox.pkl
python pacman.py -p MyApproximateQAgent -x 10000 -n 10010 -l capsuleClassic --frameTime 0.0
mv approxQweights.pkl newtables/capsuleapprox.pkl
python pacman.py -p MyApproximateQAgent -x 10000 -n 10010 -l smallClassic --frameTime 0.0
mv approxQweights.pkl newtables/classicapprox.pkl
