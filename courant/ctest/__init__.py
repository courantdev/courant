"""The Courant Test: can AI do engineering simulation, and can it tell when it is wrong?

(TRD §6, RRD RM-02.) v0 is the judge track, built from 31 real OpenFOAM v2406
runs in `certification/corpus/`. Tasks never carry their labels. Labels live
in a separate, hash-locked file that only the scorer reads.
"""
