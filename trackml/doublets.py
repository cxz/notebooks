""" 

The iterative procedure for track reconstruction starts building doublets and then tripĺets from 
pairs of layers.

* Triplet seeding is much faster and has a lower fake rate than pairs
* First, triplet seeding for higher pT (iteration 0) and lower pT (iteration 1)
* Then pixel pairs (iteration 2)
* Iteration 3 uses triplets for displaced tracks
* Iteration 4 tries to find tracks which miss a pixel layer and also tracks wich may decay within a couple of cm
of the production vertex
* Further iterations designed for tracks significantly displaced from beam line or tracks which do not leave sufficient 
pixel hits to be found in earlier iterations.
 
| https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideTrackReco
|--> https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideIterativeTracking
|--> https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideTrackRecoSequences#TrackReco

d0 and z0 refer to the transverse and longitudinal impact parameters.

"""


def initial_step():
    """    
    See https://github.com/cms-sw/cmssw/blob/master/RecoTracker/IterativeTracking/python/InitialStep_cff.py
    """

    seed_layers = []  # (?) from TkSeedingLayers

    # regions, see RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeanSpot
    pT = 0.6  # min pT in GeV
    d0 = 0.02  # origin radius -- transverse impact parameter
    sigma_z = 4.0  # nSigmaZ, z0 = 4 * sigma

    # see TkHitPairs.hitPairEDPProducer
    hit_doublets = []  # doublets(seed_layers, tracking_regions, max_element=0, product_intermediate_hit_doubles = True)

    # see RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer
    # triplets(doublets, produce_seeding_hit_sets = True,
    # seed_comparitor_pset = LowPtClusterShapeSeedComparitor
    hit_triplets = []

    # RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer
    seeds = []  # from hit_triplets

    # build quadruplets from hit doublets + layer pairs
    # phi tolerance, max chi2, bending correction, fast circle, theta cut, etc.

    # ... continue with
    # build track candidates, fitting, vertices, classification, selection
    raise

