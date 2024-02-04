
from RidgeSegment import ridgeSegment
from RidgeOrient import ridgeOrient
from RidgeFreq import ridgeFreq
from RidgeFilter import ridgeFilter


def imageEnhance(img):
    blksze = 16
    thresh = 0.1
    normim, mask = ridgeSegment(img, blksze, thresh)

    gradientsigma = 1
    blocksigma = 7
    orientsmoothsigma = 7
    orientim = ridgeOrient(normim, gradientsigma, blocksigma, orientsmoothsigma)

    blksze = 38
    windsze = 5
    minWaveLength = 5
    maxWaveLength = 15
    freq, medfreq = ridgeFreq(normim, mask, orientim, blksze, windsze, minWaveLength, maxWaveLength)

    freq = medfreq * mask
    kx = 0.65
    ky = 0.65
    newim = ridgeFilter(normim, orientim, freq, kx, ky)

    return (newim < -3)
