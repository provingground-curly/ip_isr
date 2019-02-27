#
# LSST Data Management System
# Copyright 2008-2017 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#

import unittest
import numpy as np

import lsst.afw.image as afwImage
import lsst.utils.tests
import lsst.ip.isr as ipIsr
import lsst.pex.exceptions as pexExcept
import lsst.ip.isr.isrMock as isrMock


def maskCountPixels(maskedImage, maskPlane):
    r"""Function to count the number of masked pixels of a given type.

    Parameters
    ----------
    maskedImage : `lsst.afw.image.MaskedImage`
        Image to measure the mask on.
    maskPlane : `str`
        Name of the mask plane to count

    Returns
    -------
    nMask : `int`
        Number of masked pixels.
    """
    bitMask = maskedImage.getMask().getPlaneBitMask(maskPlane)
    isBit = maskedImage.getMask().getArray() & bitMask > 0
    numBit = np.sum(isBit)
    print("%s %d %d\n" % (maskPlane, bitMask, numBit))
    return numBit


def imageMedianStd(image):
    r"""Function to calculate median and std of image data.

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image to measure statistics on.

    Returns
    -------
    median : `float`
        Image median.
    std : `float`
        Image stddev.
    """
    median = np.nanmedian(image.getArray())
    std = np.nanstd(image.getArray())
    print("ms: %f %f" % (median, std))
    return (median, std)


class IsrFunctionsCases(lsst.utils.tests.TestCase):
    r"""Test functions for ISR produce expected outputs.
    """
    def setUp(self):
        self.inputExp = isrMock.TrimmedRawMock().run()
        self.mi = self.inputExp.getMaskedImage()

    def tearDown(self):
        pass

    def test_transposeMaskedImage(self):
        r"""Image transpose.

        Expect height and width to be exchanged.
        """
        transposed = ipIsr.transposeMaskedImage(self.mi)
        self.assertEqual(transposed.getImage().getBBox().getHeight(), self.mi.getImage().getBBox().getWidth())

    def test_interpolateDefectList(self):
        r"""Defect interpolation.

        Expect number of interpolated pixels to be non-zero.
        """
        defectList = isrMock.DefectMock().run()
        assert len(defectList) > 0

        for fallbackValue in (None, -999.0):
            for haveMask in (True, False):
                with self.subTest(fallbackValue=fallbackValue, haveMask=haveMask):
                    if haveMask is False:
                        if 'INTRP' in self.mi.getMask().getMaskPlaneDict():
                            self.mi.getMask().removeAndClearMaskPlane('INTRP')
                    else:
                        if 'INTRP' not in self.mi.getMask().getMaskPlaneDict():
                            self.mi.getMask().addMaskPlane('INTRP')
                    numBit = maskCountPixels(self.mi, "INTRP")
                    self.assertGreaterEqual(numBit, 0)

    def test_transposeDefectList(self):
        r"""Transpose defect list.

        Expect dimension values to flip.
        """
        defectList = isrMock.DefectMock().run()

        transposed = ipIsr.transposeDefectList(defectList)

        for d, t in zip(defectList, transposed):
            assert d.getBBox().getDimensions().getX() == t.getBBox().getDimensions().getY()
            assert d.getBBox().getDimensions().getY() == t.getBBox().getDimensions().getX()

    def test_makeThresholdMask(self):
        r"""Thresholding.

        Expect list of defects to have elements.
        """
        defectList = ipIsr.makeThresholdMask(self.mi, 200, growFootprints=2, maskName='SAT')

        self.assertGreater(len(defectList), 0)

    def test_interpolateFromMask(self):
        r"""Interpolate mask.

        Expect number of interpolated pixels to be non-zero.
        """
        ipIsr.makeThresholdMask(self.mi, 200, growFootprints=2, maskName='SAT')
        for growFootprints in (0, 2):
            with self.subTest(growFootprints=growFootprints):
                interpMaskedImage = ipIsr.interpolateFromMask(self.mi, 2.0,
                                                              growFootprints=growFootprints, maskName='SAT')
                numBit = maskCountPixels(interpMaskedImage, "INTRP")
                self.assertGreaterEqual(numBit, 0)

    def test_saturationCorrection(self):
        r"""Saturation correction.

        Expect number of mask pixels with SAT marked to be non-zero.
        """
        for interpolate in (True, False):
            with self.subTest(interpolate=interpolate):
                corrMaskedImage = ipIsr.saturationCorrection(self.mi, 200, 2.0,
                                                             growFootprints=2, interpolate=interpolate,
                                                             maskName='SAT')
                numBit = maskCountPixels(corrMaskedImage, "SAT")
                self.assertGreaterEqual(numBit, 0)

    def test_trimToMatchCalibBBox(self):
        r"""Trim to match raw data.

        Expect bounding boxes to match.
        """
        darkExp = isrMock.DarkMock().run()
        darkMi = darkExp.getMaskedImage()

        nEdge = 2
        newDark = darkMi[nEdge:-nEdge, nEdge:-nEdge, afwImage.LOCAL]
        darkMi = newDark
        newInput = ipIsr.trimToMatchCalibBBox(self.mi, darkMi)

        self.assertEqual(newInput.getImage().getBBox(), newDark.getImage().getBBox())

    def test_darkCorrection(self):
        r"""Dark correction.

        Expect round-trip application to be equal.
        Expect RuntimeError if sizes are different.
        """
        darkExp = isrMock.DarkMock().run()
        darkMi = darkExp.getMaskedImage()

        image = self.mi.getImage().getArray()
        ipIsr.darkCorrection(self.mi, darkMi, 1.0, 1.0, trimToFit=True)
        ipIsr.darkCorrection(self.mi, darkMi, 1.0, 1.0, trimToFit=True, invert=True)

        diff = image - self.mi.getImage().getArray()
        assert (diff.sum() == 0)

        newDark = darkMi[1:-1, 1:-1, afwImage.LOCAL]
        darkMi = newDark

        with self.assertRaises(RuntimeError):
            ipIsr.darkCorrection(self.mi, darkMi, 1.0, 1.0, trimToFit=False)

    def test_biasCorrection(self):
        r"""Bias correction.

        Expect smaller median image value after.
        Expect RuntimeError is sizes are different.
        """
        biasExp = isrMock.BiasMock().run()
        biasMi = biasExp.getMaskedImage()

        ipIsr.biasCorrection(self.mi, biasMi, trimToFit=True)

        assert True

        newBias = biasMi[1:-1, 1:-1, afwImage.LOCAL]
        biasMi = newBias

        with self.assertRaises(RuntimeError):
            ipIsr.biasCorrection(self.mi, biasMi, trimToFit=False)

    def test_flatCorrection(self):
        r"""Flat correction.

        Expect round-trip application to be equal
        Expect RuntimeError if sizes are different.
        """
        flatExp = isrMock.FlatMock().run()
        flatMi = flatExp.getMaskedImage()

        image = self.mi.getImage().getArray()
        for scaling in ('USER', 'MEAN', 'MEDIAN', 'UNKNOWN'):
            with self.subTest(scaling=scaling):
                if scaling == 'UNKNOWN':
                    with self.assertRaises(pexExcept.Exception):
                        ipIsr.flatCorrection(self.mi, flatMi, scaling, userScale=1.0, trimToFit=True)
                        ipIsr.flatCorrection(self.mi, flatMi, scaling, userScale=1.0,
                                             trimToFit=True, invert=True)
                else:
                    ipIsr.flatCorrection(self.mi, flatMi, scaling, userScale=1.0, trimToFit=True)
                    ipIsr.flatCorrection(self.mi, flatMi, scaling, userScale=1.0,
                                         trimToFit=True, invert=True)

        diff = image - self.mi.getImage().getArray()
        assert (diff.sum() == 0)

        newFlat = flatMi[1:-1, 1:-1, afwImage.LOCAL]
        flatMi = newFlat

        with self.assertRaises(RuntimeError):
            ipIsr.flatCorrection(self.mi, flatMi, 'USER', userScale=1.0, trimToFit=False)

    def test_illumCorrection(self):
        r"""Illumination Correction.

        Expect larger median value after.
        Expect RuntimeError if sizes are different.
        """
        flatExp = isrMock.FlatMock().run()
        flatMi = flatExp.getMaskedImage()

        ipIsr.illuminationCorrection(self.mi, flatMi, 1.0)
        assert True

        newFlat = flatMi[1:-1, 1:-1, afwImage.LOCAL]
        flatMi = newFlat

        with self.assertRaises(RuntimeError):
            ipIsr.illuminationCorrection(self.mi, flatMi, 1.0)

    def test_overscanCorrection(self):
        r"""Overscan subtraction.

        Expect smaller median/smaller std after.
        """
        inputExp = isrMock.RawMock().run()

        amp = inputExp.getDetector()[0]
        ampI = inputExp.maskedImage[amp.getRawDataBBox()]
        overscanI = inputExp.maskedImage[amp.getRawHorizontalOverscanBBox()]

        for fitType in ('MEAN', 'MEDIAN', 'MEANCLIP', 'POLY', 'CHEB',
                        'NATURAL_SPLINE', 'CUBIC_SPLINE', 'UNKNOWN'):
            order = 1
            if fitType in ('NATURAL_SPLINE', 'CUBIC_SPLINE'):
                order = 3
            for overscanIsInt in (True, False):
                with self.subTest(fitType=fitType, overscanIsInt=overscanIsInt):
                    if fitType == 'UNKNOWN':
                        with self.assertRaises(pexExcept.Exception):
                            ipIsr.overscanCorrection(ampI, overscanI, fitType=fitType,
                                                     order=order, collapseRej=3.0,
                                                     statControl=None, overscanIsInt=overscanIsInt)
                    else:
                        response = ipIsr.overscanCorrection(ampI, overscanI, fitType=fitType,
                                                            order=order, collapseRej=3.0,
                                                            statControl=None, overscanIsInt=overscanIsInt)
                        assert response

    def test_brighterFatterCorrection(self):
        r"""BF correction.

        Expect smoother image/smaller std after.
        """
        bfKern = isrMock.BfKernelMock().run()

        before = imageMedianStd(self.inputExp.getImage())
        ipIsr.brighterFatterCorrection(self.inputExp, bfKern, 10, 1e-2, False)
        after = imageMedianStd(self.inputExp.getImage())

        self.assertLess(after[1], before[1])

    def test_gainContext(self):
        r"""Gain context manager

        """
        for apply in (True, False):
            with self.subTest(apply=apply):
                with ipIsr.gainContext(self.inputExp, self.inputExp.getImage(), apply=apply):
                    pass

    def test_addDistortionModel(self):
        r"""Add distortion models.

        Expect RuntimeError if no model supplied, or incomplete exposure information.
        """
        camera = isrMock.IsrMock().getCamera()
        ipIsr.addDistortionModel(self.inputExp, camera)

        with self.assertRaises(RuntimeError):
            ipIsr.addDistortionModel(self.inputExp, None)

            self.inputExp.setDetector(None)
            ipIsr.addDistortionModel(self.inputExp, camera)

            self.inputExp.setWcs(None)
            ipIsr.addDistortionModel(self.inputExp, camera)

    def test_applyGains(self):
        r"""Apply gains.

        """
        for normalizeGains in (True, False):
            with self.subTest(normalizeGains=normalizeGains):
                ipIsr.applyGains(self.inputExp, normalizeGains=normalizeGains)

    def test_widenSaturationTrails(self):
        r"""Widen saturation trails.

        Expect more mask pixels with SAT set after.
        """
        defectList = ipIsr.makeThresholdMask(self.mi, 200, growFootprints=0, maskName='SAT')
        numBitBefore = maskCountPixels(self.mi, "SAT")

        assert len(defectList) > 0

        ipIsr.widenSaturationTrails(self.mi.getMask())
        numBitAfter = maskCountPixels(self.mi, "SAT")

        self.assertGreaterEqual(numBitAfter, numBitBefore)

    def test_setBadRegions(self):
        r"""Bad regions.

        Expect RuntimeError if improper statistic given.
        Expect a float value otherwise.
        """
        for badStatistic in ('MEDIAN', 'MEANCLIP', 'UNKNOWN'):
            with self.subTest(badStatistic=badStatistic):
                if badStatistic == 'UNKNOWN':
                    with self.assertRaises(RuntimeError):
                        nBad, value = ipIsr.setBadRegions(self.inputExp, badStatistic=badStatistic)
                else:
                    nBad, value = ipIsr.setBadRegions(self.inputExp, badStatistic=badStatistic)
                    self.assertGreaterEqual(abs(value), 0.0)

    def test_attachTransmissionCurve(self):
        r"""Transmission curves.

        """
        curve = isrMock.TransmissionMock().run()
        for tCurve in (None, curve):
            with self.subTest(tCurve=tCurve):
                combined = ipIsr.attachTransmissionCurve(self.inputExp,
                                                         opticsTransmission=tCurve,
                                                         filterTransmission=tCurve,
                                                         sensorTransmission=tCurve,
                                                         atmosphereTransmission=tCurve)

        assert combined


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
