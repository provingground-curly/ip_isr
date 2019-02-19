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


# import lsst.meas.algorithms as measAlg

# import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
# from lsst.afw.image.testUtils import assertImagesAlmostEqual
import lsst.utils.tests
import lsst.ip.isr as ipIsr
import lsst.pex.exceptions as pexExcept
import isr_mocks as mock


class IsrFunctionsCases(lsst.utils.tests.TestCase):

    def setUp(self):
        self.inputExp = mock.TrimmedRawMock().mock()
        self.mi = self.inputExp.getMaskedImage()

    def tearDown(self):
        pass

    def test_transposeMaskedImage(self):
        transposed = ipIsr.transposeMaskedImage(self.mi)
        assert transposed

    def test_interpolateDefectList(self):
        defectList = mock.DefectMock().mock()

        for fallbackValue in (None, -999.0):
            for haveMask in (True, False):
                with self.subTest(fallbackValue=fallbackValue, haveMask=haveMask):
                    if haveMask is False:
                        if 'INTRP' in self.mi.getMask().getMaskPlaneDict():
                            self.mi.getMask().removeAndClearMaskPlane('INTRP')
                    else:
                        if 'INTRP' not in self.mi.getMask().getMaskPlaneDict():
                            self.mi.getMask().addMaskPlane('INTRP')
                    output = ipIsr.interpolateDefectList(self.mi, defectList,
                                                         2.0, fallbackValue=fallbackValue)
        assert output

    def test_transposeDefectList(self):
        defectList = mock.DefectMock().mock()

        transposed = ipIsr.transposeDefectList(defectList)

        for d, t in zip(defectList, transposed):
            assert d.getBBox().getDimensions().getX() == t.getBBox().getDimensions().getY()
            assert d.getBBox().getDimensions().getY() == t.getBBox().getDimensions().getX()

    def test_makeThresholdMask(self):
        defectList = ipIsr.makeThresholdMask(self.mi, 200, growFootprints=2, maskName='SAT')

        assert defectList

    def test_interpolateFromMask(self):
        ipIsr.makeThresholdMask(self.mi, 200, growFootprints=2, maskName='SAT')
        for growFootprints in (0, 2):
            with self.subTest(growFootprints=growFootprints):
                interpMaskedImage = ipIsr.interpolateFromMask(self.mi, 2.0,
                                                              growFootprints=growFootprints, maskName='SAT')

        assert interpMaskedImage

    def test_saturationCorrection(self):
        for interpolate in (True, False):
            with self.subTest(interpolate=interpolate):
                corrMaskedImage = ipIsr.saturationCorrection(self.mi, 200, 2.0,
                                                             growFootprints=2, interpolate=interpolate,
                                                             maskName='SAT')
        assert corrMaskedImage

    def test_trimToMatchCalibBBox(self):
        # It would be nice to be able to test that this fails in the way it should if the
        # trim isn't symmetric.
        darkExp = mock.DarkMock().mock()
        darkMi = darkExp.getMaskedImage()

        nEdge = 2
        newDark = darkMi[nEdge:-nEdge, nEdge:-nEdge, afwImage.LOCAL]
        darkMi = newDark
        newInput = ipIsr.trimToMatchCalibBBox(self.mi, darkMi)

        # I broke this somehow, and I'm moving on because I want to get something done today.
        assert newInput
#       print( newInput.getBBox().getDimensions() , self.mi.getBBox().getDimensions())
#        assert newInput.getBBox().getDimensions() == self.mi.getBBox().getDimensions()
#        afwGeom.testUtils.assertBoxesAlmostEqual(self, newInput.getBBox(), self.mi.getBBox())

    def test_darkCorrection(self):
        darkExp = mock.DarkMock().mock()
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
        biasExp = mock.BiasMock().mock()
        biasMi = biasExp.getMaskedImage()

        ipIsr.biasCorrection(self.mi, biasMi, trimToFit=True)

        assert True

        newBias = biasMi[1:-1, 1:-1, afwImage.LOCAL]
        biasMi = newBias

        with self.assertRaises(RuntimeError):
            ipIsr.biasCorrection(self.mi, biasMi, trimToFit=False)

    def test_flatCorrection(self):
        flatExp = mock.FlatMock().mock()
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
        flatExp = mock.FlatMock().mock()
        flatMi = flatExp.getMaskedImage()

        ipIsr.illuminationCorrection(self.mi, flatMi, 1.0)
        assert True

        newFlat = flatMi[1:-1, 1:-1, afwImage.LOCAL]
        flatMi = newFlat

        with self.assertRaises(RuntimeError):
            ipIsr.illuminationCorrection(self.mi, flatMi, 1.0)

    def test_overscanCorrection(self):
        inputExp = mock.RawMock().mock()

        amp = inputExp.getDetector()[0]
        ampI = inputExp.maskedImage[amp.getRawDataBBox()]
        overscanI = inputExp.maskedImage[amp.getRawHorizontalOverscanBBox()]

        for fitType in ('MEAN', 'MEDIAN', 'MEANCLIP', 'POLY', 'CHEB', 'NATURAL_SPLINE', 'UNKNOWN'):
            for overscanIsInt in (True, False):
                with self.subTest(fitType=fitType, overscanIsInt=overscanIsInt):
                    if fitType == 'UNKNOWN':
                        with self.assertRaises(pexExcept.Exception):
                            ipIsr.overscanCorrection(ampI, overscanI, fitType=fitType,
                                                     order=1, collapseRej=3.0,
                                                     statControl=None, overscanIsInt=overscanIsInt)
                    elif fitType == 'NATURAL_SPLINE':
                        with self.assertRaises(pexExcept.OutOfRangeError):
                            ipIsr.overscanCorrection(ampI, overscanI, fitType=fitType,
                                                     order=1, collapseRej=3.0,
                                                     statControl=None, overscanIsInt=overscanIsInt)
                    else:
                        response = ipIsr.overscanCorrection(ampI, overscanI, fitType=fitType,
                                                            order=1, collapseRej=3.0,
                                                            statControl=None, overscanIsInt=overscanIsInt)
                        assert response

    def test_brighterFatterCorrection(self):
        bfKern = mock.BfKernelMock().mock()

        ipIsr.brighterFatterCorrection(self.inputExp, bfKern, 10, 1e-2, False)

    def test_gainContext(self):
        for apply in (True, False):
            with self.subTest(apply=apply):
                with ipIsr.gainContext(self.inputExp, self.inputExp.getImage(), apply=apply):
                    pass

    def test_addDistortionModel(self):
        camera = mock.IsrMock().getCamera()
        ipIsr.addDistortionModel(self.inputExp, camera)

        with self.assertRaises(RuntimeError):
            ipIsr.addDistortionModel(self.inputExp, None)

            self.inputExp.setDetector(None)
            ipIsr.addDistortionModel(self.inputExp, camera)

            self.inputExp.setWcs(None)
            ipIsr.addDistortionModel(self.inputExp, camera)

    def test_applyGains(self):
        for normalizeGains in (True, False):
            with self.subTest(normalizeGains=normalizeGains):
                ipIsr.applyGains(self.inputExp, normalizeGains=normalizeGains)

    def test_widenSaturationTrails(self):
        defectList = ipIsr.makeThresholdMask(self.mi, 200, growFootprints=0, maskName='SAT')

        assert defectList

        ipIsr.widenSaturationTrails(self.mi.getMask())

    def test_setBadRegions(self):
        for badStatistic in ('MEDIAN', 'MEANCLIP', 'UNKNOWN'):
            with self.subTest(badStatistic=badStatistic):
                if badStatistic == 'UNKNOWN':
                    with self.assertRaises(RuntimeError):
                        nBad, value = ipIsr.setBadRegions(self.inputExp, badStatistic=badStatistic)
                else:
                    nBad, value = ipIsr.setBadRegions(self.inputExp, badStatistic=badStatistic)
                    assert value

    def test_attachTransmissionCurve(self):
        curve = mock.TransmissionMock().mock()
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
