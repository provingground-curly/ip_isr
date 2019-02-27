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

import lsst.utils.tests
from lsst.ip.isr.isrTask import (IsrTask, IsrTaskConfig)
from lsst.ip.isr.isrQa import IsrQaConfig
import lsst.ip.isr.isrMock as isrMock


# CZW Need better place to put test functions.
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
    return numBit


class IsrTaskTestCases(lsst.utils.tests.TestCase):
    r"""Test IsrTask methods with trimmed raw data.
    """
    def setUp(self):
        self.config = IsrTaskConfig()
        self.config.qa = IsrQaConfig()
        self.task = IsrTask(config=self.config)
        self.dataRef = isrMock.DataRefMock()
        self.camera = isrMock.IsrMock().getCamera()

        self.inputExp = isrMock.TrimmedRawMock().run()
        self.amp = self.inputExp.getDetector()[0]
        self.mi = self.inputExp.getMaskedImage()

    def tearDown(self):
        pass

    def test_readIsrData(self):
        r"""Test that all necessary calibration frames are retrieved.
        """
        for transmission in (True, False):
            with self.subTest(transmission=transmission):
                self.config.doAttachTransmissionCurve = transmission
                self.task = IsrTask(config=self.config)
                results = self.task.readIsrData(self.dataRef, self.inputExp)
                assert results
                if self.config.doBias is True:
                    assert results.bias is not None
                if self.config.doDark is True:
                    assert results.dark is not None
                if self.config.doFlat is True:
                    assert results.flat is not None
                if self.config.doFringe is True:
                    assert results.fringes is not None
                if self.config.doDefect is True:
                    assert results.defects is not None
                if self.config.doBrighterFatter is True:
                    assert results.bfKernel is not None
                if self.config.doAttachTransmissionCurve is True:
                    assert results.opticsTransmission is not None
                    assert results.filterTransmission is not None
                    assert results.sensorTransmission is not None
                    assert results.atmosphereTransmission is not None

    def test_ensureExposure(self):
        r"""Test that an exposure has a usable instance class.
        """
        assert self.task.ensureExposure(self.inputExp, self.camera, 0)

    def test_convertItoF(self):
        r"""Test conversion from integer to floating point pixels.
        """
        assert self.task.convertIntToFloat(self.inputExp)

    def test_updateVariance(self):
        r"""Test the variance calculation works.

        The variance image should have a larger median value after
        this operation.
        """
        medianBefore = np.median(self.inputExp.variance[self.amp.getBBox()].getArray())
        self.task.updateVariance(self.inputExp, self.amp)
        medianAfter = np.median(self.inputExp.variance[self.amp.getBBox()].getArray())
        self.assertGreater(medianAfter, medianBefore)

    def test_darkCorrection(self):
        r"""Test the dark subtraction works.

        The median image value should decrease after this operation.
        """
        darkIm = isrMock.DarkMock().run()

        medianBefore = np.median(self.inputExp.image[self.amp.getBBox()].getArray())
        self.task.darkCorrection(self.inputExp, darkIm)
        medianAfter = np.median(self.inputExp.image[self.amp.getBBox()].getArray())
        self.assertLess(medianAfter, medianBefore)

    def test_darkCorrectionDefaultDarkTime(self):
        r"""Test the dark subtraction works with missing visitInfo.

        The median image value should decrease after this operation.
        """
        darkIm = isrMock.DarkMock().run()
        darkIm.getInfo().setVisitInfo(None)

        medianBefore = np.median(self.inputExp.image[self.amp.getBBox()].getArray())
        self.task.darkCorrection(self.inputExp, darkIm)
        medianAfter = np.median(self.inputExp.image[self.amp.getBBox()].getArray())
        self.assertLess(medianAfter, medianBefore)

    def test_flatCorrection(self):
        r"""Test the flat correction.

        The image median should increase (divide by < 1).
        """
        flatIm = isrMock.FlatMock().run()
        stdBefore = np.median(self.inputExp.image[self.amp.getBBox()].getArray())
        self.task.flatCorrection(self.inputExp, flatIm)
        stdAfter = np.median(self.inputExp.image[self.amp.getBBox()].getArray())
        self.assertGreater(stdAfter, stdBefore)

    def test_saturationDetection(self):
        r"""Test the saturation level detection/masking.
        """
        self.amp.setSaturation(1000.0)
        self.task.saturationDetection(self.inputExp, self.amp)

        bitValue = self.inputExp.getMask().getPlaneBitMask("SAT")
        nMaskL1 = len(self.inputExp.getMask().getArray() & bitValue)

        self.amp.setSaturation(25.0)
        self.task.saturationDetection(self.inputExp, self.amp)
        nMaskL2 = len(self.inputExp.getMask().getArray() & bitValue)
        self.assertLessEqual(nMaskL1, nMaskL2)

    def test_measureBackground(self):
        r"""Test that the background measurement runs successfully.
        """
        self.config.qa.flatness.meshX = 20
        self.config.qa.flatness.meshY = 20

        for qaConfig in (None, self.config.qa):
            with self.subTest(qaConfig=qaConfig):
                self.task.measureBackground(self.inputExp, qaConfig)
                assert True

    def test_flatContext(self):
        r"""Test that the flat context manager runs successfully.
        """
        darkExp = isrMock.DarkMock().run()
        flatExp = isrMock.FlatMock().run()

        for dark in (None, darkExp):
            with self.subTest(dark=dark):
                with self.task.flatContext(self.inputExp, flatExp, dark):
                    assert True


class IsrTaskUnTrimmedTestCases(lsst.utils.tests.TestCase):
    r"""Test IsrTask methods using untrimmed raw data.
    """
    def setUp(self):
        self.config = IsrTaskConfig()
        self.config.qa = IsrQaConfig()
        self.task = IsrTask(config=self.config)

        self.mockConfig = isrMock.IsrMockConfig()
        self.mockConfig.isTrimmed = False
        self.dataRef = isrMock.DataRefMock(config=self.mockConfig)
        self.camera = isrMock.IsrMock(config=self.mockConfig).getCamera()

        self.inputExp = isrMock.RawMock(config=self.mockConfig).run()
        self.amp = self.inputExp.getDetector()[0]
        self.mi = self.inputExp.getMaskedImage()

    def tearDown(self):
        pass

    def test_overscanCorrection(self):
        r"""Test that overscan subtraction works.

        This should reduce the image variance with a full fit.
        The default fitType of MEDIAN will reduce the median value.

        This needs to operate on a RawMock() to have overscan data to use.
        """
        medianBefore = np.median(self.inputExp.image[self.amp.getRawDataBBox()].getArray())
        oscanResults = self.task.overscanCorrection(self.inputExp, self.amp)
        assert oscanResults
        assert oscanResults.imageFit
        assert oscanResults.overscanFit
        assert oscanResults.overscanImage

        medianAfter = np.median(self.inputExp.image[self.amp.getRawDataBBox()].getArray())

        self.assertLess(medianAfter, medianBefore)

    def test_runDataRef(self):
        r"""Test ISR operates on a dataRef correctly.
        """
        self.config.doLinearize = False
        self.config.doWrite = False
        self.task = IsrTask(config=self.config)

        assert self.task.runDataRef(self.dataRef)

    def test_run(self):
        r"""Test ISR operates across a variety of configuration options.

        This naively flips all options on and off in order to keep the
        number of tests performed small.

        Output results should be tested more precisely by the
        individual function tests.
        """
        self.config.qa.flatness.meshX = 20
        self.config.qa.flatness.meshY = 20
        self.config.doWrite = False
        self.config.doLinearize = False
        for doTheThing in (True, False):
            with self.subTest(doTheThing=doTheThing):
                self.config.doConvertIntToFloat = doTheThing
                self.config.doSaturation = doTheThing
                self.config.doSuspect = doTheThing
                self.config.doSetBadRegions = doTheThing
                self.config.doOverscan = doTheThing
                self.config.doBias = doTheThing
                self.config.doVariance = doTheThing
                self.config.doWidenSaturationTrails = doTheThing
                self.config.doBrighterFatter = doTheThing
                self.config.doDefect = doTheThing
                self.config.doSaturationInterpolation = doTheThing
                self.config.doDark = doTheThing
                self.config.doStrayLight = doTheThing
                self.config.doFlat = doTheThing
                self.config.doFringe = doTheThing
                self.config.doAddDistortionModel = doTheThing
                self.config.doMeasureBackground = doTheThing
                self.config.doVignette = doTheThing
                self.config.doAttachTransmissionCurve = doTheThing
                self.config.doUseOpticsTransmission = doTheThing
                self.config.doUseFilterTransmission = doTheThing
                self.config.doUseSensorTransmission = doTheThing
                self.config.doUseAtmosphereTransmission = doTheThing
                self.config.qa.saveStats = doTheThing
                self.config.qa.doThumbnailOss = doTheThing
                self.config.qa.doThumbnailFlattened = doTheThing

                self.config.doApplyGains = not doTheThing
                self.config.doCameraSpecificMasking = doTheThing
                self.config.vignette.doWriteVignettePolygon = doTheThing

                self.task = IsrTask(config=self.config)
                results = self.task.run(self.inputExp,
                                        camera=self.camera,
                                        bias=self.dataRef.get("bias"),
                                        dark=self.dataRef.get("dark"),
                                        flat=self.dataRef.get("flat"),
                                        bfKernel=self.dataRef.get("bfKernel"),
                                        defects=self.dataRef.get("defects"),
                                        fringes=self.dataRef.get("fringes"),
                                        opticsTransmission=self.dataRef.get("transmission_"),
                                        filterTransmission=self.dataRef.get("transmission_"),
                                        sensorTransmission=self.dataRef.get("transmission_"),
                                        atmosphereTransmission=self.dataRef.get("transmission_")
                                        )
                assert results
                assert results.exposure

    def test_failCases(self):
        r"""Test ISR operates across a variety of configuration options.

        This naively flips all options on and off in order to keep the
        number of tests performed small.

        Output results should be tested more precisely by the
        individual function tests.
        """
        self.config.qa.flatness.meshX = 20
        self.config.qa.flatness.meshY = 20
        self.config.doWrite = False
        self.config.doLinearize = False

        self.config.doConvertIntToFloat = True
        self.config.doSaturation = True
        self.config.doSuspect = True
        self.config.doSetBadRegions = True
        self.config.doOverscan = True
        self.config.doBias = True
        self.config.doVariance = True
        self.config.doWidenSaturationTrails = True
        self.config.doBrighterFatter = True
        self.config.doDefect = True
        self.config.doSaturationInterpolation = True
        self.config.doDark = True
        self.config.doStrayLight = True
        self.config.doFlat = True
        self.config.doFringe = True
        self.config.doAddDistortionModel = True
        self.config.doMeasureBackground = True
        self.config.doVignette = True
        self.config.doAttachTransmissionCurve = True
        self.config.doUseOpticsTransmission = True
        self.config.doUseFilterTransmission = True
        self.config.doUseSensorTransmission = True
        self.config.doUseAtmosphereTransmission = True
        self.config.qa.saveStats = True
        self.config.qa.doThumbnailOss = True
        self.config.qa.doThumbnailFlattened = True

        self.config.doApplyGains = False
        self.config.doCameraSpecificMasking = True
        self.config.vignette.doWriteVignettePolygon = True

        # This breaks it
        self.config.doCrosstalk = True

        with self.assertRaises(RuntimeError):
            self.task = IsrTask(config=self.config)
            results = self.task.run(self.inputExp,
                                    camera=self.camera,
                                    bias=self.dataRef.get("bias"),
                                    dark=self.dataRef.get("dark"),
                                    flat=self.dataRef.get("flat"),
                                    bfKernel=self.dataRef.get("bfKernel"),
                                    defects=self.dataRef.get("defects"),
                                    fringes=self.dataRef.get("fringes"),
                                    opticsTransmission=self.dataRef.get("transmission_"),
                                    filterTransmission=self.dataRef.get("transmission_"),
                                    sensorTransmission=self.dataRef.get("transmission_"),
                                    atmosphereTransmission=self.dataRef.get("transmission_"),
                                    )

            assert results

    def test_runCases(self):
        r"""Test special cases of configuration to confirm expected behavior.

        Output results should be tested more precisely by the
        individual function tests.
        """

        self.config.qa.flatness.meshX = 20
        self.config.qa.flatness.meshY = 20
        self.config.doWrite = False
        self.config.doLinearize = False
        self.config.doConvertIntToFloat = True
        self.config.doOverscan = True
        self.config.overscanFitType = "POLY"
        self.config.overscanOrder = 1
        self.config.doBias = False
        self.config.doVariance = True
        self.config.doBrighterFatter = True
        self.config.doDark = True
        self.config.doStrayLight = True
        self.config.doFlat = True
        self.config.doFringe = True
        self.config.doCrosstalk = False
        self.config.doAddDistortionModel = True
        self.config.doMeasureBackground = True
        self.config.doVignette = True
        self.config.doAttachTransmissionCurve = True
        self.config.doUseOpticsTransmission = True
        self.config.doUseFilterTransmission = True
        self.config.doUseSensorTransmission = True
        self.config.doUseAtmosphereTransmission = True
        self.config.qa.saveStats = True
        self.config.qa.doThumbnailOss = True
        self.config.qa.doThumbnailFlattened = True
        self.config.doCameraSpecificMasking = True
        self.config.vignette.doWriteVignettePolygon = True

        self.config.doSaturation = False
        self.config.doWidenSaturationTrails = False
        self.config.doSaturationInterpolation = False
        self.config.doSuspect = False
        self.config.doSetBadRegions = False
        self.config.doDefect = False

        self.config.doBrighterFatter = False
        for testIteration in range(0, 7):
            with self.subTest(testIteration=testIteration):
                if testIteration >= 0:
                    self.config.saturation = 1000.0
                    self.config.doSaturation = True
                elif testIteration >= 1:
                    self.config.doWidenSaturationTrails = True
                elif testIteration >= 2:
                    self.config.doSaturationInterpolation = True
                elif testIteration >= 3:
                    self.config.doSuspect = True
                elif testIteration >= 4:
                    self.config.numEdgeSuspect = 5
                elif testIteration >= 5:
                    self.config.doDefect = True
                elif testIteration >= 6:
                    self.config.doSetBadRegions = True

                self.task = IsrTask(config=self.config)

                results = self.task.run(self.dataRef.get("raw"),
                                        camera=self.camera,
                                        bias=self.dataRef.get("bias"),
                                        dark=self.dataRef.get("dark"),
                                        flat=self.dataRef.get("flat"),
                                        bfKernel=self.dataRef.get("bfKernel"),
                                        defects=self.dataRef.get("defects"),
                                        fringes=self.dataRef.get("fringes"),
                                        opticsTransmission=self.dataRef.get("transmission_"),
                                        filterTransmission=self.dataRef.get("transmission_"),
                                        sensorTransmission=self.dataRef.get("transmission_"),
                                        atmosphereTransmission=self.dataRef.get("transmission_")
                                        )
                results.exposure.writeFits("../czw%d.fits" % (testIteration))
                assert results
                assert results.exposure
                print("%d: %d %d %d %d" % (testIteration,
                                           maskCountPixels(results.exposure, "SAT"),
                                           maskCountPixels(results.exposure, "INTRP"),
                                           maskCountPixels(results.exposure, "SUSPECT"),
                                           maskCountPixels(results.exposure, "BAD")))

                if testIteration >= 0:
                    self.assertGreaterEqual(maskCountPixels(results.exposure, "SAT"), 0)
                if testIteration >= 1:
                    self.assertGreaterEqual(maskCountPixels(results.exposure, "SAT"), 0)
                if testIteration >= 2:
                    self.assertGreaterEqual(maskCountPixels(results.exposure, "SAT"), 0)
                    self.assertGreaterEqual(maskCountPixels(results.exposure, "INTRP"), 0)
                if testIteration >= 3:
                    self.assertGreaterEqual(maskCountPixels(results.exposure, "SAT"), 0)
                    self.assertGreaterEqual(maskCountPixels(results.exposure, "INTRP"), 0)
                    self.assertGreaterEqual(maskCountPixels(results.exposure, "SUSPECT"), 0)
                elif testIteration >= 4:
                    pass
                elif testIteration >= 5:
                    pass
                elif testIteration >= 6:
                    self.assertGreaterEqual(maskCountPixels(results.exposure, "BAD"), 0)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
