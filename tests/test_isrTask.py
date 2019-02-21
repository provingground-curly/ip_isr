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
# import tempfile

import lsst.utils.tests
from lsst.ip.isr.isrTask import (IsrTask, IsrTaskConfig)
from lsst.ip.isr.isrQa import IsrQaConfig
import lsst.ip.isr.isrMock as isrMock


class IsrTaskTestCases(lsst.utils.tests.TestCase):

    def setUp(self):
        self.config = IsrTaskConfig()
        self.config.qa = IsrQaConfig()
        self.task = IsrTask(config=self.config)
        self.dataRef = isrMock.DataRefMock()
        self.camera = isrMock.IsrMock().getCamera()

        self.inputExp = isrMock.TrimmedRawMock().mock()
        self.amp = self.inputExp.getDetector()[0]
        self.mi = self.inputExp.getMaskedImage()

    def tearDown(self):
        pass

    def test_readIsrData(self):
        for transmission in (True, False):
            with self.subTest(transmission=transmission):
                self.config.doAttachTransmissionCurve = transmission
                self.task = IsrTask(config=self.config)
                results = self.task.readIsrData(self.dataRef, self.inputExp)
                assert results

    def test_getIsrExposure(self):
        pass

    def test_ensureExposure(self):
        assert self.task.ensureExposure(self.inputExp, self.camera, 0)

    def test_convertItoF(self):
        assert self.task.convertIntToFloat(self.inputExp)

    def test_overscanCorrection(self):
        assert self.task.overscanCorrection(self.inputExp, self.amp)

    def test_updateVariance(self):
        self.task.updateVariance(self.inputExp, self.amp)
        assert True

    def test_darkCorrection(self):
        darkIm = isrMock.DarkMock().mock()

        self.task.darkCorrection(self.inputExp, darkIm)
        assert True

        darkIm.getInfo().setVisitInfo(None)

        self.task.darkCorrection(self.inputExp, darkIm)
        assert True

    def test_flatCorrection(self):
        flatIm = isrMock.FlatMock().mock()
        self.task.flatCorrection(self.inputExp, flatIm)
        assert True

    def test_saturationDetection(self):
        self.task.saturationDetection(self.inputExp, self.amp)
        assert True

        self.amp.setSaturation(25.0)
        self.task.saturationDetection(self.inputExp, self.amp)
        assert True

    def test_measureBackground(self):
        self.config.qa.flatness.meshX = 20
        self.config.qa.flatness.meshY = 20

        for qaConfig in (None, self.config.qa):
            with self.subTest(qaConfig=qaConfig):
                self.task.measureBackground(self.inputExp, qaConfig)
                assert True

    def test_flatContext(self):
        darkExp = isrMock.DarkMock().mock()
        flatExp = isrMock.FlatMock().mock()

        for dark in (None, darkExp):
            with self.subTest(dark=dark):
                with self.task.flatContext(self.inputExp, flatExp, dark):
                    assert True

    def testIsrWrapperTask(self):
        #        outPath = tempfile.mkdtemp() if outputName is None else "{}-isrTask".format(outputName)
        #        dataId = dict(visit=3)
        #        dataIdStrList = ["%s=%s" % (key, val) for key, val in dataId.items()]
        #        fullResult = IsrWrapperTask.parseAndRun(
        #            args=[inputDir, "--output", outPath, "--clobber-config", "--doraise",
        #                  "--id"] + dataIdStrList + ["--config", "isr.doAddDistortionModel=False"],
        #            doReturnResults=True
        #        )
        #        assert fulResult
        assert True


# class IsrTaskUnTrimmedTestCases(lsst.utils.tests.TestCase):

#     def setUp(self):
#         self.config = IsrTaskConfig()
#         self.config.qa = IsrQaConfig()
#         self.task = IsrTask(config=self.config)
#         self.dataRef = isrMock.DataRefMock()
#         self.camera = isrMock.IsrMock().getCamera()

#         self.inputExp = isrMock.RawMock().mock()
#         self.amp = self.inputExp.getDetector()[0]
#         self.mi = self.inputExp.getMaskedImage()

#     def tearDown(self):
#         pass

#     def test_runDataRef(self):
#         self.config.doLinearize = False
#         self.task = IsrTask(config=self.config)

#         assert self.task.runDataRef(self.dataRef)

#     def test_run(self):
#         self.config.qa.flatness.meshX = 20
#         self.config.qa.flatness.meshY = 20

#         self.config.doLinearize = False
#         for doTheThing in (True, False):
#             with self.subTest(doTheThing=doTheThing):
#                 self.config.doConvertIntToFloat = doTheThing
#                 self.config.doSaturation = doTheThing
#                 self.config.doSuspect = doTheThing
#                 self.config.doSetBadRegions = doTheThing
#                 self.config.doOverscan = doTheThing
#                 self.config.doBias = doTheThing
#                 self.config.doVariance = doTheThing
#                 self.config.doWidenSaturationTrails = doTheThing
#                 self.config.doBrighterFatter = doTheThing
#                 self.config.doDefect = doTheThing
#                 self.config.doSaturationInterpolation = doTheThing
#                 self.config.doDark = doTheThing
#                 self.config.doStrayLight = doTheThing
#                 self.config.doFlat = doTheThing
#                 self.config.doFringe = doTheThing
#                 self.config.doAddDistortionModel = doTheThing
#                 self.config.doMeasureBackground = doTheThing
#                 self.config.doVignette = doTheThing
#                 self.config.doAttachTransmissionCurve = doTheThing
#                 self.config.doUseOpticsTransmission = doTheThing
#                 self.config.doUseFilterTransmission = doTheThing
#                 self.config.doUseSensorTransmission = doTheThing
#                 self.config.doUseAtmosphereTransmission = doTheThing
#                 self.config.qa.saveStats = doTheThing
#                 self.config.qa.doThumbnailOss = doTheThing
#                 self.config.qa.doThumbnailFlattened = doTheThing

# #                self.config.doCrosstalk = doTheThing
# #                self.config.doCrosstalkBeforeAssemble = doTheThing
#                 self.config.doApplyGains = not doTheThing
#                 self.config.doCameraSpecificMasking = doTheThing
#                 self.config.vignette.doWriteVignettePolygon = doTheThing

#                 self.task = IsrTask(config=self.config)
#                 #  self.dataRef.get("defects"),
#                 results = self.task.run(self.inputExp,
#                                         camera=self.camera,
#                                         bias=self.dataRef.get("bias"),
#                                         dark=self.dataRef.get("dark"),
#                                         flat=self.dataRef.get("flat"),
#                                         bfKernel=self.dataRef.get("bfKernel"),
#                                         defects=self.dataRef.get("defects"),
#                                         fringes=self.dataRef.get("fringes"),
#                                         opticsTransmission=self.dataRef.get("transmission_"),
#                                         filterTransmission=self.dataRef.get("transmission_"),
#                                         sensorTransmission=self.dataRef.get("transmission_"),
#                                         atmosphereTransmission=self.dataRef.get("transmission_")
#                                         )
#                 assert results


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
