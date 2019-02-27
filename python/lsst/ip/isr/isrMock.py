# This file is part of ip_isr.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import tempfile

import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom.utils as afwUtils
import lsst.afw.cameraGeom.testUtils as afwTestUtils
from lsst.meas.algorithms import Defect
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase


__all__ = ["IsrMockConfig", "IsrMock", "RawMock", "TrimmedRawMock", "RawDictMock",
           "CalibratedRawMock", "MasterMock",
           "BiasMock", "DarkMock", "FlatMock", "FringeMock", "UntrimmedFringeMock",
           "BfKernelMock", "DefectMock", "CrosstalkCoeffMock", "TransmissionMock",
           "DataRefMock"]


class IsrMockConfig(pexConfig.Config):
    r"""Configuration parameters for isrMock.

    These parameters produce generic fixed position signals from
    various sources, and combine them in a way that matches how those
    signals are combined to create real data.  The camera used is the
    test camera defined by the afwUtils code.
    """
    # Detector parameters.  "Exposure" parameters.
    isLsstLike = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="If True, products have one raw image per amplifier, otherwise, one raw image per detector",
    )
    isTrimmed = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="If True, amplifiers have been trimmed and mosaicked to remove regions outside the data BBox",
    )
    detectorIndex = pexConfig.Field(
        dtype=int,
        default=20,
        doc="Index for the detector to use.  The default value uses a standard 2x4 array of amps.",
    )
    rngSeed = pexConfig.Field(
        dtype=int,
        default=20000913,
        doc="Seed for random number generator used to add noise.",
    )
    gain = pexConfig.Field(
        dtype=float,
        default=1.0,
        doc="Gain for simulated data.",
    )
    expTime = pexConfig.Field(
        dtype=float,
        default=5.0,
        doc="Exposure time for simulated data.",
    )

    # Signal parameters
    skyLevel = pexConfig.Field(
        dtype=float,
        default=1000.0,
        doc="Background contribution to be generated from 'the sky'.",
    )
    skySigma = pexConfig.Field(
        dtype=float,
        default=32.0,
        doc="Background noise contribution to be generated from 'the sky'.",
    )
    sourceFlux = pexConfig.Field(
        dtype=float,
        default=45000.0,
        doc="Flux level (in DN) of a simulated 'astronomical source'.",
    )
    overscanScale = pexConfig.Field(
        dtype=float,
        default=100.0,
        doc="Amplitude of the ramp function to add to overscan data.",
    )
    biasLevel = pexConfig.Field(
        dtype=float,
        default=8000.0,
        doc="Background contribution to be generated from the bias offset.",
    )
    biasSigma = pexConfig.Field(
        dtype=float,
        default=90.0,
        doc="Background noise contribution to be generated from the bias.",
    )
    darkRate = pexConfig.Field(
        dtype=float,
        default=5.0,
        doc="Background level contribution to be generated from dark current.",
    )
    darkTime = pexConfig.Field(
        dtype=float,
        default=5.0,
        doc="Exposure time for the dark current contribution.",
    )
    flatScale = pexConfig.Field(
        dtype=float,
        default=320.0,
        doc="Scale factor to apply to the flat dome.  Should not need to be so big.",
    )
    fringeScale = pexConfig.Field(
        dtype=float,
        default=200.0,
        doc="Scale factor to apply to the fringe ripple.",
    )

    # Inclusion parameters
    doSky = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Apply 'sky' signal to output image.",
    )
    doSource = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Add simulated source to output image.",
    )
    doCrosstalk = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Apply simulated crosstalk to output image.  Not currently useful (det.hasCrosstalk()=False)",
    )
    doOverscan = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="If untrimmed, add overscan ramp to overscan and regions.",
    )
    doBias = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Add bias signal to data.",
    )
    doDark = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Add dark signal to data.",
    )
    doFlat = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Add flat signal to data.",
    )
    doFringe = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Add fringe signal to data.",
    )

    # Datasets to create and return instead of generating an image.
    doTransmissionCurve = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Return a simulated transmission curve.",
    )
    doDefects = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Return a simulated defect list.",
    )
    doBrighterFatter = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Return a simulated Brighter-Fatter kernel.",
    )
    doCrosstalkCoeffs = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Return the matrix of crosstalk coefficients.",
    )
    doDataRef = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Return a simulated gen2 butler dataRef.",
    )
    doGenerateImage = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Generate an output image?",
    )
    doGenerateData = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Generate some other data structure?",
    )
    doGenerateDict = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Generate a dict of exposure amplifiers instead of an afwImage.Exposure.",
    )


class IsrMock(pipeBase.Task):
    r"""Class to generate consistent mock images for ISR testing.

    ISR testing currently relies on one-off fake images that do not
    accurately mimic the full set of detector effects.  This class
    uses the test camera/detector/amplifier structure defined in
    `lsst.afw.cameraGeom.testUtils` to avoid making the test data
    dependent on any of the actual obs package formats.
    """
    ConfigClass = IsrMockConfig
    _DefaultName = "isrMock"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rng = np.random.RandomState(self.config.rngSeed)
        self.crosstalkCoeffs = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        self.bfKernel = np.array([[1., 4., 7., 4., 1.],
                                  [4., 16., 26., 16., 4.],
                                  [7., 26., 41., 26., 7.],
                                  [4., 16., 26., 16., 4.],
                                  [1., 4., 7., 4., 1.]]) / 273.0

    def run(self):
        r"""Generate a mock ISR product, and return it.

        Returns
        -------
        image : `lsst.afw.image.Exposure`
            Simulated ISR image with signals added.
        dataProduct :
            Simulated ISR data products.
        None :
            Returned if no valid configuration was found.
        """
        if self.config.doGenerateImage:
            return self.makeImage()
        elif self.config.doGenerateData:
            return self.makeData()
        else:
            return None

    def makeData(self):
        r"""Generate simulated ISR data.

        Currently, only the class defined crosstalk coefficient
        matrix, brighter-fatter kernel, a constant unity transmission
        curve, or a simple single-entry defect list can be generated.

        Returns
        -------
        dataProduct :
            Simulated ISR data product.
        """
        if self.config.doBrighterFatter is True:
            return self.makeBfKernel()
        elif self.config.doDefects is True:
            return self.makeDefectList()
        elif self.config.doTransmissionCurve is True:
            return self.makeTransmissionCurve()
        elif self.config.doCrosstalkCoeffs is True:
            return self.crosstalkCoeffs
        else:
            return None

    def makeBfKernel(self):
        r"""Generate a simple Gaussian brighter-fatter kernel.

        Returns
        -------
        kernel : `numpy.ndarray`
            Simulated BF kernel.
        """
        return self.bfKernel

    def makeDefectList(self):
        r"""Generate a simple single-entry defect list.

        Returns
        -------
        defectList : `list` of Defects`
            Simulated defect list
        """
        defectList = []
        bbox = afwGeom.BoxI(afwGeom.PointI(0, 0),
                            afwGeom.ExtentI(40, 50))
        defectList.append(Defect(bbox))
        return defectList

    def makeCrosstalkCoeff(self):
        r"""Generate the simulated crosstalk coefficients.

        Returns
        -------
        coeffs : `numpy.ndarray`
            Simulated crosstalk coefficients.
        """

        return self.crosstalkCoeffs

    def makeTransmissionCurve(self):
        r"""Generate a simulated flat transmission curve.

        Returns
        -------
        transmission : `lsst.afw.image.TransmissionCurve`
            Simulated transmission curve.
        """

        return afwImage.TransmissionCurve.makeIdentity()

    def makeImage(self):
        r"""Generate a simulated ISR image.

        Returns
        -------
        exposure : `lsst.afw.image.Exposure` or `dict`
            Simulated ISR image data.

        Notes
        -----
        This method currently constructs a "raw" data image by:
            * Generating a simulated sky with noise
            * Adding a single Gaussian "star"
            * Adding a bias offset (and noise)
            * Adding a overscan gradient parallel to the pixel y-axis
            * Adding dark current (and noise)
            * Multiplying the frame by the simulated flat
            * Adding the fringe signal
            * Simulating crosstalk by adding a scaled version of the
              source amplifier to the target amplifiers.

        The exposure with image data constructed this way is in one of
        three formats.
            * A single image, with overscan and prescan regions retained
            * A single image, with overscan and prescan regions trimmed
            * A `dict`, containing the amplifer data indexed by the
              amplifier name.

        To truly match the expected construction, crosstalk should be
        added prior to the bias, nonlinearity prior to the dark, and
        the brighter-fatter convolution prior to the flat.
        """
        exposure = self.getExposure()

        for idx, amp in enumerate(exposure.getDetector()):
            bbox = None
            if self.config.isTrimmed is True:
                bbox = amp.getBBox()
            else:
                bbox = amp.getRawDataBBox()

            ampData = exposure.image[bbox]

            if self.config.doSky is True:
                self.amplifierAddNoise(ampData, self.config.skyLevel, self.config.skySigma)
            if self.config.doSource is True:
                if idx == 0:
                    self.amplifierAddSource(ampData, self.config.sourceFlux)

            if self.config.doFringe is True:
                self.amplifierAddFringe(amp, ampData, self.config.fringeScale)

            if self.config.doFlat is True:
                if ampData.getArray().sum() == 0.0:
                    self.amplifierAddNoise(ampData, 1.0, 0.0)
                self.amplifierAddFlat(amp, ampData, self.config.flatScale)

            # CZW: Brighter fatter?

            if self.config.doDark is True:
                self.amplifierAddNoise(ampData, self.config.darkRate * self.config.darkTime,
                                       np.sqrt(self.config.darkRate * self.config.darkTime))

        # Non-linearity/CTE

        if self.config.doCrosstalk is True:
            for idxS, ampS in enumerate(exposure.getDetector()):
                for idxT, ampT in enumerate(exposure.getDetector()):
                    if self.config.isTrimmed is True:
                        ampDataS = exposure.image[ampS.getBBox()]
                        ampDataT = exposure.image[ampT.getBBox()]
                    else:
                        ampDataS = exposure.image[ampS.getRawDataBBox()]
                        ampDataT = exposure.image[ampT.getRawDataBBox()]
                    self.amplifierAddCT(ampDataS, ampDataT, self.crosstalkCoeffs[idxT][idxS])

        for idx, amp in enumerate(exposure.getDetector()):
            bbox = None
            if self.config.isTrimmed is True:
                bbox = amp.getBBox()
            else:
                bbox = amp.getRawDataBBox()

            ampData = exposure.image[bbox]

            if self.config.doBias is True:
                self.amplifierAddNoise(ampData, self.config.biasLevel, self.config.biasSigma)

            if self.config.doOverscan is True:
                oscanBBox = amp.getRawHorizontalOverscanBBox()
                oscanData = exposure.image[oscanBBox]
                self.amplifierAddNoise(oscanData, self.config.biasLevel, self.config.biasSigma)

                self.amplifierAddYGradient(ampData, -1.0 * self.config.overscanScale,
                                           1.0 * self.config.overscanScale)
                self.amplifierAddYGradient(oscanData, -1.0 * self.config.overscanScale,
                                           1.0 * self.config.overscanScale)

        if self.config.doGenerateDict is True:
            expDict = dict()
            for amp in exposure.getDetector():
                expDict[amp.getName()] = exposure
            return expDict
        else:
            return exposure

    # afw primatives to construct the image structure
    def getCamera(self):
        r"""Construct a test camera object.

        Returns
        -------
        camera : `lsst.afw.cameraGeom.camera`
            Test camera.
        """
        cameraWrapper = afwTestUtils.CameraWrapper(self.config.isLsstLike)
        camera = cameraWrapper.camera
        return camera

    def getExposure(self):
        r"""Construct a test exposure.

        The test exposure has a simple WCS set, as well as a list of
        unlikely header keywords that can be removed during ISR
        processing to exercise that code.

        Returns
        -------
        exposure : `lsst.afw.exposure.Exposure`
            Construct exposure containing masked image of the
            appropriate size.
        """
        camera = self.getCamera()
        detector = camera[self.config.detectorIndex]
        image = afwUtils.makeImageFromCcd(detector,
                                          isTrimmed=self.config.isTrimmed,
                                          showAmpGain=False,
                                          rcMarkSize=0,
                                          binSize=1,
                                          imageFactory=afwImage.ImageF)

        var = afwImage.ImageF(image.getDimensions())
        mask = afwImage.Mask(image.getDimensions())
        image.assign(0.0)

        maskedImage = afwImage.makeMaskedImage(image, mask, var)
        exposure = afwImage.makeExposure(maskedImage)
        exposure.setDetector(detector)
        exposure.setWcs(self.getWcs())

        visitInfo = afwImage.VisitInfo(exposureTime=1.0, darkTime=1.0)
        exposure.getInfo().setVisitInfo(visitInfo)

        metadata = exposure.getMetadata()
        metadata.add("SHEEP", 7.3, "number of sheep on farm")
        metadata.add("MONKEYS", 155, "monkeys per tree")
        metadata.add("VAMPIRES", 4, "I just like vampires.")

        for amp in exposure.getDetector():
            amp.setLinearityCoeffs((0., 1., 0., 0.))
            amp.setLinearityType("Polynomial")
            amp.setGain(self.config.gain)

        exposure.getImage().getArray()[:] = np.zeros(exposure.getImage().getDimensions()).transpose()
        exposure.getMask().getArray()[:] = np.zeros(exposure.getMask().getDimensions()).transpose()
        exposure.getVariance().getArray()[:] = np.zeros(exposure.getVariance().getDimensions()).transpose()

        return exposure

    def getWcs(self):
        r"""Construct a dummy WCS object.

        Taken from the deprecated ip_isr/examples/exampleUtils.py.

        This is also almost certainly wrong, given the distortion and
        pixel scale listed in the afwTestUtils camera definition.  I
        also suspect this isn't neede for any ISR test, so it may be
        dropped.

        Returns
        -------
        wcs : `lsst.afw.geom.SkyWcs`
            Test WCS transform.
        """
        return afwGeom.makeSkyWcs(crpix=afwGeom.Point2D(0.0, 0.0),
                                  crval=afwGeom.SpherePoint(45.0, 45.0, afwGeom.degrees),
                                  cdMatrix=afwGeom.makeCdMatrix(scale=1.0*afwGeom.degrees))

    def localCoordToExpCoord(self, amp, x, y):
        r"""Convert between a local amplifier coordinate and the full
        exposure coordinate.

        Parameters
        ----------
        amp : `lsst.afw.image.ImageF`
            Amplifier image to use for conversions.
        x : int
            X-coordinate of the point to transform.
        y : int
            Y-coordinate of the point to transform.

        Returns
        -------
        u : int
            Transformed x-coordinate.
        v : int
            Transformed y-coordinate.

        Notes
        -----
        The output is transposed intentionally here, to match the
        internal transpose between numpy and afw.image coordinates.
        """
        if self.config.isTrimmed:
            u = x + amp.getBBox().getBeginX()
            v = y + amp.getBBox().getBeginY()
        else:
            u = x + amp.getRawDataBBox().getBeginX()
            v = y + amp.getRawDataBBox().getBeginY()

        return (v, u)

    # Simple data values.
    def amplifierAddNoise(self, ampData, mean, sigma):
        r"""Add Gaussian noise to an amplifier's image data.

         This method operates in the amplifier coordinate frame.

        Parameters
        ----------
        ampData : `lsst.afw.image.ImageF`
            Amplifier image to operate on.
        mean : float
            Mean value of the Gaussian noise.
        sigma : float
            Sigma of the Gaussian noise.
        """
        ampArr = ampData.getArray()
        ampArr[:] = ampArr[:] + self.rng.normal(mean, sigma,
                                                size=ampData.getDimensions()).transpose()

    def amplifierAddYGradient(self, ampData, start, end):
        r"""Add a y-axis linear gradient to an amplifier's image data.

         This method operates in the amplifier coordinate frame.

        Parameters
        ----------
        ampData : `lsst.afw.image.ImageF`
            Amplifier image to operate on.
        start : float
            Start value of the gradient (at y=0).
        end : float
            End value of the gradient (at y=ymax).
        """
        nPixY = ampData.getDimensions().getY()
        ampArr = ampData.getArray()
        ampArr[:] = ampArr[:] + (np.interp(range(nPixY), (0, nPixY - 1), (start, end)).reshape(nPixY, 1) +
                                 np.zeros(ampData.getDimensions()).transpose())

    def amplifierAddSource(self, ampData, scale):
        r"""Add a single Gaussian source to an amplifier.

         This method operates in the amplifier coordinate frame.

        Parameters
        ----------
        ampData : `lsst.afw.image.ImageF`
            Amplifier image to operate on.
        scale : float
            Peak flux of the source to add.
        """
        for x in range(0, ampData.getDimensions().getX()):
            for y in range(0, ampData.getDimensions().getY()):
                ampData.getArray()[y][x] = (ampData.getArray()[y][x] +
                                            scale * np.exp(-0.5 * ((x - 50)**2 + (y - 25)**2)**2 / 3.0**2))

    def amplifierAddCT(self, ampDataSource, ampDataTarget, scale):
        r"""Add a scaled copy of an amplifier to another, simulating crosstalk.

         This method operates in the amplifier coordinate frame.

        Parameters
        ----------
        ampData : `lsst.afw.image.ImageF`
            Amplifier image to operate on.
        scale : float
            Peak flux of the source to add.

        Notes
        -----
        This simulates simple crosstalk between amplifiers.
        """
        for x in range(0, ampDataSource.getDimensions().getX()):
            for y in range(0, ampDataSource.getDimensions().getY()):
                ampDataTarget.getArray()[y][x] = (ampDataTarget.getArray()[y][x] +
                                                  scale * ampDataSource.getArray()[y][x])

    # Complex data values
    def amplifierAddFringe(self, amp, ampData, scale):
        r"""Add a fringe-like ripple pattern to an amplifier's image data.

        Parameters
        ----------
        amp : `lsst.afw.ampInfo.AmpInfoRecord`
            Amplifier to operate on.  Needed for amp<->exp coordinate transforms.
        ampData : `lsst.afw.image.ImageF`
            Amplifier image to operate on.
        scale : float
            Peak intensity scaling for the ripple.

        Notes
        -----
        This uses an offset sinc function to generate a ripple
        pattern.  True fringes have much finer structure, but this
        pattern should be visually identifiable.  The (x, y)
        coordinates are in the frame of the amplifier, and (u, v) in
        the frame of the full trimmed image.
        """
        for x in range(0, ampData.getDimensions().getX()):
            for y in range(0, ampData.getDimensions().getY()):
                (u, v) = self.localCoordToExpCoord(amp, x, y)
                ampData.getArray()[y][x] = (ampData.getArray()[y][x] +
                                            scale * np.sinc(((u - 100)/50)**2 + (v / 50)**2))

    def amplifierAddFlat(self, amp, ampData, size):
        r"""Add a flat-like dome pattern to an amplifier's image data.

        Parameters
        ----------
        amp : `lsst.afw.ampInfo.AmpInfoRecord`
            Amplifier to operate on.  Needed for amp<->exp coordinate transforms.
        ampData : `lsst.afw.image.ImageF`
            Amplifier image to operate on.
        size : float
            Size of the dome to use.  A larger value results in a
            smaller center-to-edge fall off.

        Notes
        -----
        This uses a centered 2-d Gaussian to simulate an illumination
        pattern that falls off towards the edge of the detector.  The
        e(x, y) coordinates are in the frame of the amplifier, and (u, v)
        in the frame of the full trimmed image.
        """
        for x in range(0, ampData.getDimensions().getX()):
            for y in range(0, ampData.getDimensions().getY()):
                (u, v) = self.localCoordToExpCoord(amp, x, y)
                f = np.exp(-0.5 * ((u - 100)**2 + (v - 100)**2) / size**2)
                ampData.getArray()[y][x] = (ampData.getArray()[y][x] * f)


class RawMock(IsrMock):
    r"""Generate a raw exposure suitable for ISR.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config.isTrimmed = False
        self.config.doGenerateImage = True
        self.config.doGenerateDict = False
        self.config.doOverscan = True
        self.config.doSky = True
        self.config.doSource = True
        self.config.doCrosstalk = False
        self.config.doBias = True
        self.config.doDark = True


class TrimmedRawMock(RawMock):
    r"""Generate a trimmed raw exposure.
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.config.isTrimmed = True
        self.config.doOverscan = False


class CalibratedRawMock(RawMock):
    r"""Generate a trimmed raw exposure.
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.config.isTrimmed = True
        self.config.doOverscan = False
        self.config.doBias = True
        self.config.doDark = False
        self.config.doFlat = False
        self.config.doFringe = False
        self.config.doCrosstalk = True
        self.config.biasLevel = 0.0
        self.config.biasSigma = 0.0
        self.config.skySigma = 100.0


class RawDictMock(RawMock):
    r"""Generate a raw exposure dict suitable for ISR.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config.doGenerateDict = True


class MasterMock(IsrMock):
    r"""Parent class for those that make master calibrations.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config.isTrimmed = True
        self.config.doGenerateImage = True
        self.config.doOverscan = False
        self.config.doSky = False
        self.config.doSource = False
        self.config.doCrosstalk = False

        self.config.doBias = False
        self.config.doDark = False
        self.config.doFlat = False
        self.config.doFringe = False


class BiasMock(MasterMock):
    r"""Simulated master bias calibration.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config.doBias = True
        self.config.biasSigma = 30.0


class DarkMock(MasterMock):
    r"""Simulated master dark calibration.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config.doDark = True
        self.config.darkTime = 1.0


class FlatMock(MasterMock):
    r"""Simulated master flat calibration.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config.doFlat = True


class FringeMock(MasterMock):
    r"""Simulated master fringe calibration.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config.doFringe = True


class UntrimmedFringeMock(FringeMock):
    r"""Simulated untrimmed master fringe calibration.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config.isTrimmed = False


class BfKernelMock(IsrMock):
    r"""Simulated brighter-fatter kernel.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.config.doGenerateImage = False
        self.config.doGenerateData = True
        self.config.doBrighterFatter = True
        self.config.doDefects = False
        self.config.doCrosstalkCoeffs = False
        self.config.doTransmissionCurve = False


class DefectMock(IsrMock):
    r"""Simulated defect list.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.config.doGenerateImage = False
        self.config.doGenerateData = True
        self.config.doBrighterFatter = False
        self.config.doDefects = True
        self.config.doCrosstalkCoeffs = False
        self.config.doTransmissionCurve = False


class CrosstalkCoeffMock(IsrMock):
    r"""Simulated crosstalk coefficient matrix.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.config.doGenerateImage = False
        self.config.doGenerateData = True
        self.config.doBrighterFatter = False
        self.config.doDefects = False
        self.config.doCrosstalkCoeffs = True
        self.config.doTransmissionCurve = False


class TransmissionMock(IsrMock):
    r"""Simulated transmission curve.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.config.doGenerateImage = False
        self.config.doGenerateData = True
        self.config.doBrighterFatter = False
        self.config.doDefects = False
        self.config.doCrosstalkCoeffs = False
        self.config.doTransmissionCurve = True


class DataRefMock(object):
    r"""Simulated gen2 butler data ref.

    Currently only supports get and put operations, which are most
    likely to be called for data in ISR processing.

    """
    dataId = "isrMock Fake Data"
    darkval = 2.  # e-/sec
    oscan = 250.  # DN
    gradient = .10
    exptime = 15  # seconds
    darkexptime = 40.  # seconds

    def __init__(self, **kwargs):
        if 'config' in kwargs.keys():
            self.config = kwargs['config']
        else:
            self.config = None

    def get(self, dataType, **kwargs):
        r"""Return an appropriate data product.

        Parameters
        ----------
        dataType : `str`
            Type of data product to return.

        Returns
        -------
        mock : IsrMock.run() result
            The output product.
        """
        if "_filename" in dataType:
            return tempfile.mktemp(), "mock"
        elif 'transmission_' in dataType:
            return TransmissionMock(config=self.config).run()
        elif dataType == 'ccdExposureId':
            return 20090913
        elif dataType == 'camera':
            return IsrMock(config=self.config).getCamera()
        elif dataType == 'raw':
            return RawMock(config=self.config).run()
        elif dataType == 'bias':
            return BiasMock(config=self.config).run()
        elif dataType == 'dark':
            return DarkMock(config=self.config).run()
        elif dataType == 'flat':
            return FlatMock(config=self.config).run()
        elif dataType == 'fringe':
            return FringeMock(config=self.config).run()
        elif dataType == 'defects':
            return DefectMock(config=self.config).run()
        elif dataType == 'bfKernel':
            return BfKernelMock(config=self.config).run()
        elif dataType == 'linearizer':
            return None
        elif dataType == 'crosstalkSources':
            return None
        else:
            return None

    def put(self, exposure, filename):
        r"""Write an exposure to a FITS file.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Image data to write out.
        filename : `str`
            Base name of the output file.
        """
        exposure.writeFits(filename+".fits")
