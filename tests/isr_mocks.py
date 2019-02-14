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
import pdb
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
import lsst.afw.cameraGeom.utils as afwUtils
import lsst.afw.cameraGeom.testUtils as afwTestUtils
from lsst.meas.algorithms import Defect


class IsrMock():
    r"""Class to generate consistent mock images for ISR testing.

    ISR testing currently relies on one-off fake images that do not
    accurately mimic the full set of detector effects.  This class
    uses the test camera/detector/amplifier structure defined in
    `lsst.afw.cameraGeom.testUtils` to avoid making the test data
    dependent on any of the actual obs package formats.

    Parameters
    ----------
    isLsstLike : bool, optional
        If True, make repository products with one raw image per
        amplifier, otherwise, use one raw image per detector.
    isTrimmed : bool, optional
        If True, the amplifers have been trimmed and mosaicked to
        remove regions outside the data bounding box.
    detectorIndex : int, optional
        Index of the detector of the test camera to use.  The default
        value of 20 uses a "standard" 2x4 array of amplifiers.
    rng_seed : int, optional
        Seed for the random number generator used to add noise.
    gain : float, optional
        Detector gain.
    background : float, optional
        Background level for the image to be constructed.
    noise : float, optional
        Noise level to add to the image to be constructed.
    expTime : float, optional
        Exposure time for the simulated image (used for dark current
        calculation).
    biasLevel : float, optional
        Background bias level for the simulated image.
    biasNoise : float, optional
        Scatter in bias level.
    darkRate : float, optional
        Number of counts-per-second added in dark current.
    fringeIntensity : float, optional
        Flux scale factor to apply to the simulated fringe pattern.
    """
    crosstalkKernel = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    def __init__(self, isLsstLike=True, isTrimmed=True, detectorIndex=20,
                 rng_seed=20000913,
                 gain=1.0,
                 background=0.0, noise=50.0,
                 expTime=5.0,
                 biasLevel=8000.0, biasNoise=250.0,
                 darkRate=20.0,
                 fringeIntensity = 0.01,
             ):
        self.isLsstLike = isLsstLike
        self.isTrimmed = isTrimmed
        self.detectorIndex = detectorIndex
        self.rng_seed = rng_seed
        self.gain = gain
        self.background = background
        self.noise = noise
        self.expTime = expTime
        self.biasLevel=biasLevel
        self.biasNoise=biasNoise
        self.darkRate=darkRate
        self.fringeIntensity=fringeIntensity

        self.rng = np.random.RandomState(self.rng_seed)
        self.camera = self.getCamera()

    def getCamera(self):
        r"""Construct a test camera object.

        Returns
        -------
        camera : `lsst.afw.cameraGeom.camera`
            Test camera.
        """
        cameraWrapper = afwTestUtils.CameraWrapper(self.isLsstLike)
        camera = cameraWrapper.camera
        return camera

    def getDetector(self, detectorIndex=20):
        r"""Construct a test detector.

        Parameters
        ----------
        detectorIndex : int, optional
            Index of the detector of the test camera to use.  The default
            value of 20 uses a "standard" 2x4 array of amplifiers.

        Returns
        -------
        detector : `lsst.afw.cameraGeom.detector`
            Test detector.
        """
        detector = self.camera[detectorIndex]
        return detector

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

    def getExposure(self, detectorIndex=20):
        r"""Construct a test exposure.

        Parameters
        ----------
        detectorIndex : int, optional
            Index of the detector of the test camera to use.  The default
            value of 20 uses a "standard" 2x4 array of amplifiers.

        Returns
        -------
        exposure : `lsst.afw.exposure.Exposure`
            Construct exposure containing masked image of the
            appropriate size.
        """
        detector = self.getDetector(detectorIndex)

        image = afwUtils.makeImageFromCcd(detector,
                                          isTrimmed=self.isTrimmed,
                                          showAmpGain=False,
                                          rcMarkSize=0,
                                          binSize=1,
                                          imageFactory=afwImage.ImageF)
        image.assign(0.0)
        var = afwImage.ImageF(image.getDimensions())
        mask = afwImage.Mask(image.getDimensions())

        maskedImage = afwImage.makeMaskedImage(image, mask, var)
        exposure = afwImage.makeExposure(maskedImage)
        exposure.setDetector(detector)
        exposure.setWcs(self.getWcs())
        return exposure

    def expMock(self, detectorIndex=20):
        r"""Return a simple exposure.

        Parameters
        ----------
        detectorIndex : int, optional
            Index of the detector of the test camera to use.  The default
            value of 20 uses a "standard" 2x4 array of amplifiers.

        Returns
        -------
        exposure : `lsst.afw.exposure.Exposure`
            Construct exposure containing masked image of the
            appropriate size.
        """
        exposure = self.getExposure(detectorIndex)
        expData = exposure.image[exposure.getBBox()]

        return exposure

    def ampMock(self, detectorIndex=20):
        r"""Return a simple exposure amplifier-by-amplifier.

        This method also sets the linearity parameters for the
        amplifier to a simple linear polynomial.

        Parameters
        ----------
        detectorIndex : int, optional
            Index of the detector of the test camera to use.  The default
            value of 20 uses a "standard" 2x4 array of amplifiers.

        Returns
        -------
        exposure : `lsst.afw.exposure.Exposure`
            Construct exposure containing masked image of the
            appropriate size.
        """

        exposure = self.getExposure(detectorIndex)

        for amp in exposure.getDetector():
            amp.setLinearityCoeffs((0., 1., 0., 0.))
            amp.setLinearityType("Polynomial")

        return exposure

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
        if self.isTrimmed:
            u = a + amp.getBBox().getBeginX()
            v = b + amp.getBBox().getBeginY()
        else:
            u = a + amp.getRawDataBBox().getBeginX()
            v = b + amp.getRawDataBBox().getBeginY()

        return (v, u)

    def amplifierZero(self, ampData):
        r"""Zero an amplifier's image data.

        Parameters
        ----------
        ampData : `lsst.afw.image.ImageF`
            Amplifier image to operate on.
        """
        ampData.getArray()[:] = np.zeros(ampData.getDimensions()).transpose()

    def amplifierUnity(self, ampData):
        r"""Set an amplifier's image data to 1.0.

        Parameters
        ----------
        ampData : `lsst.afw.image.ImageF`
            Amplifier image to operate on.
        """
        ampData.getArray()[:] = np.ones(ampData.getDimensions()).transpose()

    def amplifierAddNoise(self, ampData, mean, sigma):
        r"""Add Gaussian noise to an amplifier's image data.

        Parameters
        ----------
        ampData : `lsst.afw.image.ImageF`
            Amplifier image to operate on.
        mean : float
            Mean value of the Gaussian noise.
        sigma : float
            Sigma of the Gaussian noise.
        """
        ampData.getArray()[:] = ampData.getArray()[:] + self.rng.normal(mean, sigma,
                                                                        size=ampData.getDimensions()).transpose()

    def amplifierAddYGradient(self, ampData, start, end):
        r"""Add a y-axis linear gradient to an amplifier's image data.

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
        ampData.getArray()[:] = ampData.getArray()[:] + (np.interp(range(nPixY), (0, nPixY - 1), (start, end)).reshape(nPixY,1) +
                                                         np.zeros(ampData.getDimensions()).transpose())

    def amplifierAddFringe(self, ampData, scale):
        r"""Add a fringe-like ripple pattern to an amplifier's image data.

        Parameters
        ----------
        ampData : `lsst.afw.image.ImageF`
            Amplifier image to operate on.
        scale : float
            Peak intensity scaling for the ripple.

        Notes
        -----
        This uses an offset sinc function to generate a ripple
        pattern.  True fringes have much finer structure, but this
        pattern should be visually identifiable.
        """
        for x in range(0, ampData.getDimensions().getX()):
            for y in range(0, ampData.getDimensions().getY()):
                (u, v) = self.localCoordToExpCoord(ampData, x, y)
                ampData.getArray()[y][x] = ampData.getArray()[y][x] + scale * np.sinc( ((u - 100)/50)**2 + (v / 50)**2)

    def amplifierAddFlat(self, ampData, size):
        r"""Add a flat-like dome pattern to an amplifier's image data.

        Parameters
        ----------
        ampData : `lsst.afw.image.ImageF`
            Amplifier image to operate on.
        size : float
            Size of the dome to use.  A larger value results in a
            smaller center-to-edge fall off.

        Notes
        -----
        This uses a centered 2-d Gaussian to simulate an illumination
        pattern that falls off towards the edge of the detector.
        """
        for x in range(0, ampData.getDimensions().getX()):
            for y in range(0, ampData.getDimensions().getY()):
                (u, v) = self.localCoordToExpCoord(ampData, x, y)
                ampData.getArray()[y][x] = ampData.getArray()[y][x] * np.exp(-0.5 * ( (u - 100)**2 + (v - 100)**2 )**2 / size**2)

    def amplifierAddSource(self, ampData, scale):
        r"""Add a single Gaussian source to an amplifier.

        Parameters
        ----------
        ampData : `lsst.afw.image.ImageF`
            Amplifier image to operate on.
        scale : float
            Peak flux of the source to add.
        """
        for x in range(0, ampData.getDimensions().getX()):
            for y in range(0, ampData.getDimensions().getY()):
                ampData.getArray()[y][x] = ampData.getArray()[y][x] + scale * np.exp(-0.5 * ( (x - 50)**2 + (y - 25)**2 )**2 / 3.0**2)

    def amplifierAddCT(self, ampDataSource, ampDataTarget, scale):
        r"""Add a scaled copy of an amplifier to another.

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
                ampDataTarget.getArray()[y][x] = ampDataTarget.getArray()[y][x] + scale * ampDataSource.getArray()[y][x]

    def mock(self):
        r"""Generate appropriate mock data.

        Should be defined by subclasses to allow different products to
        be constructed.
        """
        pass


class RawMock(IsrMock):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("isTrimmed", False)
        super().__init__(*args, **kwargs)

    def mock(self):
        r"""Generate a raw exposure.

        Returns
        -------
        exposure : `lsst.afw.exposure.Exposure`
            Simulated raw exposure.

        Notes
        -----
        The amplifiers of this device should be equal to:
            fringe + flat * ( ( linearity * cte *
                                (raw + overscan + bias + crosstalk) + dark ) X
                               brighterFatter )
        where:
            raw = sky + source
        """
        exposure = self.ampMock()
        exposure.image[exposure.getBBox()] = 0.0

        for idx, amp in enumerate(exposure.getDetector()):
            ampData = exposure.image[amp.getRawDataBBox()]
            overscanData = exposure.image[amp.getRawHorizontalOverscanBBox()]

            self.amplifierAddNoise(ampData, 0.0 * idx, self.noise)
            self.amplifierAddYGradient(ampData, -100, 100)

            if idx == 0:
                self.amplifierAddSource(ampData, 35000)

            self.amplifierAddNoise(overscanData, 0.0, self.noise)
            self.amplifierAddYGradient(overscanData, -100, 100)

        for idxS, ampS in enumerate(exposure.getDetector()):
            for idxT, ampT in enumerate(exposure.getDetector()):
                ampDataS = exposure.image[ampS.getRawDataBBox()]
                ampDataT = exposure.image[ampT.getRawDataBBox()]
                self.amplifierAddCT(ampDataS, ampDataT, self.crosstalkKernel[idxT][idxS])

        return exposure


class TrimmedRawMock(IsrMock):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("isTrimmed", True)
        super().__init__(*args, **kwargs)

    def mock(self):
        r"""Generate a raw exposure.

        Returns
        -------
        exposure : `lsst.afw.exposure.Exposure`
            Simulated trimmed raw exposure.

        Notes
        -----
        The amplifiers of this device should be equal to:
            fringe + flat * ( ( linearity * cte *
                                (raw + bias + crosstalk) + dark ) X
                               brighterFatter )
        where:
            raw = sky + source
        with the overscan subtracted and the amplifiers trimmed.
        """
        exposure = self.expMock()
        exposure.image[exposure.getBBox()] = 0.0

        for idx, amp in enumerate(exposure.getDetector()):
            ampData = exposure.image[amp.getBBox()]

            self.amplifierAddNoise(ampData, 0.0 * idx, self.noise)

            if idx == 0:
                self.amplifierAddSource(ampData, 35000)

        for idxS, ampS in enumerate(exposure.getDetector()):
            for idxT, ampT in enumerate(exposure.getDetector()):
                ampDataS = exposure.image[ampS.getBBox()]
                ampDataT = exposure.image[ampT.getBBox()]
                self.amplifierAddCT(ampDataS, ampDataT, self.crosstalkKernel[idxT][idxS])

        return exposure

class BiasMock(IsrMock):
    def mock(self):
        r"""Generate a master bias correction.

        Returns
        -------
        exposure : `lsst.afw.exposure.Exposure`
            Simulated bias exposure.

        Notes
        -----
        bias = biasLevel +/- sigma(biasNoise)
        """
        exposure = self.expMock()
        exposure.image[exposure.getBBox()] = 0.0

        for amp in exposure.getDetector():
            ampData = exposure.image[amp.getBBox()]

            self.amplifierAddNoise(ampData, self.biasLevel, self.biasNoise)

        return exposure

class DarkMock(IsrMock):
    def mock(self):
        r"""Generate a master dark correction.

        Returns
        -------
        exposure : `lsst.afw.exposure.Exposure`
            Simulated dark exposure.

        Notes
        -----
        dark = darkRate * expTime +/- sqrt(darkRate * expTime)
        """
        exposure = self.expMock()
        exposure.image[exposure.getBBox()] = 0.0

        for amp in exposure.getDetector():
            ampData = exposure.image[amp.getBBox()]

            self.amplifierAddNoise(ampData, self.darkRate * self.expTime,
                                   np.sqrt(self.darkRate * self.expTime))
        return exposure

class FlatMock(IsrMock):
    def mock(self):
        r"""Generate a master flat correction.

        Returns
        -------
        exposure : `lsst.afw.exposure.Exposure`
            Simulated flat exposure.

        Notes
        -----
        flat = dome * (1.0 +/- 0.01)
        """
        exposure = self.expMock()
        exposure.image[exposure.getBBox()] = 0.0

        for amp in exposure.getDetector():
            ampData = exposure.image[amp.getBBox()]

            self.amplifierAddNoise(ampData, 1.0, 0.01)
            self.amplifierAddFlat(ampData, 30000.0)


        return exposure

class FringeMock(IsrMock):
    def mock(self):
        r"""Generate a master fringe correction.

        Returns
        -------
        exposure : `lsst.afw.exposure.Exposure`
            Simulated fringe exposure.

        Notes
        -----
        fringe = sinc + (0.0 +/- 0.01)
        """
        # Sinc functional form 
        exposure = self.expMock()
        exposure.image[exposure.getBBox()] = 0.0

        for amp in exposure.getDetector():
            ampData = exposure.image[amp.getBBox()]

            self.amplifierAddNoise(ampData, 0.0, 0.01)
            self.amplifierAddFringe(ampData, 1000)

        return exposure

class BfKernelMock(IsrMock):
    def mock(self):
        r"""Generate a simple Gaussian brighter-fatter kernel.

        Returns
        -------
        kernel : `numpy.ndarray`
            Simulated BF kernel.
        """
        kernel = np.array([[1., 4., 7., 4., 1.],
                           [4., 16., 26., 16., 4.],
                           [7., 26., 41., 26., 7.],
                           [4., 16., 26., 16., 4.],
                           [1., 4., 7., 4., 1.]])
        kernel = kernel / 273.0
        return kernel

class DefectMock(IsrMock):
    def mock(self):
        r"""Generate a simple defect box.

        Returns
        -------
        defectList : `list` of `Defects`
            Simulated defect list
        """
        defectList = []
        bbox = afwGeom.BoxI(afwGeom.PointI(0, 0),
                            afwGeom.ExtentI(40, 50))
        defectList.append(Defect(bbox))

        return defectList

class TransmissionMock(IsrMock):
    def mock(self):
        r"""Generate a simulated transmission curve.

        Returns
        -------
        transmission : `lsst.afw.image.TransmissionCurve`
            Simulated transmission curve.
        """
        throughputMin = 0.0
        throughputMax = 0.0
        throughput = np.array([0.285, 0.360, 0.517, 0.647, 0.748, 0.818, 0.864, 0.910, 0.936, 0.948, 0.951, 0.958,
                               0.954, 0.956, 0.950, 0.938, 0.929, 0.916, 0.908, 0.899, 0.892, 0.882, 0.863, 0.856,
                               0.829, 0.825, 0.821, 0.780, 0.745, 0.692, 0.609, 0.517, 0.378, 0.209, 0.088, 0.047])
        wavelength = np.array([3600.,  3800., 4000., 4200., 4400., 4600., 4800., 5000., 5200., 5400.,
                               5600.,  5800., 6000., 6200., 6400., 6600., 6800., 7000., 7200., 7400.,
                               7600.,  7800., 8000., 8200., 8400., 8600., 8800., 9000., 9200., 9400.,
                               9600.,  9800., 10000., 10200., 10400., 10600.])


        # I'm not convinced I have this correct.
        # example: https://github.com/lsst/afw/blob/7a2ee2b6ab51af334669bc5ddece2c90ce2c87ca/tests/test_sourceMatch.py
        # ""    https://github.com/lsst/analysis/blob/2614498b8549528a53230aeb48b37dfce248fa86/python/lsst/analysis/cosmos.py
        schema = afwTable.SimpleTable.makeMinimalSchema()
        schema.addField(afwTable.Field["D"]("throughputAtMin", "throughput below minimum wavelength"))
        schema.addField(afwTable.Field["D"]("throughputAtMax", "throughput above maximum wavelength"))
        schema.addField(afwTable.Field["D"]("throughput", "array of known throughput values"))
        schema.addField(afwTable.Field["D"]("wavelengths", "array of known wavelength values"))

        transmission = afwTable.SimpleCatalog(schema)
        transmission['throughputAtMin'][:] = throughputMin
        transmission['throughputAtMax'][:] = throughputMax
        transmission['throughput'][:] = throughput[0]
        transmission['wavelengths'][:] = wavelength[0]

        return transmission

class ButlerMock(IsrMock):
    def mock(self, mockType='raw'):
        if mockType == 'raw':
            return RawMock().mock()
        elif mockType == 'bias':
            return BiasMock().mock()
        elif mockType == 'flat':
            return FlatMock().mock()
        elif mockType == 'dark':
            return DarkMock().mock()
        elif mockType == 'fringe':
            return FringeMock().mock()
