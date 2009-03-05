// -*- LSST-C++ -*- 
/**
  * \file
  *
  * \ingroup isr
  *
  * \brief Implementation of the templated Instrument Signature Removal 
  * stage of the nightly LSST Image Processing Pipeline.
  *
  * \author Nicole M. Silvestri / ACB, University of Washington
  *
  * Contact: nms@astro.washington.edu
  *
  * \version
  *
  * LSST Legalese here...
  */
	
#ifndef LSST_IP_ISR_ISR_H
#define LSST_IP_ISR_ISR_H
	
#include <string>
#include <vector>

#include <lsst/afw/math.h>
#include <lsst/afw/image.h>
#include <lsst/pex/exceptions/Exception.h>
#include <lsst/pex/logging/Trace.h>

/** \brief Remove all non-astronomical counts from the Chunk Exposure's pixels.
  * 
  */
	
namespace lsst {
namespace ip {
namespace isr {

    /** Signature strings associated with each stage of the ISR
     * 
     * @note Added to Exposure metadata after stage processing
     *
     * @note As implementation detail, no more than 8 characters for fits
     * compliance?
     */
    std::string const& ISR_LIN   = "ISR_LIN";    ///< Linearization
    std::string const& ISR_OSCAN = "ISR_OSCAN";  ///< Overscan
    std::string const& ISR_TRIM  = "ISR_TRIM";   ///< Trim
    std::string const& ISR_BIAS  = "ISR_BIAS";   ///< Bias 
    std::string const& ISR_DFLAT = "ISR_DFLAT";  ///< Dome flat
    std::string const& ISR_ILLUM = "ISR_ILLUM";  ///< Illumination correction
    std::string const& ISR_BADP  = "ISR_BADP";   ///< Bad pixel mask
    std::string const& ISR_SAT   = "ISR_SAT";    ///< Saturated pixels
    std::string const& ISR_FRIN  = "ISR_FRIN";   ///< Fringe correction
    std::string const& ISR_DARK  = "ISR_DARK";   ///< Dark correction

    enum StageId {
        ISR_LINid   = 0x1,   ///< Linearization
        ISR_OSCANid = 0x2,   ///< Overscan
        ISR_TRIMid  = 0x4,   ///< Trim
        ISR_BIASid  = 0x8,   ///< Bias 
        ISR_DFLATid = 0x10,  ///< Dome flat
        ISR_ILLUMid = 0x20,  ///< Illumination correction
        ISR_BADPid  = 0x40,  ///< Bad pixel mask
        ISR_SATid   = 0x80,  ///< Saturated pixels
        ISR_FRINid  = 0x100, ///< Fringe correction
        ISR_DARKid  = 0x100, ///< Dark correction
    };

    /** Multiplicative linearization lookup table
     *
     * @ingroup isr
     */
    template <typename ImageT>
    class LookupTableMultiplicative {
    public:
        typedef typename lsst::afw::image::MaskedImage<ImageT>::x_iterator x_iterator;
        typedef typename lsst::afw::image::MaskedImage<ImageT>::Pixel PixelT;

        LookupTableMultiplicative(std::vector<double> table) : 
            _table(table), _max(table.size()) {};
        virtual ~LookupTableMultiplicative() {};

        void apply(lsst::afw::image::MaskedImage<ImageT> &image) {

            for (int y = 0; y != image.getHeight(); ++y) {
                for (x_iterator ptr = image.row_begin(y); ptr != image.row_end(y); ++ptr) {
                    int ind = static_cast<int>(ptr.image() + 0.5);  // Rounded pixel value
                    if (ind >= _max){
                        throw LSST_EXCEPT(lsst::pex::exceptions::Exception, 
                                          "Pixel value out of range in LookupTableMultiplicative::apply");
                    }
                    PixelT p = PixelT((*ptr).image() * _table[ind], 
                                      (*ptr).mask(), 
                                      (*ptr).variance() * _table[ind] * _table[ind]);
                    *ptr = p;
                }
            }
        }

        // Return the lookup table
        std::vector<double> getTable() const { return _table; }
    private:
        std::vector<double> _table;
        int _max;
    };

    /** Linearization lookup table with replacement
     *
     * @ingroup isr
     */
    template <typename ImageT>
    class LookupTableReplace {
    public:
        typedef typename lsst::afw::image::MaskedImage<ImageT>::x_iterator x_iterator;
        typedef typename lsst::afw::image::MaskedImage<ImageT>::Pixel PixelT;

        LookupTableReplace(std::vector<double> table) : 
            _table(table), _max(table.size()) {};
        virtual ~LookupTableReplace() {};

        void apply(lsst::afw::image::MaskedImage<ImageT> &image, double gain=1.0) {
            double igain = 1.0 / gain;
            for (int y = 0; y != image.getHeight(); ++y) {
                for (x_iterator ptr = image.row_begin(y); ptr != image.row_end(y); ++ptr) {
                    int ind = static_cast<int>(ptr.image() + 0.5);  // Rounded pixel value
                    if (ind >= _max){
                        throw LSST_EXCEPT(lsst::pex::exceptions::Exception, 
                                          "Pixel value out of range in LookupTableReplace::apply");
                    }
                    PixelT p = PixelT(_table[ind], 
                                      (*ptr).mask(), 
                                      _table[ind] * igain);
                    *ptr = p;
                }
            }
        }

        // Return the lookup table
        std::vector<double> getTable() const { return _table; }
    private:
        std::vector<double> _table;
        int _max;
    };





    template<typename ImagePixelT>
    lsst::afw::math::FitResults findBestFit(
        lsst::afw::image::MaskedImage<ImagePixelT> const &maskedImage,
        std::string const &funcForm,
        int funcOrder,
        double stepSize
        );

    lsst::afw::image::BBox stringParse(
        std::string &section
        );

    template<typename ImagePixelT>
    void fitFunctionToImage(
        lsst::afw::image::MaskedImage<ImagePixelT> &maskedImage,
        lsst::afw::math::Function1<double> const &function
        );

    

}}} // namespace lsst::ip::isr
	
#endif // !defined(LSST_IP_ISR_ISR_H)
