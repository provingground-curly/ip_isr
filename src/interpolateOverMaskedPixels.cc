// -*- LSST-C++ -*- // fixed format comment for emacs
/**
  * \file
  *
  * \ingroup isr
  *
  * \brief Implementation of the templated utility function, Interpolate
  * Over Masked Pixels.
  *
  * \author Nicole M. Silvestri, University of Washington
  */
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>
#include <boost/format.hpp>

#include <lsst/afw/image/Exposure.h>
#include <lsst/afw/math/Function.h>
#include <lsst/afw/image/Mask.h>
#include <lsst/afw/image/MaskedImage.h>
#include <lsst/daf/base/DataProperty.h>
#include <lsst/pex/exceptions/Exception.h>
// #include <lsst/utils/Trace.h>
#include <lsst/pex/policy/Policy.h>

#include "lsst/ip/isr/isr.h"

/** \brief Utility function for interpolation .  Eventually will accept several
  * interpolation methods - implementing the process of linear prediction
  * (Press and Rybicki 1993) first.
  */

template<typename ImageT, typename MaskT>
lsst::afw::image::Exposure<ImageT, MaskT> interpolateOverMaskedpixels(
    lsst::afw::image::Exposure<ImageT, MaskT> &chunkMaskedImage,    
    lsst::daf::base::DataProperty::PtrType &chunkMetaData, 
    std::string interpMethod 
    ) { 

    

   
}

/************************************************************************/
/* Explicit instantiations */