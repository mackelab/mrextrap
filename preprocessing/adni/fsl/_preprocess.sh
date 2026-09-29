#!/bin/sh

# General FSL anatomical processing pipeline
#
#   Mark Jenkinson
#   FMRIB Image Analysis Group
#
#   Copyright (C) 2012 University of Oxford
#
#   Part of FSL - FMRIB's Software Library
#   http://www.fmrib.ox.ac.uk/fsl
#   fsl@fmrib.ox.ac.uk
#
#   Developed at FMRIB (Oxford Centre for Functional Magnetic Resonance
#   Imaging of the Brain), Department of Clinical Neurology, Oxford
#   University, Oxford, UK
#
#
#   LICENCE
#
#   FMRIB Software Library, Release 5.0 (c) 2012, The University of
#   Oxford (the "Software")
#
#   The Software remains the property of the University of Oxford ("the
#   University").
#
#   The Software is distributed "AS IS" under this Licence solely for
#   non-commercial use in the hope that it will be useful, but in order
#   that the University as a charitable foundation protects its assets for
#   the benefit of its educational and research purposes, the University
#   makes clear that no condition is made or to be implied, nor is any
#   warranty given or to be implied, as to the accuracy of the Software,
#   or that it will be suitable for any particular purpose or for use
#   under any specific conditions. Furthermore, the University disclaims
#   all responsibility for the use which is made of the Software. It
#   further disclaims any liability for the outcomes arising from using
#   the Software.
#
#   The Licensee agrees to indemnify the University and hold the
#   University harmless from and against any and all claims, damages and
#   liabilities asserted by third parties (including claims for
#   negligence) which arise directly or indirectly from the use of the
#   Software or the sale of any products based on the Software.
#
#   No part of the Software may be reproduced, modified, transmitted or
#   transferred in any form or by any means, electronic or mechanical,
#   without the express permission of the University. The permission of
#   the University is not required if the said reproduction, modification,
#   transmission or transference is done without financial return, the
#   conditions of this Licence are imposed upon the receiver of the
#   product, and all original and amended source code is included in any
#   transmitted product. You may be held legally responsible for any
#   copyright infringement that is caused or encouraged by your failure to
#   abide by these terms and conditions.
#
#   You are not permitted under this Licence to use this Software
#   commercially. Use for which any financial return is received shall be
#   defined as commercial use, and includes (1) integration of all or part
#   of the source code or the Software into a product for sale or license
#   by or on behalf of Licensee to third parties or (2) use of the
#   Software or any derivative of it for research with the final aim of
#   developing software products for sale or license to a third party or
#   (3) use of the Software or any derivative of it for research with the
#   final aim of developing non-software products for sale or license to a
#   third party, or (4) use of the Software to provide any service to an
#   external organisation for which payment is received. If you are
#   interested in using the Software commercially, please contact Oxford
#   University Innovation ("OUI"), the technology transfer company of the
#   University, to negotiate a licence. Contact details are:
#   Innovation@innovation.ox.ac.uk quoting reference DE/9564.
export LC_ALL=C

set -e
LOGFILE=log.txt

# The following is a debugging line (displays all commands as they are executed)
# set -x

Usage() {
    echo "Usage: $(basename $0) [options] -i <structural image>"
    echo "       $(basename $0) [options] -d <existing anat directory>"
    #echo "       `basename $0` [options] --list=<list of image names OR a text file>"
    echo " "
    echo "Arguments (You may specify one or more of):"
    echo "  -i <strucural image>         filename of input image (for one image only)"
    echo "  -d <anat dir>                directory name for existing .anat directory where this script will be run in place"
    echo "  -o <output directory>        basename of directory for output (default is input image basename followed by .anat)"
    #echo "  --list=<image list>          specifies a list of images to be averaged (either a comma separated list of image names with no spaces, or the filename for a text file containing the individual image filenames)"
    echo "  --clobber                    if .anat directory exist (as specified by -o or default from -i) then delete it and make a new one"
    echo "  --strongbias                 used for images with very strong bias fields"
    echo "  --weakbias                   used for images with smoother, more typical, bias fields (default setting)"
    echo "  --noreorient                 turn off step that does reorientation 2 standard (fslreorient2std)"
    echo "  --nocrop                     turn off step that does automated cropping (robustfov)"
    echo "  --nobet                      turn off step that does brain extraction (BET or registration) - to use this the input image must already be brain extracted"
    echo "  --nobias                     turn off steps that do bias field correction (via FAST)"
    echo "  --noreg                      turn off steps that do registration to standard (FLIRT and FNIRT)"
    echo "  --nononlinreg                turn off step that does non-linear registration (FNIRT)"
    echo "  --noseg                      turn off step that does tissue-type segmentation (FAST)"
    echo "  --nosubcortseg               turn off step that does sub-cortical segmentation (FIRST)"
    echo "  -s <value>                   specify the value for bias field smoothing (the -l option in FAST)"
    echo "  -t <type>                    specify the type of image (choose one of T1 T2 PD - default is T1)"
    echo "  -m <lesion mask>             use the mask image to exclude areas (e.g. lesions) -  voxels=1 in mask are excluded/deweighted"
    echo "  --nosearch                   specify that linear registration uses the -nosearch option (FLIRT)"
    echo "  --betfparam                  specify f parameter for BET (only used if not running non-linear reg and also wanting brain extraction done)"
    echo "  --nocleanup                  do not remove intermediate files"
    echo " "
}

# extracts the option name from any version (-- or -)
get_opt1() {
    arg=$(echo $1 | sed 's/=.*//')
    echo $arg
}

# get arg for -- options
get_arg1() {
    if [ X$(echo $1 | grep '=') = X ]; then
        echo "Option $1 requires an argument" 1>&2
        exit 1
    else
        arg=$(echo $1 | sed 's/.*=//')
        if [ X$arg = X ]; then
            echo "Option $1 requires an argument" 1>&2
            exit 1
        fi
        echo $arg
    fi
}

# get image filename from -- options
get_imarg1() {
    arg=$(get_arg1 $1)
    arg=$($FSLDIR/bin/remove_ext $arg)
    echo $arg
}

# get arg for - options (need to pass both $1 and $2 to this)
get_arg2() {
    if [ X$2 = X ]; then
        echo "Option $1 requires an argument" 1>&2
        exit 1
    fi
    echo $2
}

# get arg of image filenames for - options (need to pass both $1 and $2 to this)
get_imarg2() {
    arg=$(get_arg2 $1 $2)
    arg=$($FSLDIR/bin/remove_ext $arg)
    echo $arg
}

run() {
    echo $@ >>$LOGFILE
    $@
}

quick_smooth() {
    in=$1
    out=$2
    run $FSLDIR/bin/fslmaths $in -subsamp2 -subsamp2 -subsamp2 -subsamp2 vol16
    run $FSLDIR/bin/flirt -in vol16 -ref $in -out $out -noresampblur -applyxfm -paddingsize 16
    # possibly do a tiny extra smooth to $out here?
    run $FSLDIR/bin/imrm vol16
}

# Parse input arguments

# default values
inputimage=
imagelist=
outputname=
anatdir=
lesionmask=

strongbias=no
do_reorient=yes
do_crop=yes
do_bet=yes
do_optibet=yes
do_biasrestore=yes
do_reg=yes
do_nonlinreg=yes
do_seg=yes
do_subcortseg=yes
do_cleanup=yes
clobber=no
multipleimages=no
use_lesionmask=no

nosearch=
niter=1
smooth=20
betfparam=0.1
type=1 # For FAST: 1 = T1w, 2 = T2w, 3 = PD

# Parse! Parse! Parse!

if [ $# -eq 0 ]; then
    Usage
    exit 0
fi
if [ $# -lt 2 ]; then
    Usage
    exit 1
fi
while [ $# -ge 1 ]; do
    iarg=$(get_opt1 $1)
    case "$iarg" in

    -i)
        inputimage=$(get_imarg2 $1 $2)
        shift 2
        ;;
    -o)
        outputname=$(get_arg2 $1 $2)
        shift 2
        ;;
    -d)
        anatdir=$(get_arg2 $1 $2)
        shift 2
        ;;
    -s)
        smooth=$(get_arg2 $1 $2)
        shift 2
        ;;
    -m)
        use_lesionmask=yes
        lesionmask=$(get_arg2 $1 $2)
        shift 2
        ;;
    -t)
        typestr=$(get_arg2 $1 $2)
        if [ $typestr = T1 ]; then type=1; fi
        if [ $typestr = T2 ]; then type=2; fi
        if [ $typestr = PD ]; then type=3; fi
        shift 2
        ;;
    --list)
        imagelist=$(get_arg1 $1)
        multipleimages=yes
        shift
        ;;
    --clobber)
        clobber=yes
        shift
        ;;
    --noreorient)
        do_reorient=no
        shift
        ;;
    --nocrop)
        do_crop=no
        shift
        ;;
    --nobet)
        do_bet=no
        shift
        ;;
    --nooptibet)
        do_optibet=no
        shift
        ;;
    --noreg)
        do_reg=no
        shift
        ;;
    --nononlinreg)
        do_nonlinreg=no
        shift
        ;;
    --noseg)
        do_seg=no
        shift
        ;;
    --nosubcortseg)
        do_subcortseg=no
        shift
        ;;
    --nobias)
        do_biasrestore=no
        shift
        ;;
    --nosearch)
        nosearch=-nosearch
        shift
        ;;
    --strongbias)
        strongbias=yes
        niter=5
        smooth=10
        shift
        ;;
    --weakbias)
        strongbias=no
        niter=10
        smooth=20
        shift
        ;;
    --betfparam)
        betfparam=$(get_arg1 $1)
        shift
        ;;
    --nocleanup)
        do_cleanup=no
        shift
        ;;
    -v)
        verbose=yes
        shift
        ;;
    -h)
        Usage
        exit 0
        ;;
    *)
        #if [ `echo $1 | sed 's/^\(.\).*/\1/'` = "-" ] ; then
        echo "Unrecognised option $1" 1>&2
        exit 1
        #fi
        #shift;;
        ;;
    esac
done

### Sanity checking of arguments

if [ X$inputimage = X ] && [ X$anatdir = X ] && [ X"$imagelist" = X ]; then
    #echo "One of the compulsory arguments -i, -d or --list MUST be used"
    echo "One of the compulsory arguments -i or -d MUST be used"
    exit 1
fi

if [ $type != 1 ]; then
    if [ $do_nonlinreg = yes ]; then
        echo "ERROR: Cannot do non-linear registration with non-T1 images, please re-run with --nononlinreg"
        exit 1
    fi
    if [ $do_subcortseg = yes ]; then
        echo "ERROR: Cannot perform subcortical segmentation (with FIRST) on a non-T1 image, please re-run with --nosubcortseg"
        exit 1
    fi
fi

T1=T1
if [ $type = 2 ]; then T1=T2; fi
if [ $type = 3 ]; then T1=PD; fi

betopts="-f ${betfparam}"

###### Now do some work!

# setup output directory (or go to existing one)

if [ X$anatdir = X ]; then
    if [ X${outputname} = X ]; then
        outputname=$inputimage
    fi

    if [ -d ${outputname}.anat ]; then
        if [ $clobber = no ]; then
            echo "ERROR: Directory ${outputname}.anat already exists!"
            exit 1
        else
            rm -rf ${outputname}.anat
        fi
    fi
    mkdir ${outputname}.anat
else
    if [ X${inputimage} != X ]; then
        echo "ERROR: Cannot specify both -d and -i"
        exit 1
    fi
    if [ $multipleimages = yes ]; then
        echo "ERROR: Cannot specify both -d and --list"
        exit 1
    fi
    outputname=$($FSLDIR/bin/fsl_abspath $anatdir)
    outputname=$(echo $outputname | sed 's/\.anat$//')
    if [ ! -d ${outputname}.anat ]; then
        echo "ERROR: Directory ${outputname}.anat not found"
        exit 1
    fi
    inputimage=$anatdir/T1
fi

# some initial reporting for the log file
echo "Script invoked from directory = $(pwd)" >>${outputname}.anat/$LOGFILE
echo "Output directory = ${outputname}.anat" >>${outputname}.anat/$LOGFILE

if [ $multipleimages = yes ]; then
    if [ X${inputimage} != X ]; then
        echo "ERROR: Cannot specify both -i and --list"
        exit 1
    fi
    im1=$(echo $imagelist | sed 's/,/ /g' | awk '{ print $1 }')
    if [ $FSLDIR/bin/imtest $im1 = 1 ]; then
        # this is a comma separated list of image names
        namelist="$(echo ${imagelist} | sed 's/,/ /g')"
    else
        # this is a file containing the image names
        namelist="$(cat ${imagelist})"
    fi
    for name in $namelist; do
        if [ $FSLDIR/bin/imtest $name = 0 ]; then
            echo "ERROR: Cannot find image $name"
            exit 1
        fi
        $FSLDIR/bin/fslmaths $name ${outputname}.anat/${T1}_${num}
    done
    echo "Input images are ${namelist}" >>${outputname}.anat/$LOGFILE
else
    $FSLDIR/bin/fslmaths ${inputimage} ${outputname}.anat/${T1}
    echo "Input image is ${inputimage}" >>${outputname}.anat/$LOGFILE
fi

if [ $use_lesionmask = yes ]; then
    $FSLDIR/bin/fslmaths $lesionmask ${outputname}.anat/lesionmask
    echo "Lesion mask is ${lesionmask}" >>${outputname}.anat/$LOGFILE
fi

cd ${outputname}.anat
echo " " >>$LOGFILE

# now the real work

#### AVERAGING MULTIPLE SCANS
# required input: list of images linked from the input argument with names ${T1}_[0-9]*
# output: ${T1}
if [ $multipleimages = yes ]; then
    date
    echo "Averaging list of input images"
    mkdir average_dir
    run $FSLDIR/bin/AnatomicalAverage -w average_dir -o ${T1} $($FSLDIR/bin/imglob ${T1}_*)
fi

#### FIXING NEGATIVE RANGE
# required input: ${T1}
# output: ${T1}
minval=$($FSLDIR/bin/fslstats ${T1} -p 0)
maxval=$($FSLDIR/bin/fslstats ${T1} -p 100)
if [ X$(echo "if ( $minval < 0 ) { 1 }" | bc -l) = X1 ]; then
    if [ X$(echo "if ( $maxval > 0 ) { 1 }" | bc -l) = X1 ]; then
        # if there are just some negative values among the positive ones then reset zero to the min value
        run ${FSLDIR}/bin/fslmaths ${T1} -sub $minval ${T1} -odt float
    else
        # if all values are negative then make them positive, but retain any zeros as zeros
        run ${FSLDIR}/bin/fslmaths ${T1} -bin -binv zeromask
        run ${FSLDIR}/bin/fslmaths ${T1} -sub $minval -mas zeromask ${T1} -odt float
    fi
fi

#### REORIENTATION 2 STANDARD
# required input: ${T1}
# output: ${T1} (modified) [ and ${T1}_orig and .mat ]
if [ $do_reorient = yes ]; then
    date
    echo "Reorienting to standard orientation"
    run $FSLDIR/bin/fslmaths ${T1} ${T1}_orig
    run $FSLDIR/bin/fslreorient2std ${T1} >${T1}_orig2std.mat
    run $FSLDIR/bin/convert_xfm -omat ${T1}_std2orig.mat -inverse ${T1}_orig2std.mat
    run $FSLDIR/bin/fslreorient2std ${T1} ${T1}
fi

#### AUTOMATIC CROPPING
# required input: ${T1}
# output: ${T1} (modified) [ and ${T1}_fullfov plus various .mats ]
if [ $do_crop = yes ]; then
    date
    echo "Automatically cropping the image"
    run $FSLDIR/bin/immv ${T1} ${T1}_fullfov
    run $FSLDIR/bin/robustfov -i ${T1}_fullfov -r ${T1} -m ${T1}_roi2nonroi.mat -b 210 | grep [0-9] | tail -1 >${T1}_roi.log
    # combine this mat file and the one above (if generated)
    if [ $do_reorient = yes ]; then
        run $FSLDIR/bin/convert_xfm -omat ${T1}_nonroi2roi.mat -inverse ${T1}_roi2nonroi.mat
        run $FSLDIR/bin/convert_xfm -omat ${T1}_orig2roi.mat -concat ${T1}_nonroi2roi.mat ${T1}_orig2std.mat
        run $FSLDIR/bin/convert_xfm -omat ${T1}_roi2orig.mat -inverse ${T1}_orig2roi.mat
    fi
fi

### LESION MASK
# make appropriate (reoreinted and cropped) lesion mask (or a default blank mask to simplify the code later on)
if [ $use_lesionmask = yes ]; then
    if [ -f ${T1}_orig2std.mat ]; then transform=${T1}_orig2std.mat; fi
    if [ -f ${T1}_orig2roi.mat ]; then transform=${T1}_orig2roi.mat; fi # this takes precedence if both exist
    if [ X$transform != X ]; then
        $FSLDIR/bin/fslmaths lesionmask lesionmask_orig
        $FSLDIR/bin/flirt -in lesionmask_orig -ref ${T1} -applyxfm -interp nearestneighbour -init ${transform} -out lesionmask
    fi
else
    $FSLDIR/bin/fslmaths ${T1} -mul 0 lesionmask
fi
$FSLDIR/bin/fslmaths lesionmask -bin lesionmask
$FSLDIR/bin/fslmaths lesionmask -binv lesionmaskinv

##### BIAS FIELD CORRECTION (main work, although also refined later on if segmentation run)
# required input: ${T1}
# output: ${T1}_biascorr  [ other intermediates to be cleaned up ]
if [ $do_biasrestore = yes ]; then
    if [ $strongbias = yes ]; then
        date
        echo "Estimating and removing field (stage 1 - large-scale fields)"
        # for the first step (very gross bias field) don't worry about the lesionmask
        # the following is a replacement for : run $FSLDIR/bin/fslmaths ${T1} -s 20 ${T1}_s20
        quick_smooth ${T1} ${T1}_s20
        run $FSLDIR/bin/fslmaths ${T1} -div ${T1}_s20 ${T1}_hpf
        if [ $do_bet = yes ]; then
            # get a rough brain mask - it can be *VERY* rough (i.e. missing huge portions of the brain or including non-brain, but non-background) - use -f 0.1 to err on being over inclusive
            run $FSLDIR/bin/bet ${T1}_hpf ${T1}_hpf_brain -m -f 0.1

        else
            run $FSLDIR/bin/fslmaths ${T1}_hpf ${T1}_hpf_brain
            run $FSLDIR/bin/fslmaths ${T1}_hpf_brain -bin ${T1}_hpf_brain_mask
        fi
        run $FSLDIR/bin/fslmaths ${T1}_hpf_brain_mask -mas lesionmaskinv ${T1}_hpf_brain_mask
        # get a smoothed version without the edge effects
        run $FSLDIR/bin/fslmaths ${T1} -mas ${T1}_hpf_brain_mask ${T1}_hpf_s20
        quick_smooth ${T1}_hpf_s20 ${T1}_hpf_s20
        quick_smooth ${T1}_hpf_brain_mask ${T1}_initmask_s20
        run $FSLDIR/bin/fslmaths ${T1}_hpf_s20 -div ${T1}_initmask_s20 -mas ${T1}_hpf_brain_mask ${T1}_hpf2_s20
        run $FSLDIR/bin/fslmaths ${T1} -mas ${T1}_hpf_brain_mask -div ${T1}_hpf2_s20 ${T1}_hpf2_brain
        # make sure the overall scaling doesn't change (equate medians)
        med0=$($FSLDIR/bin/fslstats ${T1} -k ${T1}_hpf_brain_mask -P 50)
        med1=$($FSLDIR/bin/fslstats ${T1}_hpf2_brain -k ${T1}_hpf_brain_mask -P 50)
        run $FSLDIR/bin/fslmaths ${T1}_hpf2_brain -div $med1 -mul $med0 ${T1}_hpf2_brain
        date
        echo "Estimating and removing bias field (stage 2 - detailed fields)"
        run $FSLDIR/bin/fslmaths ${T1}_hpf2_brain -mas lesionmaskinv ${T1}_hpf2_maskedbrain
        run $FSLDIR/bin/fast -o ${T1}_initfast -l ${smooth} -b -B -t $type --iter=${niter} --nopve --fixed=0 -v ${T1}_hpf2_maskedbrain
        run $FSLDIR/bin/fslmaths ${T1}_initfast_restore -mas lesionmaskinv ${T1}_initfast_maskedrestore
        run $FSLDIR/bin/fast -o ${T1}_initfast2 -l ${smooth} -b -B -t $type --iter=${niter} --nopve --fixed=0 -v ${T1}_initfast_maskedrestore
        run $FSLDIR/bin/fslmaths ${T1}_hpf_brain_mask ${T1}_initfast2_brain_mask
    else
        if [ $do_bet = yes ]; then
            # get a rough brain mask - it can be *VERY* rough (i.e. missing huge portions of the brain or including non-brain, but non-background) - use -f 0.1 to err on being over inclusive
            run $FSLDIR/bin/bet ${T1} ${T1}_initfast2_brain -m -f 0.1

        else
            run $FSLDIR/bin/fslmaths ${T1} ${T1}_initfast2_brain
            run $FSLDIR/bin/fslmaths ${T1}_initfast2_brain -bin ${T1}_initfast2_brain_mask
        fi
        run $FSLDIR/bin/fslmaths ${T1}_initfast2_brain ${T1}_initfast2_restore
    fi
    # redo fast again to try and improve bias field
    run $FSLDIR/bin/fslmaths ${T1}_initfast2_restore -mas lesionmaskinv ${T1}_initfast2_maskedrestore
    run $FSLDIR/bin/fast -o ${T1}_fast -l ${smooth} -b -B -t $type --iter=${niter} --nopve --fixed=0 -v ${T1}_initfast2_maskedrestore

    date
    echo "Extrapolating bias field from central region"
    # use the latest fast output
    run $FSLDIR/bin/fslmaths ${T1} -div ${T1}_fast_restore -mas ${T1}_initfast2_brain_mask ${T1}_fast_totbias
    run $FSLDIR/bin/fslmaths ${T1}_initfast2_brain_mask -ero -ero -ero -ero -mas lesionmaskinv ${T1}_initfast2_brain_mask2
    run $FSLDIR/bin/fslmaths ${T1}_fast_totbias -sub 1 ${T1}_fast_totbias
    run $FSLDIR/bin/fslsmoothfill -i ${T1}_fast_totbias -m ${T1}_initfast2_brain_mask2 -o ${T1}_fast_bias
    run $FSLDIR/bin/fslmaths ${T1}_fast_bias -add 1 ${T1}_fast_bias
    run $FSLDIR/bin/fslmaths ${T1}_fast_totbias -add 1 ${T1}_fast_totbias
    # run $FSLDIR/bin/fslmaths ${T1}_fast_totbias -sub 1 -mas ${T1}_initfast2_brain_mask -dilall -add 1 ${T1}_fast_bias  # alternative to fslsmoothfill
    run $FSLDIR/bin/fslmaths ${T1} -div ${T1}_fast_bias ${T1}_biascorr

    #### optibet run
    # ./scripts/optiBET.sh ${T1}_biascorr

    if [ $do_optibet = yes ]; then
        SCRIPT=$(realpath $0)
        SCRIPTPATH=$(dirname $SCRIPT)
        $SCRIPTPATH/optiBET.sh ${T1}_biascorr
        # produces ${T1}_biascorr_optiBET_brain and ${T1}_biascorr_optiBET_brain_mask
    fi
else
    run $FSLDIR/bin/fslmaths ${T1} ${T1}_biascorr
fi

###### REGISTRATION ####
########################
###----------------------------
# only rigid registration (6 dof) and scaling+rigid (7 dof) to standard MNI space
if [ $do_reg = yes ]; then
    echo "Registering to standard space (linear)"
    if [ $do_optibet = yes ]; then
        # linear registration (6 dof) to standard space
        run $FSLDIR/bin/flirt -dof 6 -in ${T1}_biascorr_optiBET_brain -ref $FSLDIR/data/standard/MNI152_${T1}_1mm_brain -omat ${T1}_to_MNI_lin_stripped.mat -out ${T1}_to_MNI_lin_stripped
        # linear registration (7 dof) to standard space
        run $FSLDIR/bin/flirt -dof 7 -in ${T1}_biascorr_optiBET_brain -ref $FSLDIR/data/standard/MNI152_${T1}_1mm_brain -omat ${T1}_to_MNI_lin_stripped_scaled.mat -out ${T1}_to_MNI_lin_stripped_scaled
    fi
    # anyways do the non-optibet version
    echo "linear registration (6 dof) to standard space"
    run $FSLDIR/bin/flirt -dof 6 -in ${T1}_biascorr -ref $FSLDIR/data/standard/MNI152_${T1}_1mm -omat ${T1}_to_MNI_lin.mat -out ${T1}_to_MNI_lin
    echo "linear registration (7 dof) to standard space"
    run $FSLDIR/bin/flirt -dof 7 -in ${T1}_biascorr -ref $FSLDIR/data/standard/MNI152_${T1}_1mm -omat ${T1}_to_MNI_lin_scaled.mat -out ${T1}_to_MNI_lin_scaled
fi

if [ $do_cleanup = yes ]; then
    date
    echo "Cleaning up intermediate files"
    run $FSLDIR/bin/imrm ${T1}_biascorr_bet_mask ${T1}_biascorr_bet ${T1}_biascorr_brain_mask2 ${T1}_biascorr_init ${T1}_biascorr_maskedbrain ${T1}_biascorr_to_std_sub ${T1}_fast_bias_idxmask ${T1}_fast_bias_init ${T1}_fast_bias_vol2 ${T1}_fast_bias_vol32 ${T1}_fast_totbias ${T1}_hpf* ${T1}_initfast* ${T1}_s20 ${T1}_initmask_s20
fi
