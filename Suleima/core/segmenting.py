from totalsegmentator.python_api import totalsegmentator
from core.CT_3D import *
import os


def segmenting_volumes1(case: CardiacCT, skipSegmentation=False):
    """
    Generates preview images and radiomics features for an already-segmented cardiac CT.
    Assumes that segmentation results are already present in case.segmentPath.
    """
    casePath = case.casePath
    segmentPath = case.segmentPath  # Should point to the folder with segmentation output
    ctPath = f"{casePath}/fullCT.nii.gz"
    if not os.path.exists(ctPath):
        print(f"❌ Full CT for {case.caseID} does not exist.")
        return

    if not os.path.exists(segmentPath):
        print(f"❌ Segment path for {case.caseID} not found. Cannot run preview/radiomics.")
        return

    _ = totalsegmentator(
        input_path=ctPath,
        output_path=segmentPath,
        license_number="aca_BWYHC6UQQFDU8A",
        task="heartchambers_highres",
        body_seg=True,
        preview=True,
        radiomics=True,
        #skip_segmentation=skipSegmentation
        skip_segmentation=True
    )
    
    
import subprocess

def segmenting_volumes(case: CardiacCT, skipSegmentation=True):
    ctPath = os.path.join(case.casePath, "fullCT.nii.gz")
    segmentPath = case.segmentPath

    if not os.path.exists(ctPath):
        print(f"❌ Full CT for {case.caseID} does not exist.")
        return

    os.makedirs(segmentPath, exist_ok=True)

    cmd = [
        "totalsegmentator",
        ctPath,
        segmentPath,
        "--task", "heartchambers_highres",
        "--body_seg",
        "--radiomics",
        "--preview",
    ]

    if skipSegmentation:
        cmd.append("--skip_segmentation")

    subprocess.run(cmd, check=True)
 