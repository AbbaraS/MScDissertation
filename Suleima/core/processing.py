

import numpy as np

def crop(case):
    lv, rv, la, ra, myo = case.lv, case.rv, case.la, case.ra, case.myo
    ct_data = case.ct_data

    binary_mask = ((lv + rv + la + ra + myo) > 0).astype(np.uint8)
    coords = np.array(np.where(binary_mask))
    if coords.size == 0:
        case.log_error("No heart mask found.")
        return

    x_min, y_min, z_min = coords.min(axis=1)
    x_max, y_max, z_max = coords.max(axis=1)
    x0, x1 = max(x_min, 0), min(x_max + 1, ct_data.shape[0])
    y0, y1 = max(y_min, 0), min(y_max + 1, ct_data.shape[1])
    z0, z1 = max(z_min, 0), min(z_max + 1, ct_data.shape[2])

    case.log_error(f"3D bounding box coordinates- x: {x0}-{x1}, y: {y0}-{y1}, z: {z0}-{z1}")

    # Save cropped versions as attributes if you want
    case.croppedCT = ct_data[x0:x1, y0:y1, z0:z1]
    case.croppedLV = lv[x0:x1, y0:y1, z0:z1]
    case.croppedRV = rv[x0:x1, y0:y1, z0:z1]
    case.croppedLA = la[x0:x1, y0:y1, z0:z1]
    case.croppedRA = ra[x0:x1, y0:y1, z0:z1]
    case.croppedMYO = myo[x0:x1, y0:y1, z0:z1]
    
    