#deviation angle
import numpy as np


# optical flow of each frame is represented in a 2d array of shape(n,2) where n is the number of pixels being tracked
#opt_flw used for the current's frame optical flow
#opt_flw_old used for the previous' frame optical flow

#numerator = np.sum(np.multiply(opt_flw,opt_flw_old),axis=1)
#denominator = np.sqrt(np.sum(opt_flw**2,axis=1)) * np.sqrt(np.sum(opt_flw_old**2,axis=1))
#angle_difference = np.sum(np.multiply(opt_flw,opt_flw_old),axis=1)/(np.sqrt(np.sum(opt_flw**2,axis=1)) * np.sqrt(np.sum(opt_flw_old**2,axis=1)))

def angle_difference_feature(opt_flw,opt_flw_old):
    rms=np.sqrt(np.sum(opt_flw**2,axis=1))
    angle_difference = np.sum(np.multiply(opt_flw,opt_flw_old),axis=1)/(rms * np.sqrt(np.sum(opt_flw_old**2,axis=1)))
    return rms * angle_difference

#test
opt_flw_old=np.array([[1, 2],[3, 4],[5, 6],[7, 8]],dtype='float64')
opt_flw=np.array([[1, 2],[3, 4],[5, 6],[7, 8]],dtype='float64')
print(angle_difference_feature(opt_flw,opt_flw_old))


""" calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        
    the calcOpticalFlowPyrLK() function returns a vector of 2D points of the next pixels to track
    of shape(n,1,2)
    assuming that we have 
        prevPts ( a 2d vector of the previous pts to track)
        nextPts ( a 2d vector of the next pts to track)
        we can estimate frame's optical flow value by substracting prevPts to nextPts
        we end up with a 2d points of optical flow values for the frame of shape (n,2,1)
        In order to use the angle difference, the optical flow array need to be reshaped into an (n,2) array
        arr.reshape(arr.shape[0],2)
        """