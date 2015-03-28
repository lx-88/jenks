from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef jenks_matrices_init(np.ndarray[DTYPE_t, ndim=1] data, Py_ssize_t n_classes):
    cdef Py_ssize_t n_data = data.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] lower_class_limits
    cdef np.ndarray[DTYPE_t, ndim=2] variance_combinations

    lower_class_limits = np.zeros((n_data+1, n_classes+1), dtype=DTYPE)
    lower_class_limits[1, 1:n_classes+1] = 1.0

    inf = float('inf')
    variance_combinations = np.zeros((n_data+1, n_classes+1), dtype=DTYPE)
    variance_combinations[2:n_data+1, 1:n_classes+1] = inf

    return lower_class_limits, variance_combinations
   
@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
@cython.nonecheck(False)
cdef jenks_matrices(np.ndarray[DTYPE_t, ndim=1] data, Py_ssize_t n_classes):
    cdef np.ndarray[DTYPE_t, ndim=2] lower_class_limits
    cdef np.ndarray[DTYPE_t, ndim=2] variance_combinations
    lower_class_limits, variance_combinations = jenks_matrices_init(data, n_classes)

    cdef Py_ssize_t l, sl = 2, nl = data.shape[0] + 1
    cdef Py_ssize_t m
    cdef Py_ssize_t j, jm1
    cdef Py_ssize_t i4
    cdef Py_ssize_t lower_class_limit
    cdef float sum = 0.0
    cdef float sum_squares = 0.0
    cdef float w = 0.0
    cdef float val
    cdef float variance = 0.0

    for l in range(2, nl):
        sum = 0.0
        sum_squares = 0.0
        w = 0.0
   
        for m in range(1, l+1):
            # `III` originally
            lower_class_limit = l - m + 1
            i4 = lower_class_limit - 1

            val = data[i4]

            # here we're estimating variance for each potential classing
            # of the data, for each potential number of classes. `w`
            # is the number of data points considered so far.
            w += 1.0

            # increase the current sum and sum-of-squares
            sum += val
            sum_squares += val * val

            # the variance at this point in the sequence is the difference
            # between the sum of squares and the total x 2, over the number
            # of samples.
            variance = sum_squares - (sum * sum) / w

            if i4 != 0:
                for j in range(2, n_classes+1):
                    jm1 = j - 1
                    if variance_combinations[l, j] >= (variance + variance_combinations[i4, jm1]):
                        lower_class_limits[l, j] = lower_class_limit
                        variance_combinations[l, j] = variance + variance_combinations[i4, jm1]

        lower_class_limits[l, 1] = 1.
        variance_combinations[l, 1] = variance

    return lower_class_limits, variance_combinations


def jenks(data, n_classes):
    if n_classes > len(data):
        return

    data = np.array(data, dtype=DTYPE)
    data.sort()

    lower_class_limits, variance_combinations = jenks_matrices(data, n_classes)

    k = data.shape[0] - 1
    kclass = [0.] * (n_classes+1)
    countNum = n_classes

    kclass[n_classes] = data[len(data) - 1]
    kclass[0] = data[0]

    while countNum > 1:
        elt = int(lower_class_limits[k][countNum] - 2)
        kclass[countNum - 1] = data[elt]
        k = int(lower_class_limits[k][countNum] - 1)
        countNum -= 1

    return kclass



def getQualityMetrics( data, breaks, n_classes ):
    """
    The Goodness of Variance Fit (GVF) is found by taking the
    difference between the squared deviations
    from the array mean (SDAM) and the squared deviations from the
    class means (SDCM), and dividing by the SDAM

    adapted from https://gist.github.com/drewda/1299198
    """

    if n_classes > len(data):
        return
    data = np.array(data, dtype=np.float32)
    data.sort()
    data = list(data)

    listMean = sum(data)/len(data)

    SDAM = 0.0
    for i in range(0,len(data)):
        sqDev = (data[i] - listMean)**2
        SDAM += sqDev

    SDCM = 0.0
    SDCM_list = list()
    for i in range(0,n_classes):
        if breaks[i] == 0:
            classStart = 0
        else:
            classStart = data.index(breaks[i])
            classStart += 1
        classEnd = data.index(breaks[i+1])
        classList = data[classStart:classEnd+1]

        try:
            classMean = sum(classList)/len(classList)
        except ZeroDivisionError:
            classMean = 1.0

        preSDCM = 0.0
        for j in range(0,len(classList)):
            sqDev2 = (classList[j] - classMean)**2
            preSDCM += sqDev2
        SDCM_list.append(preSDCM)
        SDCM += preSDCM
    return ((SDAM - SDCM)/SDAM, SDCM_list)


def classifyData(data, breaks, class_deviations, n_classes):
    """
    Modified version of function getQualityMetrics which is derived from https://gist.github.com/drewda/1299198
    The purpose of this function is to assign classes/groups based on the Jenks breaks to the unsorted data.
    Also report SDCM for each assignment attached to the data.
    """
    print "input:"
    print data
    print breaks
    print class_deviations
    print n_classes

    if n_classes > len(data):
        return

    data = np.array(data, dtype=np.float32)
    data_copy = sorted(list(data))

    classCol = np.zeros((data.shape[0],1), dtype=np.dtype('a100'))
    qualityCols = np.zeros((data.shape[0],1), dtype=np.float32)

    for i in range(0, n_classes):
        className = "class-{0}".format(str(i+1))
        classStart = breaks[i]
        classEnd = breaks[i+1]
        SDCM = class_deviations[i]

        if i == 0:
            qualityCols[(classStart == data)] = SDCM
            classCol[(classStart == data)] = className
        qualityCols[((classStart < data) & (data <= classEnd))] = SDCM
        classCol[((classStart < data) & (data <= classEnd))] = className
    return classCol, qualityCols

