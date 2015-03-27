from jenks import jenks, getQualityMetrics, classifyData


jenks_test = [-6.50e-1,
 1.10e+0,
 1.40e+0,
 8.00e-1,
 6.80e-1,
 -7.60e-1,
 1.10e+0,
 -8.70e-1,
 1.10e+0,
 1.50e+0,
 1.10e+0,
 7.40e-1,
 5.60e-1,
 1.30e+0,
 8.90e-1,
 8.80e-1,
 1.20e+0,
 8.40e-1,
 6.30e-1, 1.10e+0]

dta = [5.2,4.2,5.121,6.2,7.4,8.1,1.1,2.6,3.2,4.12,5.34,6.21,7.123,2.1345,1.1,3.1]
dta = jenks_test

N_CLASSES = 3

print "Length of data: {0}".format(len(dta))
print "jenks(dta, {0})".format(N_CLASSES)

breaks = jenks(dta, N_CLASSES)
GCF, class_deviations = getQualityMetrics(dta, breaks, N_CLASSES)

print breaks

print "GCF: {0}".format(GCF)
print "Class deviations: {0}".format(class_deviations)


result = classifyData(dta, breaks, class_deviations, N_CLASSES)
print "11asdaa11"
print result

assignments, SDCM = classifyData(dta, breaks, class_deviations, N_CLASSES)

print assignments
print SDCM