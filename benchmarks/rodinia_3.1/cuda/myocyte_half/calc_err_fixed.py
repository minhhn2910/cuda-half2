#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import os.path
def read_target(target_file):

				# format a1,a2,a3...

				list_target = []
				with open(target_file) as conf_file:
								for line in conf_file:
												line.replace(' ', '')

												# remove unexpected space

												array = line.split(',')
												for target in array:
																try:
																				if len(target) > 0 and target != '\n':
																								list_target.append(float(target))
																except:
																				print 'Failed to parse target file'
				return list_target

def check_output(floating_result, target_result):

# TODO: modify this func to return checksum error. instead of true and false. feed the checsum error to greedy decision func

	if len(floating_result) != len(target_result):
								print 'Error : floating result has length: %s while target_result has length: %s' \
												% (len(floating_result), len(target_result))
								print floating_result
								return 0.00
				signal_sqr = 0.00
				error_sqr = 0.00
				sum_rel_err = 0.0
				for i in range(len(floating_result)):
					#   signal_sqr += target_result[i] ** 2
					#   error_sqr += (floating_result[i] - target_result[i]) ** 2
		if(target_result[i]!= 0): 
			sum_rel_err = sum_rel_err + abs((floating_result[i] - target_result[i])/target_result[i])
		else:
			sum_rel_err = sum_rel_err + abs(floating_result[i])
	
								# ~ print error_sqr

	return sum_rel_err/len(floating_result)
				#sqnr = 0.00
				#if error_sqr != 0.00:
				#    sqnr = signal_sqr / error_sqr
				#if sqnr != 0:
				#    return 1.0 / sqnr
				#else:
				#    return 0.00

def parse_output(line):
				list_target = []
				line.replace(' ', '')
				line.replace('\n', '')

								# remove unexpected space

				array = line.split(',')

#       print array

				for target in array:
								try:
												if len(target) > 0 and target != '\n':
																list_target.append(float(target))
								except:

																								# print "Failed to parse output string"

												continue

#               print list_target

				return list_target
def main(argv):
	if len(argv != 2):
		print "usage ./calc_err.py target_file approximate_file"
		exit()
	target = read_target(argv[0])
	approx_val = read_target(argv[1])
	print check_output(approx_val, target)
if __name__ == '__main__':
				main(sys.argv[1:])
