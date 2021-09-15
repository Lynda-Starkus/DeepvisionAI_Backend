import argparse
import frames2video
import os
from os import listdir, mkdir
from os.path import isfile, isdir, join
import shutil



parser = argparse.ArgumentParser(description = "The tracking and abnormal detection script")
parser.add_argument('-tracking', action='store_true', help='activates tracking od peds')
parser.add_argument('-test', type = int, help='choose the number of files to be tested', default=36)
args = parser.parse_args()

#Run utils to reorganize and label the chosen UCSD dataset samples
os.system('python utils.py -test '+str(args.test))

#Launch the test_script that uses the pretrained model 
os.system('python test_script.py')

#Output folders will contain all treated frames as images 
dir_test = [f for f in listdir('output_folders/') if isdir(join('output_folders/', f))]
dir_test.sort()

#Force folder delete 
shutil.rmtree("output_without_tracking", ignore_errors = True)
shutil.rmtree("output", ignore_errors = True)

os.mkdir('output_without_tracking/')
i = 1
for directory in dir_test:
    #Convert the output folders of test_script.py to .avi videos 
    frames2video.convert_frames_to_video('output_folders'+'/'+directory, 'output_without_tracking/'+directory+'.avi')

    #Run the tracking of all pedestrians and output the final video containing both abnormality detection and pedestrians tracking
    os.system('python track.py --source '+'output_without_tracking/'+directory+'.avi '+'--classes 0 --save-vid --device cpu --output output/test_video'+str(i))
    i+=1


