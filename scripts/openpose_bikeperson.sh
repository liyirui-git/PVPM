#!/bin/bash -x
###
 # @Author: Li, Yirui
 # @Date: 2021-06-07
 # @Description: Add display hyper-parameter in bash command, to ban displaying in display.
 # @FilePath: /liyirui/PycharmProjects/PVPM/scripts/openpose_bikeperson.sh
### 
cd /home/liyirui/PycharmProjects/openpose/
path=/home/liyirui/PycharmProjects/PVPM/dataset/BikePerson-700-origin/

image_dir=bounding_box_train/
output_dir=bounding_box_pose_train

./build/examples/openpose/openpose.bin \
--image_dir ${path}${image_dir} \
--write_images ${path}${output_dir} \
--model_pose COCO \
--write_json ${path}${output_dir} \
--heatmaps_add_parts true \
--heatmaps_add_PAFs true \
--write_heatmaps ${path}${output_dir} \
--net_resolution -1x384 \
--display 0

image_dir=bounding_box_test/
output_dir=bounding_box_pose_test

./build/examples/openpose/openpose.bin \
--image_dir ${path}${image_dir} \
--write_images ${path}${output_dir} \
--model_pose COCO \
--write_json ${path}${output_dir} \
--heatmaps_add_parts true \
--heatmaps_add_PAFs true \
--write_heatmaps ${path}${output_dir} \
--net_resolution -1x384 \
--display 0

image_dir=query/
output_dir=query_pose

./build/examples/openpose/openpose.bin \
--image_dir ${path}${image_dir} \
--write_images ${path}${output_dir} \
--model_pose COCO \
--write_json ${path}${output_dir} \
--heatmaps_add_parts true \
--heatmaps_add_PAFs true \
--write_heatmaps ${path}${output_dir} \
--net_resolution -1x384 \
--display 0
