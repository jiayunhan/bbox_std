
import pdb

def save_detection_to_file(input_dic, output_file, task):
    with open(output_file, 'w') as txt_file:
        if not input_dic:
            return
        class_list = input_dic['classes']
        bbox_list = input_dic['boxes']
        scores_list = input_dic['scores']
        for temp_class, temp_bbox, temp_score in zip(class_list, bbox_list, scores_list):
            top, left, bottom, right = temp_bbox
            if task == 'ground_truth':
                txt_file.write(temp_class + " " + str(int(left)) + " " + str(int(top)) + " " + str(int(right)) + " " + str(int(bottom)) + '\n')
            elif task == 'detection':
                txt_file.write(temp_class + " " + str(temp_score) + " " + str(int(left)) + " " + str(int(top)) + " " + str(int(right)) + " " + str(int(bottom)) + '\n')
            else:
                raise ValueError('Invalid task.')
