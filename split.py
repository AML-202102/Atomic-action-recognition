import json
import numpy as np 
import random   

for j in range(1,3):
	file = open('annotations/RobotSegSantaFe_v2_fold{}.json'.format(j))
	file_load = json.load(file)

	train = {
    	"info": file_load["info"],
    	"licenses": file_load["licenses"],
    	"categories": file_load["categories"],
    	"action_categories": file_load["action_categories"],
    	"phase_categories": file_load["phase_categories"],
    	"step_categories": file_load["step_categories"],
    	"images": [], 
    	"annotations": []}

	test = {
    	"info": file_load["info"],
    	"licenses": file_load["licenses"],
    	"categories": file_load["categories"],
    	"action_categories": file_load["action_categories"],
    	"phase_categories": file_load["phase_categories"],
    	"step_categories": file_load["step_categories"],
    	"images": [], 
    	"annotations": []}

	cases = []
	for i, image_dic in enumerate(file_load.get('images')):
		image_name = image_dic.get('file_name')
		case = image_name.split('/')[0]
		if case not in cases:
			cases.append(case)

	print("Cases in fold{}:".format(j), cases)

	#Maybe better to define the test_case so that experiences are repetible
	test_case = random.choice(cases)

	print("Random test case for fold{}:".format(j), test_case)

	for i, image_dic in enumerate(file_load.get('images')):
		image_name = image_dic.get('file_name')
		case = image_name.split('/')[0]
		if case != test_case:
			image_info = image_dic
			train['images'].append(image_info)
		if case == test_case:
			image_info = image_dic
			test['images'].append(image_info)


	for i, annot_dic in enumerate(file_load.get('annotations')):
		image_name = annot_dic.get('image_name')
		case = image_name.split('/')[0]
		if case != test_case:
			annot_info = annot_dic
			train['annotations'].append(annot_info)
		if case == test_case:
			annot_info = annot_dic
			test['annotations'].append(annot_info)

	out_path_train = 'train_fold{}.json'.format(j)
	json.dump(train, open(out_path_train, 'w'))   

	out_path_test = 'test_fold{}.json'.format(j)
	json.dump(test, open(out_path_test, 'w'))   
