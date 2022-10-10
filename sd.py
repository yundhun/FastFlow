import cv2
import os
import random
import scipy
import numpy
import timm
import dataset

'''
def defetcGen(dataset_path, class_name):
	#1. simple defect generation
	data_dir_train_good = os.path.join(dataset_path, class_name, 'train/good')
	data_dir_train_fake_ng = os.path.join(dataset_path, class_name, 'train_fake_ng/fake_ng')
	if(os.path.isdir(data_dir_train_fake_ng) is False):
		os.makedirs(data_dir_train_fake_ng)

	onlyfiles = [f for f in os.listdir(data_dir_train_good) if os.path.isfile( os.path.join(data_dir_train_good, f))] 
	for file in onlyfiles:
		source_file_dir = os.path.join(data_dir_train_good,file)
		dest_file_dir   = os.path.join(data_dir_train_fake_ng,file)
		img1 = cv2.imread(source_file_dir)
		
		#get defect size
		defect_size_max = round(img1.shape[0] * 0.3) #30%
		defect_size_min = round(img1.shape[0] * 0.05) #5%
		h,w = random.sample(range(defect_size_min,defect_size_max),2)
		#get defect position
		x,y,x2,y2 = random.sample(range(0, (img1.shape[0]-max(h,w))),4)
		tmp = img1[x:(x+w),y:(y+h),:]
		img1[x2:(x2+w),y2:(y2+h),:] = tmp
		cv2.imwrite(dest_file_dir, img1)
		print('src:', source_file_dir, 'dest:', dest_file_dir ,'shape', img1.shape)	
'''

def get_sd(backbone_name, n_feat_sd, train_loader, train_ng_loader):
	#3. get features
	feature_extractor = timm.create_model(backbone_name,pretrained=True,features_only=True,out_indices=[1, 2, 3])	
	channels = feature_extractor.feature_info.channels()
	scales = feature_extractor.feature_info.reduction()
	model = feature_extractor.to('mps')	
	model.eval()

	train_dataiter = iter(train_loader)
	fake_ng_train_dataiter = iter(train_ng_loader)

	dims_dic = {}
	dims_selected_dic = {}
	for x in range(len(train_loader)):
		inputs1 = next(train_dataiter)
		inputs2 = next(fake_ng_train_dataiter)[0]
		inputs1 = inputs1.to('mps')
		inputs2 = inputs2.to('mps')
		print('c.feature_sd:', n_feat_sd, 'inputs1.shape:', inputs1.shape)#, 'features1.shape:',features1.shape)
		print('c.feature_sd:', n_feat_sd, 'inputs2.shape:', inputs2.shape)#, 'features2.shape:',features2.shape)
		features1 = model(inputs1)
		features2 = model(inputs2)

		print("=======[features1]========")
		print("len(features1):",len(features1))
		print(features1[0].shape)
		print(features1[1].shape)
		print(features1[2].shape)
		print("=======[features2]========")
		print("len(features2):",len(features2))
		print(features2[0].shape)
		print(features2[1].shape)
		print(features2[2].shape)

		for feature_layer_idx in range(len(features1)):
			batch_size = features1[feature_layer_idx].shape[0]
			full_dim_size = features1[feature_layer_idx].shape[1]
			for i in range(batch_size):
				for j in range(full_dim_size):
					key_name = str(feature_layer_idx)+'-'+str(j)
					f1 = features1[feature_layer_idx][i,j,:,:].cpu().detach().numpy().reshape(-1)
					f2 = features2[feature_layer_idx][i,j,:,:].cpu().detach().numpy().reshape(-1)
					#print('f1==f2:',sum(f1==f2),'f1.shape:',f1.shape)
					pvalue = scipy.stats.ttest_ind(f1, f2, equal_var=False)
					#print('data index:',i,'dim no:',j,'pvalue:',pvalue.pvalue,'feature1.shape:',features1[i,j,:,:].shape,'f1.shape:',f1.shape)
					
					if j in dims_dic:
						dims_dic[key_name] = dims_dic[key_name] + [pvalue.pvalue]
					else:
						dims_dic[key_name] = [pvalue.pvalue]
					
					if pvalue.pvalue <= 0.05:
						if key_name in dims_selected_dic:
							dims_selected_dic[key_name] += 1
						else:
							dims_selected_dic[key_name] = 1

	sd_dic = dict(sorted(dims_selected_dic.items(), reverse=True, key=lambda item: item[1]))
	print(sd_dic)

	sd_list = []
	for x in sd_dic:
		sd_list = sd_list + [x]
	
	print('unsorted dim:',sd_list)
	sd2 = sd_list[:n_feat_sd]
	print('unsorted sd:',sd2)
	sd2.sort()
	print('sorted sd:',sd2)
	return sd2
