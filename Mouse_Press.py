import os
import os.path
import glob

#sad=0 (-inf,-1]
#happy=2 [1,inf)
#neutral=1

def double_click(time_list):
	double_click=0
	Double=[]
	if len(time_list)==0:
		return 0,0
	last_time=time_list[0]
	time_list=time_list[1:]
	for i in time_list:
		if i-last_time<=200:
			double_click+=1
			if last_time not in Double:
				Double.append(last_time)
			Double.append(i)
		last_time=i
	return len(time_list)-len(Double),len(Double)


def average_scroll_time(scroll_time_list):
	time=0.0
	if len(scroll_time_list)==0:
		return 0
	for x in scroll_time_list:
		time+=x
	return time/len(scroll_time_list)

feature=[]
labels=[]
#sad
sad_files=glob.glob('/E/Work/Data/Mouse Database/Emotional/Sad/*.txt')
for f in sad_files:
	file=open(f,'rb')
	List=[]
	Time=0
	click=0
	right_click=0
	left_click=0
	count=-9
	time_last=0.0
	total_time_elapsed=0.0
	left_time_elapsed=0.0
	right_time_elapsed=0.0
	scroll0_count=0.0
	scroll0_time=0.0
	scroll1_count=0.0
	scroll1_time=0.0
	scroll2_count=0.0
	scroll2_time=0.0
	scroll0_time_list=[]
	scroll1_time_list=[]
	scroll2_time_list=[]
	right_click_time=[]
	left_click_time=[]
	x_cod = []
	y_cod = []
	drag_x_cod=[]
	drag_y_cod=[]
	time = []
	drag_time=[]
	speedX = []
	speedY = []
	accX = []
	accY = []
	drag_speedX = []
	drag_speedY = []
	drag_accX = []
	drag_accY = []
	for line in file:
		words=line.split(',')	
		if count!=0 and count>0:
			try:
				Time+=long(words[len(words)-1])
			except:
				pass
		#count+=1
		if(words[0]=="MM"):
					x_cod.append(float(words[1]))
					y_cod.append(float(words[2]))
					time.append(float(words[3]))
		if(words[0]=="MD"):
					drag_x_cod.append(float(words[1]))
					drag_y_cod.append(float(words[2]))
					drag_time.append(float(words[3]))
		if words[0]=='MP' or words[0]=='MR' or words[0]=='MC' or words[0]=='MWM':
			List.append(words)
		if words[0]=='MC':
			if int(words[1])==1:
				if List[len(List)-2][0]=='MR':
					left_click+=1
					left_time_elapsed+=int(List[len(List)-2][2])
					left_click_time.append(Time-int(List[len(List)-2][2]))
			else:
				if List[len(List)-2][0]=='MR':
					right_click+=1
					right_time_elapsed+=int(List[len(List)-2][2])
					right_click_time.append(Time-int(List[len(List)-2][2]))
			click+=1
		if words[0]=='MWM':
			if int(words[4])==0:
				if scroll0_count==0:		
					scroll0_time+=0
					scroll0_count+=1
				else:
					scroll0_time+=int(words[6])
					scroll0_count+=1
			if int(words[4])==1:
				if scroll1_count==0:		
					scroll1_time+=0
					scroll1_count+=1
				else:
					scroll1_time+=int(words[6])
					scroll1_count+=1
			if int(words[4])==-1:
				if scroll2_count==0:		
					scroll2_time+=0
					scroll2_count+=1
				else:
					scroll2_time+=int(words[6])
					scroll2_count+=1
		else:
			scroll0_time_list.append(scroll0_time)
			scroll1_time_list.append(scroll1_time)
			scroll2_time_list.append(scroll2_time)
			scroll0_count=0.0
			scroll0_time=0.0
			scroll1_count=0.0
			scroll1_time=0.0
			scroll2_count=0.0
			scroll2_time=0.0
		if count%30==0 and count!=0 and count>0:
			TIME=Time-time_last
			right_click_frequency=right_click/TIME
			left_click_frequency=left_click/TIME
			try:
				right_time_elapsed=right_time_elapsed/right_click
			except:
				right_time_elapsed=0
			try:
				left_time_elapsed=left_time_elapsed/left_click
			except:
				left_time_elapsed=0
			Click=click
			left_single_click,left_double_click=double_click(left_click_time)
			right_single_click,right_double_click=double_click(right_click_time)
			#print len(time)
			#time[0]=0
			if len(time)>0:
				time[0]=0
				for index in range(len(x_cod)-1):
				    if(time[index+1]!=0):
				        speedX.append((x_cod[index+1]-x_cod[index])/time[index+1])
				        

				for index in range(len(y_cod)-1):
				    if(time[index+1]!=0  ):
				        speedY.append( (x_cod[index+1]-x_cod[index])/time[index+1])  
				        
				for index in range(len(speedX)-1):
				    if(time[index+1]!=0):
				        accX.append((speedX[index+1]-speedX[index])/time[index+1])
				        

				for index in range(len(speedY)-1):
				    if(time[index+1]!=0  ):
				        accY.append( (speedY[index+1]-speedY[index])/time[index+1])

			SpeedX_avg=average_scroll_time(speedX)
			SpeedY_avg=average_scroll_time(speedY)
			AccX_avg=average_scroll_time(accX)
			AccY_avg=average_scroll_time(accY)    

			if len(drag_time)>0:
				drag_time[0]=0
				for index in range(len(drag_x_cod)-1):
				    if(drag_time[index+1]!=0):
				        drag_speedX.append((drag_x_cod[index+1]-drag_x_cod[index])/drag_time[index+1])
				        

				for index in range(len(drag_y_cod)-1):
				    if(drag_time[index+1]!=0  ):
				        drag_speedY.append( (drag_x_cod[index+1]-drag_x_cod[index])/drag_time[index+1])  
				        
				for index in range(len(drag_speedX)-1):
				    if(drag_time[index+1]!=0):
				        drag_accX.append((drag_speedX[index+1]-drag_speedX[index])/drag_time[index+1])
				        

				for index in range(len(drag_speedY)-1):
				    if(drag_time[index+1]!=0):
				        drag_accY.append( (drag_speedY[index+1]-drag_speedY[index])/drag_time[index+1])

			drag_SpeedX_avg=average_scroll_time(speedX)
			drag_SpeedY_avg=average_scroll_time(speedY)
			drag_AccX_avg=average_scroll_time(accX)
			drag_AccY_avg=average_scroll_time(accY)
			node=[]
			node.append(TIME)
			node.append(click)
			node.append(right_click_frequency)
			node.append(right_time_elapsed)
			node.append(right_single_click)
			node.append(right_double_click)
			node.append(left_click_frequency)
			node.append(left_time_elapsed)
			node.append(left_single_click)
			node.append(left_double_click)
			node.append(average_scroll_time(scroll0_time_list))
			node.append(average_scroll_time(scroll1_time_list))
			node.append(average_scroll_time(scroll2_time_list))
			node.append(SpeedX_avg)
			node.append(SpeedY_avg)
			node.append(AccX_avg)
			node.append(AccY_avg)
			node.append(drag_SpeedX_avg)
			node.append(drag_SpeedY_avg)
			node.append(drag_AccX_avg)
			node.append(drag_AccY_avg)
			feature.append(node)
			labels.append(0)
			right_click_time=[]
			left_click_time=[]
			List=[]
			click=0
			right_click=0
			left_click=0
			count=0
			time_last=Time
			total_time_elapsed=0.0
			left_time_elapsed=0.0
			right_time_elapsed=0.0
			scroll0_time_list=[]
			scroll1_time_list=[]
			scroll2_time_list=[]
			scroll0_count=0.0
			scroll0_time=0.0
			scroll1_count=0.0
			scroll1_time=0.0
			scroll2_count=0.0
			scroll2_time=0.0
			x_cod = []
			y_cod = []
			drag_x_cod=[]
			drag_y_cod=[]
			time = []
			drag_time=[]
			speedX = []
			speedY = []
			accX = []
			accY = []
			drag_speedX = []
			drag_speedY = []
			drag_accX = []
			drag_accY = []
		count+=1
print feature
print labels


happy_files=glob.glob('/E/Work/Data/Mouse Database/Emotional/Happy/*.txt')
for f in happy_files:
	file=open(f,'rb')
	List=[]
	Time=0
	click=0
	right_click=0
	left_click=0
	count=-9
	time_last=0.0
	total_time_elapsed=0.0
	left_time_elapsed=0.0
	right_time_elapsed=0.0
	scroll0_count=0.0
	scroll0_time=0.0
	scroll1_count=0.0
	scroll1_time=0.0
	scroll2_count=0.0
	scroll2_time=0.0
	scroll0_time_list=[]
	scroll1_time_list=[]
	scroll2_time_list=[]
	right_click_time=[]
	left_click_time=[]
	x_cod = []
	y_cod = []
	drag_x_cod=[]
	drag_y_cod=[]
	time = []
	drag_time=[]
	speedX = []
	speedY = []
	accX = []
	accY = []
	drag_speedX = []
	drag_speedY = []
	drag_accX = []
	drag_accY = []
	for line in file:
		words=line.split(',')	
		if count!=0 and count>0:
			try:
				Time+=long(words[len(words)-1])
			except:
				pass
		#count+=1
		if(words[0]=="MM"):
					x_cod.append(float(words[1]))
					y_cod.append(float(words[2]))
					time.append(float(words[3]))
		if(words[0]=="MD"):
					drag_x_cod.append(float(words[1]))
					drag_y_cod.append(float(words[2]))
					drag_time.append(float(words[3]))
		if words[0]=='MP' or words[0]=='MR' or words[0]=='MC' or words[0]=='MWM':
			List.append(words)
		if words[0]=='MC':
			if int(words[1])==1:
				if List[len(List)-2][0]=='MR':
					left_click+=1
					left_time_elapsed+=int(List[len(List)-2][2])
					left_click_time.append(Time-int(List[len(List)-2][2]))
			else:
				if List[len(List)-2][0]=='MR':
					right_click+=1
					right_time_elapsed+=int(List[len(List)-2][2])
					right_click_time.append(Time-int(List[len(List)-2][2]))
			click+=1
		if words[0]=='MWM':
			if int(words[4])==0:
				if scroll0_count==0:		
					scroll0_time+=0
					scroll0_count+=1
				else:
					scroll0_time+=int(words[6])
					scroll0_count+=1
			if int(words[4])==1:
				if scroll1_count==0:		
					scroll1_time+=0
					scroll1_count+=1
				else:
					scroll1_time+=int(words[6])
					scroll1_count+=1
			if int(words[4])==-1:
				if scroll2_count==0:		
					scroll2_time+=0
					scroll2_count+=1
				else:
					scroll2_time+=int(words[6])
					scroll2_count+=1
		else:
			scroll0_time_list.append(scroll0_time)
			scroll1_time_list.append(scroll1_time)
			scroll2_time_list.append(scroll2_time)
			scroll0_count=0.0
			scroll0_time=0.0
			scroll1_count=0.0
			scroll1_time=0.0
			scroll2_count=0.0
			scroll2_time=0.0
		if count%30==0 and count!=0 and count>0:
			TIME=Time-time_last
			right_click_frequency=right_click/TIME
			left_click_frequency=left_click/TIME
			try:
				right_time_elapsed=right_time_elapsed/right_click
			except:
				right_time_elapsed=0
			try:
				left_time_elapsed=left_time_elapsed/left_click
			except:
				left_time_elapsed=0
			Click=click
			left_single_click,left_double_click=double_click(left_click_time)
			right_single_click,right_double_click=double_click(right_click_time)
			#print len(time)
			#time[0]=0
			if len(time)>0:
				time[0]=0
				for index in range(len(x_cod)-1):
				    if(time[index+1]!=0):
				        speedX.append((x_cod[index+1]-x_cod[index])/time[index+1])
				        

				for index in range(len(y_cod)-1):
				    if(time[index+1]!=0  ):
				        speedY.append( (x_cod[index+1]-x_cod[index])/time[index+1])  
				        
				for index in range(len(speedX)-1):
				    if(time[index+1]!=0):
				        accX.append((speedX[index+1]-speedX[index])/time[index+1])
				        

				for index in range(len(speedY)-1):
				    if(time[index+1]!=0  ):
				        accY.append( (speedY[index+1]-speedY[index])/time[index+1])

			SpeedX_avg=average_scroll_time(speedX)
			SpeedY_avg=average_scroll_time(speedY)
			AccX_avg=average_scroll_time(accX)
			AccY_avg=average_scroll_time(accY)    

			if len(drag_time)>0:
				drag_time[0]=0
				for index in range(len(drag_x_cod)-1):
				    if(drag_time[index+1]!=0):
				        drag_speedX.append((drag_x_cod[index+1]-drag_x_cod[index])/drag_time[index+1])
				        

				for index in range(len(drag_y_cod)-1):
				    if(drag_time[index+1]!=0  ):
				        drag_speedY.append( (drag_x_cod[index+1]-drag_x_cod[index])/drag_time[index+1])  
				        
				for index in range(len(drag_speedX)-1):
				    if(drag_time[index+1]!=0):
				        drag_accX.append((drag_speedX[index+1]-drag_speedX[index])/drag_time[index+1])
				        

				for index in range(len(drag_speedY)-1):
				    if(drag_time[index+1]!=0):
				        drag_accY.append( (drag_speedY[index+1]-drag_speedY[index])/drag_time[index+1])

			drag_SpeedX_avg=average_scroll_time(speedX)
			drag_SpeedY_avg=average_scroll_time(speedY)
			drag_AccX_avg=average_scroll_time(accX)
			drag_AccY_avg=average_scroll_time(accY)
			node=[]
			node.append(TIME)
			node.append(click)
			node.append(right_click_frequency)
			node.append(right_time_elapsed)
			node.append(right_single_click)
			node.append(right_double_click)
			node.append(left_click_frequency)
			node.append(left_time_elapsed)
			node.append(left_single_click)
			node.append(left_double_click)
			node.append(average_scroll_time(scroll0_time_list))
			node.append(average_scroll_time(scroll1_time_list))
			node.append(average_scroll_time(scroll2_time_list))
			node.append(SpeedX_avg)
			node.append(SpeedY_avg)
			node.append(AccX_avg)
			node.append(AccY_avg)
			node.append(drag_SpeedX_avg)
			node.append(drag_SpeedY_avg)
			node.append(drag_AccX_avg)
			node.append(drag_AccY_avg)
			feature.append(node)
			labels.append(2)
			right_click_time=[]
			left_click_time=[]
			List=[]
			click=0
			right_click=0
			left_click=0
			count=0
			time_last=Time
			total_time_elapsed=0.0
			left_time_elapsed=0.0
			right_time_elapsed=0.0
			scroll0_time_list=[]
			scroll1_time_list=[]
			scroll2_time_list=[]
			scroll0_count=0.0
			scroll0_time=0.0
			scroll1_count=0.0
			scroll1_time=0.0
			scroll2_count=0.0
			scroll2_time=0.0
			x_cod = []
			y_cod = []
			drag_x_cod=[]
			drag_y_cod=[]
			time = []
			drag_time=[]
			speedX = []
			speedY = []
			accX = []
			accY = []
			drag_speedX = []
			drag_speedY = []
			drag_accX = []
			drag_accY = []
		count+=1
print feature
print labels

happy_files=glob.glob('/E/Work/Data/Mouse Database/Neutral/*.txt')
for f in happy_files:
	file=open(f,'rb')
	List=[]
	Time=0
	click=0
	right_click=0
	left_click=0
	count=-9
	time_last=0.0
	total_time_elapsed=0.0
	left_time_elapsed=0.0
	right_time_elapsed=0.0
	scroll0_count=0.0
	scroll0_time=0.0
	scroll1_count=0.0
	scroll1_time=0.0
	scroll2_count=0.0
	scroll2_time=0.0
	scroll0_time_list=[]
	scroll1_time_list=[]
	scroll2_time_list=[]
	right_click_time=[]
	left_click_time=[]
	x_cod = []
	y_cod = []
	drag_x_cod=[]
	drag_y_cod=[]
	time = []
	drag_time=[]
	speedX = []
	speedY = []
	accX = []
	accY = []
	drag_speedX = [] 	
	drag_speedY = []
	drag_accX = []
	drag_accY = []
	for line in file:
		words=line.split(',')	
		if count!=0 and count>0:
			try:
				Time+=long(words[len(words)-1])
			except:
				pass
		#count+=1
		if(words[0]=="MM"):
					x_cod.append(float(words[1]))
					y_cod.append(float(words[2]))
					time.append(float(words[3]))
		if(words[0]=="MD"):
					drag_x_cod.append(float(words[1]))
					drag_y_cod.append(float(words[2]))
					drag_time.append(float(words[3]))
		if words[0]=='MP' or words[0]=='MR' or words[0]=='MC' or words[0]=='MWM':
			List.append(words)
		if words[0]=='MC':
			if int(words[1])==1:
				if List[len(List)-2][0]=='MR':
					left_click+=1
					left_time_elapsed+=int(List[len(List)-2][2])
					left_click_time.append(Time-int(List[len(List)-2][2]))
			else:
				if List[len(List)-2][0]=='MR':
					right_click+=1
					right_time_elapsed+=int(List[len(List)-2][2])
					right_click_time.append(Time-int(List[len(List)-2][2]))
			click+=1
		if words[0]=='MWM':
			if int(words[4])==0:
				if scroll0_count==0:		
					scroll0_time+=0
					scroll0_count+=1
				else:
					scroll0_time+=int(words[6])
					scroll0_count+=1
			if int(words[4])==1:
				if scroll1_count==0:		
					scroll1_time+=0
					scroll1_count+=1
				else:
					scroll1_time+=int(words[6])
					scroll1_count+=1
			if int(words[4])==-1:
				if scroll2_count==0:		
					scroll2_time+=0
					scroll2_count+=1
				else:
					scroll2_time+=int(words[6])
					scroll2_count+=1
		else:
			scroll0_time_list.append(scroll0_time)
			scroll1_time_list.append(scroll1_time)
			scroll2_time_list.append(scroll2_time)
			scroll0_count=0.0
			scroll0_time=0.0
			scroll1_count=0.0
			scroll1_time=0.0
			scroll2_count=0.0
			scroll2_time=0.0
		if count%30==0 and count!=0 and count>0:
			TIME=Time-time_last
			right_click_frequency=right_click/TIME
			left_click_frequency=left_click/TIME
			try:
				right_time_elapsed=right_time_elapsed/right_click
			except:
				right_time_elapsed=0
			try:
				left_time_elapsed=left_time_elapsed/left_click
			except:
				left_time_elapsed=0
			Click=click
			left_single_click,left_double_click=double_click(left_click_time)
			right_single_click,right_double_click=double_click(right_click_time)
			#print len(time)
			#time[0]=0
			if len(time)>0:
				time[0]=0
				for index in range(len(x_cod)-1):
				    if(time[index+1]!=0):
				        speedX.append((x_cod[index+1]-x_cod[index])/time[index+1])
				        

				for index in range(len(y_cod)-1):
				    if(time[index+1]!=0  ):
				        speedY.append( (x_cod[index+1]-x_cod[index])/time[index+1])  
				        
				for index in range(len(speedX)-1):
				    if(time[index+1]!=0):
				        accX.append((speedX[index+1]-speedX[index])/time[index+1])
				        

				for index in range(len(speedY)-1):
				    if(time[index+1]!=0  ):
				        accY.append( (speedY[index+1]-speedY[index])/time[index+1])

			SpeedX_avg=average_scroll_time(speedX)
			SpeedY_avg=average_scroll_time(speedY)
			AccX_avg=average_scroll_time(accX)
			AccY_avg=average_scroll_time(accY)    

			if len(drag_time)>0:
				drag_time[0]=0
				for index in range(len(drag_x_cod)-1):
				    if(drag_time[index+1]!=0):
				        drag_speedX.append((drag_x_cod[index+1]-drag_x_cod[index])/drag_time[index+1])
				        

				for index in range(len(drag_y_cod)-1):
				    if(drag_time[index+1]!=0  ):
				        drag_speedY.append( (drag_x_cod[index+1]-drag_x_cod[index])/drag_time[index+1])  
				        
				for index in range(len(drag_speedX)-1):
				    if(drag_time[index+1]!=0):
				        drag_accX.append((drag_speedX[index+1]-drag_speedX[index])/drag_time[index+1])
				        

				for index in range(len(drag_speedY)-1):
				    if(drag_time[index+1]!=0):
				        drag_accY.append( (drag_speedY[index+1]-drag_speedY[index])/drag_time[index+1])

			drag_SpeedX_avg=average_scroll_time(speedX)
			drag_SpeedY_avg=average_scroll_time(speedY)
			drag_AccX_avg=average_scroll_time(accX)
			drag_AccY_avg=average_scroll_time(accY)
			node=[]
			node.append(TIME)
			node.append(click)
			node.append(right_click_frequency)
			node.append(right_time_elapsed)
			node.append(right_single_click)
			node.append(right_double_click)
			node.append(left_click_frequency)
			node.append(left_time_elapsed)
			node.append(left_single_click)
			node.append(left_double_click)
			node.append(average_scroll_time(scroll0_time_list))
			node.append(average_scroll_time(scroll1_time_list))
			node.append(average_scroll_time(scroll2_time_list))
			node.append(SpeedX_avg)
			node.append(SpeedY_avg)
			node.append(AccX_avg)
			node.append(AccY_avg)
			node.append(drag_SpeedX_avg)
			node.append(drag_SpeedY_avg)
			node.append(drag_AccX_avg)
			node.append(drag_AccY_avg)
			feature.append(node)
			labels.append(1)
			right_click_time=[]
			left_click_time=[]
			List=[]
			click=0
			right_click=0
			left_click=0
			count=0
			time_last=Time
			total_time_elapsed=0.0
			left_time_elapsed=0.0
			right_time_elapsed=0.0
			scroll0_time_list=[]
			scroll1_time_list=[]
			scroll2_time_list=[]
			scroll0_count=0.0
			scroll0_time=0.0
			scroll1_count=0.0
			scroll1_time=0.0
			scroll2_count=0.0
			scroll2_time=0.0
			x_cod = []
			y_cod = []
			drag_x_cod=[]
			drag_y_cod=[]
			time = []
			drag_time=[]
			speedX = []
			speedY = []
			accX = []
			accY = []
			drag_speedX = []
			drag_speedY = []
			drag_accX = []
			drag_accY = []
		count+=1
print feature
print labels

X_train=[]
X_test=[]
y_train=[]
y_test=[]
for i in range(len(feature)):
	if i%3==0:
		X_test.append(feature[i])
		y_test.append(labels[i])
	else:
		X_train.append(feature[i])
		y_train.append(labels[i])

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
import pandas as pd

# X_train=np.array(X_train)
# y_train=np.array(y_train).reshape((len(y_train),1))

#data=np.concatenate((X_train,y_train),axis=1)
# plt.scatter(X_train[:,19],y_train)
# plt.show()


def make_ellipses(gmm, ax):
    for n, color in enumerate('rgb'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

#iris = datasets.load_iris()

# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
#skf = StratifiedKFold(iris.target, n_folds=4)
# Only take the first fold.
#train_index, test_index = next(iter(skf))


X_train =np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print X_train.shape[0],X_train.shape[1]

n_classes =len(np.unique(y_train))

# Try GMMs using different types of covariances.
classifiers = dict((covar_type, GMM(n_components=n_classes,
                    covariance_type=covar_type, init_params='wc', n_iter=20))
                   for covar_type in ['spherical', 'diag', 'tied', 'full'])

n_classifiers = len(classifiers)

plt.figure(figsize=(3 * n_classifiers / 2, 6))
plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                    left=.01, right=.99)


target_names=['sad','neutral','happy']

for index, (name, classifier) in enumerate(classifiers.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                  for i in range(n_classes)])

    # Train the other parameters using the EM algorithm.
    classifier.fit(X_train)

    h = plt.subplot(2, n_classifiers / 2, index + 1)
    #make_ellipses(classifier, h)

    # for n, color in enumerate('rgb'):
    #     data = data[target== n for target in target_names]
    #     plt.scatter(data[:, 0], data[:, 1], 0.8, color=color,
    #                 label=target_names[n])
    #Plot the test data with crosses
    for n, color in enumerate('rgb'):
        data = X_test[y_test == n]
        plt.plot(data[:, 0], data[:, 1], 'x', color=color)

    y_train_pred = classifier.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
             transform=h.transAxes)

    y_test_pred = classifier.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
             transform=h.transAxes)

    plt.xticks(())
    plt.yticks(())
    plt.title(name)

plt.legend(loc='lower right', prop=dict(size=12))


plt.show()
