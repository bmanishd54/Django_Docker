from django.shortcuts import render,redirect
from .forms import usernameForm,DateForm,UsernameAndDateForm, DateForm_2,UserSelectionForm
from django.contrib import messages
from django.contrib.auth.models import User
import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner
import time
from attendance_system_facial_recognition.settings import BASE_DIR
from attendance_system_facial_recognition.settings import MEDIA_ROOT
import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from django.contrib.auth.decorators import login_required
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from django_pandas.io import read_frame
from users.models import Present, Time
import seaborn as sns
import pandas as pd
from django.db.models import Count
#import mpld3
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams
import math
import datetime
import numpy as np
import base64
from django.http import JsonResponse, StreamingHttpResponse, HttpResponseRedirect
from django.urls import reverse
import logging
import requests

mpl.use('Agg')
@login_required
def list_employees(request):
    url = 'https://daily-inout.in/api/list-employee'
    token = '5d5db2ae77981234567891af538baa9045a84c0e889f672baf83ff24'
    headers = {
        'Authorization': f'{token}',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json().get('data', [])
            return render(request, 'recognition/list_employees.html', {'employees': data})  # Render HTML page
        else:
            return JsonResponse({'error': response.text}, status=response.status_code)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    

# views.py
def save_employee_and_update_auth_users(request):
    url = 'https://daily-inout.in/api/list-employee'
    headers = {
        'Authorization': '5d5db2ae77981234567891af538baa9045a84c0e889f672baf83ff24',
        'Content-Type': 'application/json',
    }

    # Fetch data from the API
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json().get('data', [])
        
        default_password = 'user@1234'

        # Loop through each employee and create/update the corresponding user in auth_user
        for emp in data:
            username = emp['userName']  # Use userName as the username
            email = f"{username}@example.com"  # Generate a dummy email

            # Create or update the user in the User model (auth_user)
            user, created = User.objects.update_or_create(
                username=username,
                defaults={
                    'first_name': emp['name'].split()[0],  # Set first name
                    'last_name': ' '.join(emp['name'].split()[1:]),  # Set last name
                    'email': email,
                }
            )
            
            # Set the password and save the user
            user.set_password(default_password)
            user.save()

        return JsonResponse({'status': 'success', 'message': 'Users saved and updated successfully!'})
    else:
        return JsonResponse({'status': 'error', 'message': 'Failed to fetch employee data from API.'})

def send_post_request_to_api(username,time, out):
    # URL for IN or OUT entry based on the `out` flag
    token = '5d5db2ae77981234567891af538baa9045a84c0e889f672baf83ff24'
    headers = {
        'Authorization': f'{token}',
    
    }
    if out:
        url = 'https://daily-inout.in/api/add-OutEntry'
    else:
        url = 'https://daily-inout.in/api/add-InEntry'
    
    payload = {
        "user": username,
        "time": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    response = requests.post(url, data=payload, headers=headers)
    if response.status_code == 200:
        print(f"Successfully sent {'OUT' if out else 'IN'} attendance data for {username}.")
    else:
        print(f"Failed to send {'OUT' if out else 'IN'} attendance data for {username}. "
              f"Status Code: {response.status_code}, Response: {response.text}")
        
        
#utility functions:
def username_present(username):
	if User.objects.filter(username=username).exists():
		return True

	return False

@login_required
def create_dataset(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        image_data = request.POST.get('image')
        sample_num = int(request.POST.get('sampleNum'))

        # Ensure the username directory exists
        user_folder = os.path.join(MEDIA_ROOT, 'training_dataset', username)
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)

        # Decode the base64 image data
        image_data = image_data.split(',')[1]  # Remove the data URL part
        image_data = base64.b64decode(image_data)
        np_image = np.frombuffer(image_data, dtype=np.uint8)
        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Load face detector and face aligner
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_5_face_landmarks.dat')
        fa = FaceAligner(predictor, desiredFaceWidth=96)

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame, 1)

        if len(faces) > 0:
            for face in faces:
                (x, y, w, h) = rect_to_bb(face)
                faceAligned = fa.align(frame, gray_frame, face)
                file_path = os.path.join(user_folder, f'{sample_num}.jpg')
                cv2.imwrite(file_path, faceAligned)

        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'error'})

@login_required
def process_images(request):
    if request.method == 'POST':
        # Add your image processing logic here if needed
        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'error'})

@login_required
def add_photos(request):
    # Only allow 'admin' to access this view
    if request.user.username != 'admin':
        return redirect('not-authorised')

    if request.method == 'POST':
        form = UserSelectionForm(request.POST)  # Use the updated form
        if form.is_valid():
            username = form.cleaned_data['username'].username  # Get the selected username

            # Assuming you have a function `username_present` to check if the username exists
            if username_present(username):
                messages.success(request, 'Dataset creation started. Please wait until the process completes.')
                return render(request, 'recognition/add_photos.html', {'form': form, 'username': username})
            else:
                messages.warning(request, 'No such username found. Please register employee first.')
                return redirect('dashboard')
    else:
        form = UserSelectionForm()  # Use the form with dropdown for usernames

    return render(request, 'recognition/add_photos.html', {'form': form})


def predict(face_aligned,svc,threshold=0.75):
	face_encodings=np.zeros((1,128))
	try:
		x_face_locations=face_recognition.face_locations(face_aligned)
		faces_encodings=face_recognition.face_encodings(face_aligned,known_face_locations=x_face_locations)
		if(len(faces_encodings)==0):
			return ([-1],[0])

	except:

		return ([-1],[0])

	prob=svc.predict_proba(faces_encodings)
	result=np.where(prob[0]==np.amax(prob[0]))
	if(prob[0][result[0]]<=threshold):
		return ([-1],prob[0][result[0]])

	return (result[0],prob[0][result[0]])


def vizualize_Data(embedded, targets,):

	X_embedded = TSNE(n_components=2).fit_transform(embedded)

	for i, t in enumerate(set(targets)):
		idx = targets == t
		plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

	plt.legend(bbox_to_anchor=(1, 1));
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	plt.savefig('./recognition/static/recognition/img/training_visualisation.png')
	plt.close()



def update_attendance_in_db_in(present):
	today=datetime.date.today()
	time=datetime.datetime.now()
	for person in present:
		user=User.objects.get(username=person)
		try:
		   qs=Present.objects.get(user=user,date=today)
		except :
			qs= None

		if qs is None:
			if present[person]==True:
						a=Present(user=user,username=person, date=today,present=True)
						a.save()
			else:
				a=Present(user=user,username=person,date=today,present=False)
				a.save()
		else:
			if present[person]==True:
				qs.present=True
				qs.save(update_fields=['present'])
		if present[person]==True:
			a=Time(user=user,username=person,date=today,time=time, out=False)
			a.save()
            # Send POST request to the API for IN entry
			send_post_request_to_api(person, time, out=False)



def update_attendance_in_db_out(present):
	today=datetime.date.today()
	time=datetime.datetime.now()
	for person in present:
		user=User.objects.get(username=person)
		if present[person]==True:
			a=Time(user=user,username=person,date=today,time=time, out=True)
			a.save()
            # Send POST request to the API for OUT entry
			send_post_request_to_api(person, time, out=True)
             




def check_validity_times(times_all):

	if(len(times_all)>0):
		sign=times_all.first().out
	else:
		sign=True
	times_in=times_all.filter(out=False)
	times_out=times_all.filter(out=True)
	if(len(times_in)!=len(times_out)):
		sign=True
	break_hourss=0
	if(sign==True):
			check=False
			break_hourss=0
			return (check,break_hourss)
	prev=True
	prev_time=times_all.first().time

	for obj in times_all:
		curr=obj.out
		if(curr==prev):
			check=False
			break_hourss=0
			return (check,break_hourss)
		if(curr==False):
			curr_time=obj.time
			to=curr_time
			ti=prev_time
			break_time=((to-ti).total_seconds())/3600
			break_hourss+=break_time


		else:
			prev_time=obj.time

		prev=curr

	return (True,break_hourss)


def convert_hours_to_hours_mins(hours):

	h=int(hours)
	hours-=h
	m=hours*60
	m=math.ceil(m)
	return str(str(h)+ " hrs " + str(m) + "  mins")



#used
def hours_vs_date_given_employee(present_qs,time_qs,admin=True):
	register_matplotlib_converters()
	df_hours=[]
	df_break_hours=[]
	qs=present_qs

	for obj in qs:
		date=obj.date
		times_in=time_qs.filter(date=date).filter(out=False).order_by('time')
		times_out=time_qs.filter(date=date).filter(out=True).order_by('time')
		times_all=time_qs.filter(date=date).order_by('time')
		obj.time_in=None
		obj.time_out=None
		obj.hours=0
		obj.break_hours=0
		if (len(times_in)>0):
			obj.time_in=times_in.first().time

		if (len(times_out)>0):
			obj.time_out=times_out.last().time

		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours


		else:
			obj.hours=0

		(check,break_hourss)= check_validity_times(times_all)
		if check:
			obj.break_hours=break_hourss


		else:
			obj.break_hours=0



		df_hours.append(obj.hours)
		df_break_hours.append(obj.break_hours)
		obj.hours=convert_hours_to_hours_mins(obj.hours)
		obj.break_hours=convert_hours_to_hours_mins(obj.break_hours)




	df = read_frame(qs)


	df["hours"]=df_hours
	df["break_hours"]=df_break_hours

	print(df)

	sns.barplot(data=df,x='date',y='hours')
	plt.xticks(rotation='vertical')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	if(admin):
		plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_date/1.png')
		plt.close()
	else:
		plt.savefig('./recognition/static/recognition/img/attendance_graphs/employee_login/1.png')
		plt.close()
	return qs


#used
def hours_vs_employee_given_date(present_qs,time_qs):
	register_matplotlib_converters()
	df_hours=[]
	df_break_hours=[]
	df_username=[]
	qs=present_qs

	for obj in qs:
		user=obj.user
		times_in=time_qs.filter(user=user).filter(out=False)
		times_out=time_qs.filter(user=user).filter(out=True)
		times_all=time_qs.filter(user=user)
		obj.time_in=None
		obj.time_out=None
		obj.hours=0
		obj.hours=0
		if (len(times_in)>0):
			obj.time_in=times_in.first().time
		if (len(times_out)>0):
			obj.time_out=times_out.last().time
		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours
		else:
			obj.hours=0
		(check,break_hourss)= check_validity_times(times_all)
		if check:
			obj.break_hours=break_hourss


		else:
			obj.break_hours=0


		df_hours.append(obj.hours)
		df_username.append(user.username)
		df_break_hours.append(obj.break_hours)
		obj.hours=convert_hours_to_hours_mins(obj.hours)
		obj.break_hours=convert_hours_to_hours_mins(obj.break_hours)





	df = read_frame(qs)
	df['hours']=df_hours
	df['username']=df_username
	df["break_hours"]=df_break_hours


	sns.barplot(data=df,x='username',y='hours')
	plt.xticks(rotation='vertical')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_employee/1.png')
	plt.close()
	return qs


def total_number_employees():
	qs=User.objects.all()
	return (len(qs) -1)
	# -1 to account for admin



def employees_present_today():
	today=datetime.date.today()
	qs=Present.objects.filter(date=today).filter(present=True)
	return len(qs)




#used
def this_week_emp_count_vs_date():
	today=datetime.date.today()
	some_day_last_week=today-datetime.timedelta(days=7)
	monday_of_last_week=some_day_last_week-  datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
	monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
	qs=Present.objects.filter(date__gte=monday_of_this_week).filter(date__lte=today)
	str_dates=[]
	emp_count=[]
	str_dates_all=[]
	emp_cnt_all=[]
	cnt=0





	for obj in qs:
		date=obj.date
		str_dates.append(str(date))
		qs=Present.objects.filter(date=date).filter(present=True)
		emp_count.append(len(qs))


	while(cnt<5):

		date=str(monday_of_this_week+datetime.timedelta(days=cnt))
		cnt+=1
		str_dates_all.append(date)
		if(str_dates.count(date))>0:
			idx=str_dates.index(date)

			emp_cnt_all.append(emp_count[idx])
		else:
			emp_cnt_all.append(0)







	df=pd.DataFrame()
	df["date"]=str_dates_all
	df["Number of employees"]=emp_cnt_all


	sns.lineplot(data=df,x='date',y='Number of employees')
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/this_week/1.png')
	plt.close()






#used
def last_week_emp_count_vs_date():
	today=datetime.date.today()
	some_day_last_week=today-datetime.timedelta(days=7)
	monday_of_last_week=some_day_last_week-  datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
	monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
	qs=Present.objects.filter(date__gte=monday_of_last_week).filter(date__lt=monday_of_this_week)
	str_dates=[]
	emp_count=[]


	str_dates_all=[]
	emp_cnt_all=[]
	cnt=0





	for obj in qs:
		date=obj.date
		str_dates.append(str(date))
		qs=Present.objects.filter(date=date).filter(present=True)
		emp_count.append(len(qs))


	while(cnt<5):

		date=str(monday_of_last_week+datetime.timedelta(days=cnt))
		cnt+=1
		str_dates_all.append(date)
		if(str_dates.count(date))>0:
			idx=str_dates.index(date)

			emp_cnt_all.append(emp_count[idx])

		else:
			emp_cnt_all.append(0)







	df=pd.DataFrame()
	df["date"]=str_dates_all
	df["emp_count"]=emp_cnt_all




	sns.lineplot(data=df,x='date',y='emp_count')
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/last_week/1.png')
	plt.close()








# Create your views here.
def home(request):

	return render(request, 'recognition/home.html')

@login_required
def dashboard(request):
	if(request.user.username=='admin'):
		print("admin")
		return render(request, 'recognition/admin_dashboard.html')
	else:
		print("not admin")

		return render(request,'recognition/employee_dashboard.html')


# Utility function to draw prediction and probability on the frame
def draw_prediction(frame, person_name, probability, x, y, w, h):
    # Draw the bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Prepare the text to display (name + probability)
    text = f"{person_name} ({probability:.2f}%)"

    # Place the text on top of the bounding box
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def mark_your_attendance(request):
    if request.method == 'POST':
        # Load necessary models and data
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_5_face_landmarks.dat')
        svc_save_path = "face_recognition_data/svc.sav"

        with open(svc_save_path, 'rb') as f:
            svc = pickle.load(f)

        fa = FaceAligner(predictor, desiredFaceWidth=96)
        encoder = LabelEncoder()
        encoder.classes_ = np.load('face_recognition_data/classes.npy')

        faces_encodings = np.zeros((1, 128))
        no_of_faces = len(svc.predict_proba(faces_encodings)[0])
        count = {encoder.inverse_transform([i])[0]: 0 for i in range(no_of_faces)}
        present = {encoder.inverse_transform([i])[0]: False for i in range(no_of_faces)}
        log_time = {}
        start = {}

        # Get the frame data from the POST request
        frame_data = request.POST.get('frame')
        try:
            frame_bytes = np.frombuffer(base64.b64decode(frame_data), dtype=np.uint8)
            frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            return JsonResponse({"status": "error", "message": "Failed to decode image."})

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame, 0)

        unknown_detected = True
        highest_prob = 0
        recognized_person_name = None

        for face in faces:
            # Use dlib.rectangle attributes directly
            x, y, w, h = face.left(), face.top(), face.right(), face.bottom()
            face_aligned = fa.align(frame, gray_frame, face)
            (pred, prob) = predict(face_aligned, svc)
            if pred != [-1]:
                person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                recognized_person_name = person_name
                unknown_detected = False
                if count[person_name] == 0:
                    start[person_name] = time.time()
                    count[person_name] += 1

                if count[person_name] == 4 and (time.time() - start[person_name]) > 1.2:
                    count[person_name] = 0
                else:
                    present[person_name] = True
                    log_time[person_name] = datetime.datetime.now()
                    count[person_name] += 1

                # Convert probability to a serializable format
                prob_value = prob[0] if isinstance(prob, np.ndarray) else prob
                highest_prob = max(highest_prob, prob_value)

                # Draw prediction on the frame
                draw_prediction(frame, person_name, prob_value * 100, x, y, w - x, h - y)
            else:
                person_name = "unknown"

        # Check if any known person was detected
        if unknown_detected:
            return JsonResponse({"status": "error", "message": "Person not recognized."})

        # Update attendance in the database
        update_attendance_in_db_in(present)
        if recognized_person_name:
            request.session['recognized_person_name'] = recognized_person_name
        return JsonResponse({"status": "success", "probability": highest_prob * 100})

    return JsonResponse({"status": "error", "message": "Invalid request."})

def mark_your_attendance_out(request):
    if request.method == 'POST':
        # Load necessary models and data
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_5_face_landmarks.dat')
        svc_save_path = "face_recognition_data/svc.sav"

        with open(svc_save_path, 'rb') as f:
            svc = pickle.load(f)

        fa = FaceAligner(predictor, desiredFaceWidth=96)
        encoder = LabelEncoder()
        encoder.classes_ = np.load('face_recognition_data/classes.npy')

        faces_encodings = np.zeros((1, 128))
        no_of_faces = len(svc.predict_proba(faces_encodings)[0])
        count = {encoder.inverse_transform([i])[0]: 0 for i in range(no_of_faces)}
        present = {encoder.inverse_transform([i])[0]: False for i in range(no_of_faces)}
        log_time = {}
        start = {}

        # Get the frame data from the POST request
        frame_data = request.POST.get('frame')
        try:
            frame_bytes = np.frombuffer(base64.b64decode(frame_data), dtype=np.uint8)
            frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            return JsonResponse({"status": "error", "message": "Failed to decode image."})

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame, 0)

        unknown_detected = True
        highest_prob = 0
        recognized_person_name = None

        for face in faces:
            # Use dlib.rectangle attributes directly
            x, y, w, h = face.left(), face.top(), face.right(), face.bottom()
            face_aligned = fa.align(frame, gray_frame, face)
            (pred, prob) = predict(face_aligned, svc)

            if pred != [-1]:
                person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                recognized_person_name = person_name
                unknown_detected = False
                if count[person_name] == 0:
                    start[person_name] = time.time()
                    count[person_name] += 1

                if count[person_name] == 4 and (time.time() - start[person_name]) > 1.2:
                    count[person_name] = 0
                else:
                    present[person_name] = True
                    log_time[person_name] = datetime.datetime.now()
                    count[person_name] += 1

                # Convert probability to a serializable format
                prob_value = prob[0] if isinstance(prob, np.ndarray) else prob
                highest_prob = max(highest_prob, prob_value)

                # Draw prediction on the frame
                draw_prediction(frame, person_name, prob_value * 100, x, y, w - x, h - y)
            else:
                person_name = "unknown"

        # Check if any known person was detected
        if unknown_detected:
            return JsonResponse({"status": "error", "message": "Person not recognized."})

        # Update attendance in the database
        update_attendance_in_db_out(present)
        if recognized_person_name:
            request.session['recognized_person_name'] = recognized_person_name
        return JsonResponse({"status": "success", "probability": highest_prob * 100})

    return JsonResponse({"status": "error", "message": "Invalid request."})

def home(request):
    return render(request, 'recognition/home.html')
# View to mark attendance when checking out

# Views to handle redirects after attendance is marked
def attendance_in_redirect(request):
    recognized_person_name = request.session.get('recognized_person_name', request.user.username)
    messages.success(request, f"User: {recognized_person_name}  Checked-IN successfully. ")
    return HttpResponseRedirect(reverse('home'))

def attendance_out_redirect(request):
    recognized_person_name = request.session.get('recognized_person_name', request.user.username)
    messages.success(request, f"User: {recognized_person_name}  Checked-OUT successfully.")
    return HttpResponseRedirect(reverse('home'))

@login_required
def train(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')

    training_dir = 'face_recognition_data/training_dataset'

    X = []
    y = []
    i = 0

    for person_name in os.listdir(training_dir):
        curr_directory = os.path.join(training_dir, person_name)
        if not os.path.isdir(curr_directory):
            continue

        for imagefile in image_files_in_folder(curr_directory):
            print(str(imagefile))
            image = cv2.imread(imagefile)
            try:
                encoding = face_recognition.face_encodings(image)
                if len(encoding) > 0:  # Check if an encoding was found
                    X.append(encoding[0].tolist())
                    y.append(person_name)
                    i += 1
                else:
                    print(f"No face found in {imagefile}, removing.")
                    os.remove(imagefile)
            except Exception as e:
                print(e)
                print("Error processing", imagefile, "- removed")
                os.remove(imagefile)

    if len(X) == 0 or len(set(y)) < 2:
        messages.warning(request, 'Training data is insufficient. Make sure there are images in at least two different classes.')
        return render(request, "recognition/train.html")

    targets = np.array(y)
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    X1 = np.array(X)
    print("shape: " + str(X1.shape))
    np.save('face_recognition_data/classes.npy', encoder.classes_)

    svc = SVC(kernel='linear', probability=True)
    svc.fit(X1, y)

    svc_save_path = "face_recognition_data/svc.sav"
    with open(svc_save_path, 'wb') as f:
        pickle.dump(svc, f)

    vizualize_Data(X1, targets)

    messages.success(request, 'Training Complete.')

    return render(request, "recognition/train.html")

@login_required
def not_authorised(request):
	return render(request,'recognition/not_authorised.html')



@login_required
def view_attendance_home(request):
	total_num_of_emp=total_number_employees()
	emp_present_today=employees_present_today()
	this_week_emp_count_vs_date()
	last_week_emp_count_vs_date()
	return render(request,"recognition/view_attendance_home.html", {'total_num_of_emp' : total_num_of_emp, 'emp_present_today': emp_present_today})


@login_required
def view_attendance_date(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	qs=None
	time_qs=None
	present_qs=None


	if request.method=='POST':
		form=DateForm(request.POST)
		if form.is_valid():
			date=form.cleaned_data.get('date')
			print("date:"+ str(date))
			time_qs=Time.objects.filter(date=date)
			present_qs=Present.objects.filter(date=date)
			if(len(time_qs)>0 or len(present_qs)>0):
				qs=hours_vs_employee_given_date(present_qs,time_qs)


				return render(request,'recognition/view_attendance_date.html', {'form' : form,'qs' : qs })
			else:
				messages.warning(request, f'No records for selected date.')
				return redirect('view-attendance-date')








	else:


			form=DateForm()
			return render(request,'recognition/view_attendance_date.html', {'form' : form, 'qs' : qs})


@login_required
def view_attendance_employee(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	time_qs=None
	present_qs=None
	qs=None

	if request.method=='POST':
		form=UsernameAndDateForm(request.POST)
		if form.is_valid():
			username=form.cleaned_data.get('username')
			if username_present(username):
				u=User.objects.get(username=username)

				time_qs=Time.objects.filter(user=u)
				present_qs=Present.objects.filter(user=u)
				date_from=form.cleaned_data.get('date_from')
				date_to=form.cleaned_data.get('date_to')

				if date_to < date_from:
					messages.warning(request, f'Invalid date selection.')
					return redirect('view-attendance-employee')
				else:
					time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')

					if (len(time_qs)>0 or len(present_qs)>0):
						qs=hours_vs_date_given_employee(present_qs,time_qs,admin=True)
						return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :qs})
					else:
						#print("inside qs is None")
						messages.warning(request, f'No records for selected duration.')
						return redirect('view-attendance-employee')






			else:
				print("invalid username")
				messages.warning(request, f'No such username found.')
				return redirect('view-attendance-employee')


	else:


			form=UsernameAndDateForm()
			return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :qs})




@login_required
def view_my_attendance_employee_login(request):
	if request.user.username=='admin':
		return redirect('not-authorised')
	qs=None
	time_qs=None
	present_qs=None
	if request.method=='POST':
		form=DateForm_2(request.POST)
		if form.is_valid():
			u=request.user
			time_qs=Time.objects.filter(user=u)
			present_qs=Present.objects.filter(user=u)
			date_from=form.cleaned_data.get('date_from')
			date_to=form.cleaned_data.get('date_to')
			if date_to < date_from:
					messages.warning(request, f'Invalid date selection.')
					return redirect('view-my-attendance-employee-login')
			else:


					time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')

					if (len(time_qs)>0 or len(present_qs)>0):
						qs=hours_vs_date_given_employee(present_qs,time_qs,admin=False)
						return render(request,'recognition/view_my_attendance_employee_login.html', {'form' : form, 'qs' :qs})
					else:

						messages.warning(request, f'No records for selected duration.')
						return redirect('view-my-attendance-employee-login')
	else:


			form=DateForm_2()
			return render(request,'recognition/view_my_attendance_employee_login.html', {'form' : form, 'qs' :qs})