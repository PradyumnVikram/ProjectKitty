import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import savgol_filter, medfilt
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

file_path = input('\033[31m[LOG]: Enter path to cropped video file: \033[0m')

print('[LOG]: Loading models')

with open('models/run_up.pkl', 'rb') as f:
    run_model = pickle.load(f)
with open('scalers/run_up.pkl', 'rb') as f:
    run_scaler = pickle.load(f)

with open('models/take_off.pkl', 'rb') as f:
    take_model = pickle.load(f)
with open('scalers/take_off.pkl', 'rb') as f:
    take_scaler = pickle.load(f)

with open('models/flight.pkl', 'rb') as f:
    flight_model = pickle.load(f)
with open('scalers/flight.pkl', 'rb') as f:
    flight_scaler = pickle.load(f)

with open('models/landing.pkl', 'rb') as f:
    landing_model = pickle.load(f)
with open('scalers/landing.pkl', 'rb') as f:
    landing_scaler = pickle.load(f)


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
estimator = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)

def measure_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return angle

def trunk_angle_calc(a, b):
    a = np.array(a)
    b = np.array(b)
    ba = (a-b)
    cosine_angle = np.dot(ba, np.array([0,0,1]))/(np.linalg.norm(ba)+1e-6)
    return np.arccos(np.clip(cosine_angle, -1.0, 1.0))

def extract_features_dataset(file_path):    
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    pos = (50, 100)
    frames = []

    hip_vel_y = []
    hip_vel_x = []
    hip_y = []
    hip_x = []
    lower_foot = []
    left_knee_angle = []
    right_knee_angle = []
    hip_leg_angle = []
    trunk_angle = []
    
    rs = 12
    ls = 11
    lh = 23
    rh = 24
    lk = 25
    rk = 26
    la = 27
    ra = 28
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = estimator.process(frame)
        if results.pose_landmarks:
            if not results.pose_landmarks.landmark[ra].visibility>0.5:
                continue
            landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
           
            hip_y.append((landmarks[lh][1]+landmarks[rh][1])/2)
            hip_x.append((landmarks[lh][0]+landmarks[rh][0])/2)
            lower_foot.append(min(landmarks[la][1], landmarks[ra][1]))
            left_knee_angle.append(measure_angle(landmarks[lh], landmarks[lk], landmarks[la]))
            right_knee_angle.append(measure_angle(landmarks[rh], landmarks[rk], landmarks[ra]))
            hip_leg_angle.append(measure_angle(landmarks[rs], landmarks[rh], landmarks[rk]))
    
            mid_shoulder = (np.array(landmarks[ls]) + np.array(landmarks[rs])) / 2
            mid_hip = (np.array(landmarks[lh]) + np.array(landmarks[rh])) / 2
        
            trunk_angle.append(trunk_angle_calc(mid_shoulder, mid_hip))
    
            if len(hip_y)>1:
                hip_vel_y.append((hip_y[-1] - hip_y[-2])*fps)
            else:
                hip_vel_y.append(0)
    
            if len(hip_x)>1:
                hip_vel_x.append((hip_x[-1] - hip_x[-2])*fps)
            else:
                hip_vel_x.append(0)
            frames.append(frame)
        #cv2.putText(frame, str(count), pos, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3, cv2.LINE_AA)
        #cv2.imshow('', frame)
        #if cv2.waitKey(25) & 0xFF == ord('q'):
        #    break
    cap.release()
    cv2.destroyAllWindows()
    
    df = np.column_stack([hip_vel_y, hip_vel_x, hip_y, trunk_angle, left_knee_angle, right_knee_angle, hip_leg_angle, lower_foot])

    return df

def cluster_df(df, skip_frames=25):

    final_df = []
    
    features_smooth = savgol_filter(df, window_length=7, polyorder=3, axis=0)
    features_norm = (features_smooth - features_smooth.mean(axis=0)) / features_smooth.std(axis=0)
    
    kmeans = KMeans(n_clusters=4, random_state=0).fit(features_norm)
    labels = kmeans.labels_
    
    labels_smooth = medfilt(labels, kernel_size=7)
    
    labels_smooth = labels_smooth[skip_frames:]
    df = df[skip_frames:]
    final_df = []
    final_frames = []
    phase_order = []
    count  = 0
    
    for i in labels_smooth:
        if i in phase_order:
            continue
        else:
            phase_order.append(i)
    
    smoothen_labels = np.array([0 for i in range(len(labels_smooth))])
    
    for phase in phase_order:
        indices = np.where(labels_smooth == phase)[0]
        if len(indices) == 0:
            continue
        start = indices[0]
        end = indices[-1]
        smoothen_labels[start:end+1] = phase

    for idx, phase in enumerate(phase_order):
        final_frames = []
        while smoothen_labels[count] == phase:
            final_frames.append(df[count])
            
            count += 1
            if count == len(smoothen_labels):
                break
        final_df.append(final_frames)

    return final_df


def predict(to_process, i, model, scaler):

    to_process[i] = np.array(to_process[i])
    coord = scaler.transform([[np.mean(to_process[i][:, 1]), np.mean(to_process[i][:, 3])]])
    prediction = (model.predict(coord))[0]
    coord = coord[0]
    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    inside = model.decision_function(grid_points) >= 0
    contour_center = grid_points[inside].mean(axis=0)
    feature_to_fix = np.argmax(abs(contour_center - coord))
    action = (coord - contour_center)[feature_to_fix] > 0

    return prediction, feature_to_fix, action


print('[LOG]: Extracting features')
df = extract_features_dataset(file_path)
to_process = cluster_df(df)


feature_descript = [
                    {0:['Decrease Speed', 'Increase Speed'],
                     1:['Decrease back angle with Vertical', 'Increase back angle with verticle']},
                    {0:['Decrease knee flexion','Increase knee flexion'],
                     1:['Decline back more', 'Incline back to the ground more']},
                    {0:['Maintain lesser hip-leg angle', 'Maintain greater hip-leg angle'],
                     1:['Bend your back more','Keep your back straighter']},
                    {0:['Keep chest away from legs', 'Push chest towards legs'],
                     1:['Keep legs slightly more angles', 'Keep legs straighter']}]


if to_process[0]:
    prediction, feature, action = predict(to_process, 0, run_model, run_scaler)
    if prediction == -1:
        print(f'\033[31m[ANOMALY - Run Up]: {feature_descript[0][feature][action]}\033[0m')
    else:
        print('\033[32m[LOG]: No correction suggested\033[0m')
else:
    print('[LOG]: No features to analyse')

if to_process[1]:
    prediction, feature, action = predict(to_process, 1, take_model, take_scaler)
    if prediction == -1:
        print(f'\033[31m[ANOMALY - Take Off]: {feature_descript[1][feature][action]}\033[0m')
    else:
        print('\033[32m[LOG]: No correction suggested\033[0m')
else:
    print('[LOG]: No features to analyse')

if to_process[2]:    
    prediction, feature, action = predict(to_process, 2, flight_model, flight_scaler)
    if prediction == -1:
        print(f'\033[31m[ANOMALY - Flight]: {feature_descript[2][feature][action]}\033[0m')
    else:
        print('\033[32m[LOG]: No correction suggested\033[0m')
else:
    print('[LOG]: No features to analyse')

if to_process[3]:
    prediction, feature, action = predict(to_process, 3, landing_model, landing_scaler)
    if prediction == -1:
        print(f'\033[31m[ANOMALY - Landing]: {feature_descript[3][feature][action]}\033[0m')
    else:
        print('\033[32m[LOG]: No correction suggested\033[0m')
else:
    print('[LOG]: No features to analyse')









