import firebase_admin
from firebase_admin import credentials, db
import csv
import math

DATABASE_URL = 'https://indoornavigation-ca342-default-rtdb.asia-southeast1.firebasedatabase.app/'
CREDENTIALS_FILE = 'indoornavigation-ca342-firebase-adminsdk-fbsvc-53b6bc9add.json'
DATA_PATH = 'taps'

# --- Initialize Firebase ---
cred = credentials.Certificate(CREDENTIALS_FILE)
firebase_admin.initialize_app(cred, {
    'databaseURL': DATABASE_URL
})
ref = db.reference(DATA_PATH)
data = ref.get()

if not data:
    print("No data found.")
    exit()

val = []

f1 = open('data.csv', 'w')
writer = csv.writer(f1)

hearder = ['mageX', 'imageY', 'magX', 'magY', 'magZ', 'mag_val', 'lat', 'lon', 'ts', 'floor']
writer.writerow(hearder)

for i, (key, val) in enumerate(data.items()):
        val1 = list()
        val1.append(float(val['imageX']))
        val1.append(float(val['imageY']))
        val1.append(float(val['x']))
        val1.append(float(val['y']))
        val1.append(float(val['z']))
        val1.append(math.sqrt(val1[2]**2 + val1[3]**2 + val1[4]**2))

        val1.append(val['latitude'])
        val1.append(val['longitude'])
        val1.append(float(val['ts']))
        val1.append(int(val['floor']))

        writer.writerow(val1)

