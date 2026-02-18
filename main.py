from fastapi import FastAPI
import joblib
from pydantic import BaseModel

import heapq
import math

app = FastAPI(title="Indoor Localisation API")

model = joblib.load("./models/linear_model2.pkl")


# ----------- Nested models -----------
class MagData(BaseModel):
    x: float
    y: float
    z: float
    ts: int | None = None


class GpsData(BaseModel):
    lat: float
    lon: float
    ts: int | None = None


class PredictRequest(BaseModel):
    floor:int
    room: str | None = None
    mag: MagData
    gps: GpsData


@app.post("/predict")
def predict(req: PredictRequest):
    print("REQUEST RECEIVED:", req)

    X = [[
        req.gps.lat,
        req.gps.lon,
        req.mag.x,
        req.mag.y,
        req.mag.z,
        (req.mag.x**2 + req.mag.y**2 + req.mag.z**2) ** 0.5,  # magnitude
        req.floor
    ]]

    y = model.predict(X)

    return {
        "ok": True,
        "x": float(y[0][0]),
        "y": float(y[0][1]),
    }


@app.post("/hello_world")
def hello_world():
    return {"hello": "world"}



#to run
#python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload


#backend - navigation

EVENTS = {
    "The MERN Playbook": 'A203',
    "Game Development with Godot":'A104',
    "Mind Heist": 'A103',
    "Doomsday:Protocol" : 'A101',
    "Pixel Paradox" : 'A304',
    #Day 2 events
    "Flutter Fast" : 'A104',
    "GenAI Systems" : 'A201',
    "Design and Decode" : 'A101',
    "CrackNCode" : 'A404'
}

ROOMS = {
    "A101": {"floor": 0, "coord": (230, 150)},
    "A102": {"floor": 0, "coord": (390, 140)},
    "A103": {"floor": 0, "coord": (520, 380)},
    "A104": {"floor": 0, "coord": (230, 380)},
    "Washroom": {"floor": 0, "coord": (520, 150)},
    "A201": {"floor": 1, "coord": (230, 150)},
    "A202": {"floor": 1, "coord": (390, 140)},
    "A203": {"floor": 1, "coord": (520, 380)},
    # "A204": {"floor": 1, "coord": (230, 380)}, <--Department office so no navigation
    "Washroom2": {"floor": 1, "coord": (520, 150)},
    "A301": {"floor": 2, "coord": (200,160)}, 
    "A302": {"floor": 2, "coord": (315,160)},
    "A303": {"floor": 2, "coord": (430,160)},
    "A304": {"floor": 2, "coord": (520, 380)},
    "A305": {"floor": 2, "coord": (340,380)},
    "A306": {"floor": 2, "coord": (150,380)},
    "Washroom3": {"floor": 2, "coord": (520, 160)},
    #Third floor is a copy of second floor
    "A401": {"floor": 3, "coord": (200,160)}, 
    "A402": {"floor": 3, "coord": (315,160)},
    "A403": {"floor": 3, "coord": (430,160)},
    "A404": {"floor": 3, "coord": (520, 380)},
    "A405": {"floor": 3, "coord": (340,380)},
    "A406": {"floor": 3, "coord": (150,380)},
    "Washroom4": {"floor": 3, "coord": (520, 160)},
}

STAIRS = {
    0: [(100, 255), (520, 255)],
    1: [(100, 255), (520, 255)],
    2: [(100, 250), (520, 250)],
    3: [(100, 250), (520, 250)],
}

GROUND_FLOOR = {
    "nodes":[
    (230, 150), # A101
    (230, 225), # A101 entrance C0P1
    (390, 225), # A102 entrance C0P2
    (390, 140), # A102
    (390, 290), # C0P3
    (230, 290), # A104 entrance C0P4
    (230, 380), # A104
    (520, 290),  # A103 entrance C0P5 
    (520, 380),  # A103 
    (520, 225),  # washroom entrance C0P6
    (520, 150), # washroom 
    (100, 225), # left top corner C0P7
    (100, 290), # left bottom corner C0P8 
    (100, 255),   # left stairs
    (520, 255),   # right stairs
    ],
    "edges" :{
        (100, 225): [(100, 290),(230, 225),(100, 255)], #left top corner 
        (100, 290): [(100, 225),(230, 290),(100, 255)], #left bottom corner
        (230, 225): [(230, 150),(230, 290),(390, 225),(100, 225)], #A101 entrance
        (230, 290): [(100, 290),(230, 380),(230, 225),(390, 290)], #A104 entrance 
        (520, 225): [(520, 150),(520, 290),(390, 225),(520, 255)], #washroom entrance
        (520, 150): [(520, 225)], #washroom
        (390, 225): [(390, 140),(230, 225),(520, 225),(390, 290)],#A102 enterance
        (520, 290): [(520, 380),(520, 225),(390, 290),(520, 255)], #A103 entrance
        (390, 290): [(230, 290),(520, 290),(390, 225)], #C0P3
        (230, 150): [(230, 225)], #A101
        (390, 140): [(390, 225)], #A102
        (230, 380): [(230, 290)], #A104
        (520, 380): [(520, 290)], #A103
        (100, 255): [(100, 225), (100, 290)], # left stairs
        (520, 255): [(520, 225), (520, 290)],  # right stairs

    }
}

FIRST_FLOOR = {
    "nodes":[
    (230, 150), # A101
    (230, 225), # A101 entrance C0P1
    (390, 225), # A102 entrance C0P2
    (390, 140), # A102
    (390, 290), # C0P3
    (230, 290), # A104 entrance C0P4
    (230, 380), # A104
    (520, 290),  # A103 entrance C0P5 
    (520, 380),  # A103 
    (520, 225),  # washroom entrance C0P6
    (520, 150), # washroom 
    (100, 225), # left top corner C0P7
    (100, 290), # left bottom corner C0P8 
    (100, 255),   # left stairs
    (520, 255),   # right stairs
    ],
    "edges" :{
        (100, 225): [(100, 290),(230, 225),(100, 255)], #left top corner 
        (100, 290): [(100, 225),(230, 290),(100, 255)], #left bottom corner
        (230, 225): [(230, 150),(390, 225),(100, 225)], #A101 entrance
        (230, 290): [(100, 290),(230, 380),(390, 290)], #A104 entrance 
        (520, 225): [(520, 150),(520, 290),(390, 225),(520, 255)], #washroom entrance
        (520, 150): [(520, 225)], #washroom
        (390, 225): [(390, 140),(230, 225),(520, 225)],#A102 enterance
        (520, 290): [(520, 380),(520, 225),(390, 290),(520, 255)], #A103 entrance
        (390, 290): [(230, 290),(520, 290)], #C0P3
        (230, 150): [(230, 225)], #A101
        (390, 140): [(390, 225)], #A102
        (230, 380): [(230, 290)], #A104
        (520, 380): [(520, 290)], #A103
        (100, 255): [(100, 225), (100, 290)], # left stairs
        (520, 255): [(520, 225), (520, 290)],  # right stairs

    }
}

TOP_FLOOR = {
    "nodes":[
    (100,210) ,#top left corner
    (100,250), #left stairs
    (100,290), #bottom right corner
    (200,210), #A301 entrance
    (200,160), #A301
    (315,210), #A302 entrance
    (315,160), #A302
    (430,210), #A303 entrance
    (430,160), #A303
    (520,210), #washroom entrance
    (520,160), #washroom
    (520,290), #A304 entrance
    (520,380), #A304
    (340,290), #A305 entrance
    (340,380), #A305
    (150,290), #A306 entrance
    (150,380), #A306
    (520,250), #right stairs
    ],

    "edges" :{
        (100,210) : [(200,210),(100,250)], #top left corner
        (100,250):[(100,210),(100,290)], #left stairs
        (100,290):[(100,250),(150,290)], #bottom right corner
        (200,210):[(200,160),(100,210),(315,210)], #A301 entrance
        (200,160): [(200,210)], #A301
        (315,210):[(315,160),(200,210),(430,210)], #A302 entrance
        (315,160):[(315,210)], #A302
        (430,210):[(430,160),(315,210),(520,210)], #A303 entrance
        (430,160):[(430,210)], #A303
        (520,210):[(520,160),(430,210),(520,250)], # washroom entrance
        (520,160):[(520,210)], #washroom
        (520,250):[(520,210),(520,290)], #right stairs
        (520,290):[(520,250),(340,290),(520,380)], #A304 entrance
        (520,380):[(520,290)], #A304
        (340,290):[(520,290),(150,290),(340,380)], #A305 entrance
        (340,380):[(340,290)], #A305
        (150,290):[(100,290),(340,290),(150,380)], #A306 entrance
        (150,380):[(150,290)], #A306
    }
}

FLOOR_GRAPHS = {
    0: GROUND_FLOOR,
    1: FIRST_FLOOR,
    2: TOP_FLOOR,
    3: TOP_FLOOR,
}





def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def a_star(start, goal, nodes, edges):

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}

    g_score = {node: float("inf") for node in nodes}
    g_score[start] = 0

    f_score = {node: float("inf") for node in nodes}
    f_score[start] = euclidean(start, goal)

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in edges.get(current, []):  # 🔥 CHANGED safer access

            tentative_g = g_score[current] + euclidean(current, neighbor)

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + euclidean(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []


@app.post("/navigate")
def navigate(req: PredictRequest):

    # Predict Current Location
    magnitude = math.sqrt(req.mag.x**2 + req.mag.y**2 + req.mag.z**2)

    X = [[
        req.gps.lat,
        req.gps.lon,
        req.mag.x,
        req.mag.y,
        req.mag.z,
        magnitude,
        req.floor
    ]]

    y = model.predict(X)

    start = (
        round(float(y[0][0])),
        round(float(y[0][1]))
    )

    # Get Goal 
    if not req.room:
        return {"error": "No destination room provided"}

    # if req.room not in ROOMS:
    #     return {"error": "Room not found"}

    if req.room not in EVENTS:
        return {"error": "Event not found"}
    destination = ROOMS[EVENTS[req.room]]
    dest_floor = destination["floor"]
    dest_coord = destination["coord"]

    # Select floor graph
    graph = FLOOR_GRAPHS.get(req.floor, GROUND_FLOOR)
    nodes = graph["nodes"]
    edges = graph["edges"]


    # MULTI-FLOOR LOGIC meghs

    if req.floor == dest_floor:
        goal = dest_coord
        phase = "room"
    else:
        stairs_on_floor = STAIRS.get(req.floor, [])
        goal = min(stairs_on_floor, key=lambda s: euclidean(start, s))
        phase = "stairs"

    # FIND NEAREST GRAPH NODE

    start_node = min(nodes, key=lambda n: euclidean(n, start))
    path = a_star(start_node, goal, nodes, edges)

    print("START RAW:", start)
    print("START NODE:", start_node)
    print("GOAL:", goal)
    print("PATH:", path)

    return {
        "ok": True,
        "current": start,
        "goal": goal,
        "path": path,
        "phase": phase,
        "target_floor": dest_floor
    }